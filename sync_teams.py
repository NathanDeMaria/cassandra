"""Keep the team registry covering the games cassandra replays.

    python sync_teams.py --league ncaafb              # pull, reconcile, record
    python sync_teams.py --league ncaafb --check      # report, record nothing
    python sync_teams.py --league ncaafb --yes        # don't stop to ask

`call_it_what_you_want` owns team identity, including where discoveries get
written: `record` appends them to that package's own staging file and they
take effect for every caller immediately. This script only decides *what* to
record. One pass does three things:

1. *Pull.* ESPN's scoreboard carries a team id next to every name, and
   `endgame.espn_games._parse_competitor` keeps the name and drops the id --
   so the ids have to come from ESPN directly rather than from the seasons in
   s3. The names come from the same `displayName` field endgame built the
   seasons from, so what lands here lines up with cassandra's team strings
   exactly. Each observation is tagged with the league it was seen in,
   because ESPN calls the same school "UNLV Rebels" in football and "UNLV
   Lady Rebels" in women's basketball in the same season.

2. *Reconcile.* Whatever the pull didn't explain gets checked against the
   seasons: which names in real games the registry still can't place, which
   ones two teams both answer to, and which pairs of names look like one
   school under two spellings.

3. *Ask.* Everything in that third category is a guess from heuristics, and
   putting two different schools on one id quietly destroys both teams'
   ratings in a way no metric will flag. So each one is printed with its
   evidence and waits for an answer. Rejections are remembered, so a second
   run doesn't ask again about the same wrong guess.

Nothing is written into cassandra. What gets recorded lands in the package's
local staging file (`ciwyw where`), and

    ciwyw show --output call_it_what_you_want/data/ncaa.csv

is how it becomes a diff to review in that repo. One file covers all of
college sports, since ESPN numbers teams per organization and every college
sport shares those ids.
"""

import asyncio
import json
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import aiohttp
import fire
from call_it_what_you_want import (
    COLUMNS,
    ESPN,
    AmbiguousNameError,
    NewRecordWarning,
    NewTeamWarning,
    NoNamesError,
    Team,
    TeamName,
    Teams,
    UnknownTeamError,
    default_teams,
    local_path,
    normalize,
    record,
)

# Not exported: `record` is deliberately additive and has nowhere to put a
# `same_as`, which is the one thing here that isn't a new observation. This
# is the package's own append, used rather than a second CSV writer.
from call_it_what_you_want.local import append_local
from endgame.ncaabb.gender import NcaabbGender
from endgame.ncaabb.ncaabb import (
    NCAABB_SCOREBOARD,
    POST_SEASON_START,
    POSTSEASON_GROUPS,
    REGULAR_SEASON_END,
    REGULAR_SEASON_START,
    SEASON_END,
    NcaabbGroup,
)
from endgame.ncaafb import BASE_URL as NCAAFB_URL
from endgame.ncaafb import N_REGULAR_WEEKS
from endgame.types import NcaaFbGroup, SeasonType
from endgame_aws import Config

from cassandra.alias_candidates import (
    NameCandidate,
    TeamActivity,
    check_registry,
    propose_names,
    team_activity,
    unresolved_names,
)
from cassandra.aliases import TeamNamer, namespace_for, registry_league
from cassandra.save_predictions import read_all_seasons

# ESPN's scoreboard defaults to 25 events, which would silently drop most of
# a week. This is also the cap a date window is checked against -- a response
# with this many events was truncated, and `_fetch` splits it.
_LIMIT = 400
# Days per basketball request. A week of D-I is comfortably under the cap;
# the cap is enforced regardless, so this is a request-count tradeoff and
# not a correctness one.
_WINDOW_DAYS = 7
# Polite, and enough to finish ~1400 requests in a couple of minutes.
_CONCURRENCY = 8
_TIMEOUT = aiohttp.ClientTimeout(total=60)

# One observation, and the id it was seen under.
Observation = tuple[str, TeamName]


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------


def _ncaafb_requests(first_year: int, last_year: int) -> Iterator[tuple[int, dict]]:
    """The grid `endgame.ncaafb.get_season` uses, so coverage matches the seasons."""
    for year in range(first_year, last_year + 1):
        for group in NcaaFbGroup:
            for season_type in SeasonType:
                weeks = (
                    range(1, N_REGULAR_WEEKS + 1)
                    if season_type is SeasonType.regular
                    else [1]
                )
                for week in weeks:
                    yield year, {
                        "dates": year,
                        "week": week,
                        "seasontype": season_type.value,
                        "groups": group.value,
                        "limit": _LIMIT,
                    }


def _windows(start: date, end: date, days: int = _WINDOW_DAYS) -> Iterator[str]:
    """`start`..`end` as `YYYYMMDD-YYYYMMDD` chunks the scoreboard accepts.

    A day at a time is what `endgame.ncaabb` does, and it's ~400 requests a
    season per gender. Ranges cut that by a factor of seven -- at the cost
    of the event cap, which `_fetch` handles by splitting a window that hits
    it rather than by trusting this number.
    """
    while start <= end:
        stop = min(start + timedelta(days=days - 1), end)
        yield f"{start:%Y%m%d}-{stop:%Y%m%d}"
        start = stop + timedelta(days=1)


def _ncaabb_requests(first_year: int, last_year: int) -> Iterator[tuple[int, dict]]:
    """The grid `endgame.ncaabb.get_ncaabb_season` uses, in date windows.

    A basketball season starts in November and is named for the year it
    started, which is how the seasons in s3 are numbered -- so the year a
    request belongs to is not the year in its dates.

    The postseason is a set of *groups* rather than a season type, and the
    tournaments overlap the end of the regular season, so both halves get
    walked.
    """
    today = date.today()
    for year in range(first_year, last_year + 1):
        regular_end = min(date(year + 1, *REGULAR_SEASON_END), today)
        for window in _windows(date(year, *REGULAR_SEASON_START), regular_end):
            yield year, {
                "dates": window,
                "groups": NcaabbGroup.d1.value,
                "limit": _LIMIT,
            }
        post_end = min(date(year + 1, *SEASON_END), today)
        for group in POSTSEASON_GROUPS:
            for window in _windows(date(year + 1, *POST_SEASON_START), post_end):
                yield year, {
                    "dates": window,
                    "groups": group.value,
                    "limit": _LIMIT,
                }


class _Source(NamedTuple):
    """Where a league's names come from, and how far back they go."""

    url: str
    requests: Callable[[int, int], Iterator[tuple[int, dict]]]
    first_year: int


# A league is pullable when there's a source for it. The first years match
# what endgame fetches, so a pull covers the seasons a replay will see.
_GRIDS = {
    "ncaafb": _Source(NCAAFB_URL, _ncaafb_requests, 1999),
    "mens": _Source(
        NCAABB_SCOREBOARD.format(NcaabbGender.mens.name), _ncaabb_requests, 2001
    ),
    "womens": _Source(
        NCAABB_SCOREBOARD.format(NcaabbGender.womens.name), _ncaabb_requests, 2001
    ),
}


def _placeholder(espn_id: str) -> bool:
    """ESPN's stand-in for an opponent that isn't announced yet.

    It comes back as id -1 or -2 named "TBD". Those never reach the seasons,
    which keep only completed games, but a registry with a team called TBD in
    it is nonsense.
    """
    return str(espn_id).lstrip("-").isdigit() and int(espn_id) < 0


def _observations(payload: dict, year: int, league: str) -> Iterator[Observation]:
    for event in payload.get("events") or []:
        for competition in event.get("competitions") or []:
            for competitor in competition.get("competitors") or []:
                team = competitor.get("team") or {}
                espn_id, name = team.get("id"), team.get("displayName")
                if espn_id and name and not _placeholder(espn_id):
                    # `year` off the request rather than the payload: ESPN
                    # dates January bowl games into the next calendar year,
                    # and a season is the year it started, which is how the
                    # seasons in s3 are numbered.
                    yield (
                        str(espn_id),
                        TeamName(name=name, year=year, source=ESPN, league=league),
                    )


def _split_window(dates: object) -> tuple[str, str] | None:
    """Halve a `YYYYMMDD-YYYYMMDD` window, or None if it isn't one.

    Takes whatever was in the request params, since football's `dates` is a
    year and has no window to split.
    """
    if not isinstance(dates, str) or "-" not in dates:
        return None
    start, _, stop = dates.partition("-")
    first = date.fromisoformat(start)
    last = date.fromisoformat(stop)
    if first >= last:
        return None
    middle = first + (last - first) / 2
    return (
        f"{first:%Y%m%d}-{middle:%Y%m%d}",
        f"{middle + timedelta(days=1):%Y%m%d}-{last:%Y%m%d}",
    )


async def _fetch(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    year: int,
    params: dict,
    league: str,
) -> list[Observation]:
    async with semaphore:
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"  skipped {year} {params}: {e!r}")
            return []

    observations = list(_observations(payload, year, league))
    if len(payload.get("events") or []) < _LIMIT:
        return observations

    # The window hit the event cap, so ESPN truncated it and the tail of it
    # is missing. Halve and retry rather than trust a window size: a busy
    # week silently losing its last few days is exactly the kind of gap that
    # shows up months later as a team with no name on record.
    halves = _split_window(params.get("dates"))
    if halves is None:
        print(f"  WARNING: {params} hit the {_LIMIT}-event cap and can't be split")
        return observations
    for window in halves:
        observations += await _fetch(
            session, semaphore, url, year, {**params, "dates": window}, league
        )
    return observations


async def _pull(
    league: str, registry_league: str, first_year: int, last_year: int
) -> set[Observation]:
    """Every (team, name, season) ESPN will admit to for this league.

    `registry_league` rather than `league` on the observations: these rows
    are the registry's, and it has its own name for each sport.
    """
    source = _GRIDS[league]
    requests = list(source.requests(first_year, last_year))
    print(
        f"Pulling {league} from ESPN as {registry_league}: "
        f"{len(requests)} requests, {first_year}-{last_year}"
    )

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    seen: set[Observation] = set()
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        tasks = [
            _fetch(session, semaphore, source.url, year, params, registry_league)
            for year, params in requests
        ]
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            seen.update(await task)
            if done % 200 == 0:
                print(f"  {done}/{len(requests)} requests, {len(seen)} observations")
    return seen


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


class Unrecorded(NamedTuple):
    """What a pull turned up that the registry doesn't have yet."""

    observations: list[Observation]
    # Ids with no team at all, which are worth more attention than a new
    # name: either a school nobody had looked at, or a duplicate ESPN record
    # for a team already on file, which wants a `same_as` rather than a team
    # of its own.
    new_ids: list[str]


def _unrecorded(teams: Teams, observations: set[Observation]) -> Unrecorded:
    """The observations the registry doesn't already have.

    `record` is a no-op for one it does, so this is about saying how many are
    new -- and about not making 15,000 calls to find that out.
    """
    new: list[Observation] = []
    new_ids: set[str] = set()
    for espn_id, name in sorted(observations, key=lambda o: (int(o[0]), o[1].year)):
        try:
            team: Team | None = teams.by_espn_id(espn_id)
        except UnknownTeamError:
            team = None
            new_ids.add(espn_id)
        if team is None or name not in team.names:
            new.append((espn_id, name))
    return Unrecorded(new, sorted(new_ids, key=int))


def _record_all(observations: Sequence[Observation], namespace: str) -> None:
    """Hand each new observation to the package.

    `record` warns per new observation, which is right for an application
    that stumbles on one and wrong for a scrape that expects hundreds. The
    caller has already counted what's new, so the warnings are suppressed
    here rather than printed a thousand times.
    """
    if not observations:
        print("  nothing new to record")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NewRecordWarning)
        for espn_id, name in observations:
            record(
                espn_id,
                name.name,
                name.year,
                source=name.source,
                league=name.league,
                namespace=namespace,
            )
    print(f"  recorded {len(observations)} observation(s) to {local_path(namespace)}")


def _record_same_as(namespace: str, duplicate: Team, canonical: Team) -> None:
    """Point one ESPN record at another, as a row.

    `record` can't express this -- it only ever adds an observation, and a
    `same_as` is a claim about two ids rather than about a name. One row
    carrying it is enough for `teams_from_csv` to pool them, and it stays
    additive, so nothing already on file is edited.
    """
    name = duplicate.names[0]
    append_local(
        namespace,
        COLUMNS,
        {
            "espn_id": duplicate.espn_id,
            "name": name.name,
            "year": str(name.year),
            "source": name.source,
            "league": name.league or "",
            "same_as": canonical.espn_id,
        },
    )


# --------------------------------------------------------------------------
# The seasons
# --------------------------------------------------------------------------


async def _read_activity(league: str) -> dict[str, TeamActivity] | None:
    """What the seasons in s3 actually contain, or None if unreachable.

    A pull is useful on its own, so an unreadable bucket downgrades the run to
    "pull only" rather than failing it -- there's no point refusing to record
    what ESPN just said because the seasons couldn't be checked against it.
    """
    try:
        bucket = Config.init_from_file().bucket
        print(f"Loading {league} seasons from s3://{bucket}")
        seasons = [s async for s in read_all_seasons(league, bucket)]
    except Exception as e:
        print(f"  can't read seasons ({e!r}); skipping reconciliation")
        return None
    if not seasons:
        print("  no seasons found; skipping reconciliation")
        return None
    activity = team_activity(seasons)
    print(f"  {len(seasons)} seasons, {len(activity)} distinct team names")
    return activity


# --------------------------------------------------------------------------
# Asking
# --------------------------------------------------------------------------


class Decisions:
    """Answers already given, so a rerun doesn't ask the same question twice.

    Only the noes are kept. An accepted name needs no record: it's in the
    registry afterwards, which is the same answer in a more useful form.

    Kept next to the package's staging file rather than in cassandra: it's
    the same kind of thing -- local state about local discoveries -- and it
    should not follow the repo to another machine.
    """

    def __init__(self, path: Path, interactive: bool) -> None:
        self._path = path
        self.interactive = interactive
        stored = _read_json(path)
        self.rejected: set[tuple[str, str]] = {
            (name, known) for name, known in stored.get("rejected", [])
        }
        self.distinct_ids: set[tuple[str, str]] = {
            (left, right) for left, right in stored.get("distinct_ids", [])
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "rejected": sorted(list(p) for p in self.rejected),
                    "distinct_ids": sorted(list(p) for p in self.distinct_ids),
                },
                indent=2,
            )
            + "\n"
        )

    def ask(self, question: str) -> str:
        """`y`, `n`, or `q`. Non-interactive runs answer `n` to everything."""
        if not self.interactive:
            return "n"
        while True:
            answer = input(f"{question} [y/n/q] ").strip().lower()
            if answer in ("y", "n", "q"):
                return answer
            print("  please answer y, n, or q")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _evidence(candidate: NameCandidate, namer: TeamNamer) -> str:
    return "\n".join(
        [
            f"  {candidate.name!r}",
            f"      seen {candidate.name_years[0]}-{candidate.name_years[1]}, "
            f"{candidate.name_games} games",
            f"  {candidate.known!r}  (ESPN id {namer.espn_id(candidate.known)})",
            f"      seen {candidate.known_years[0]}-{candidate.known_years[1]}, "
            f"{candidate.known_games} games",
            f"  similarity {candidate.similarity:.2f}, "
            f"{candidate.opponent_jaccard:.0%} of opponents shared across the gap, "
            f"{candidate.gap_years} year(s) between them",
        ]
    )


def _review_names(
    candidates: Sequence[NameCandidate],
    activity: dict[str, TeamActivity],
    namer: TeamNamer,
    decisions: Decisions,
    registry_league: str,
) -> list[Observation]:
    """Ask about each proposed name; return observations for the accepted ones.

    `registry_league` is the registry's name for the sport, not cassandra's.
    """
    accepted: list[Observation] = []
    pending = [c for c in candidates if (c.name, c.known) not in decisions.rejected]
    if not pending:
        return accepted

    print(f"\n{len(pending)} name(s) to rule on:\n")
    for i, candidate in enumerate(pending, start=1):
        espn_id = namer.espn_id(candidate.known)
        if espn_id is None:
            # Neither side has an id, so there's no team to attach a name to.
            # That's a team to create upstream, not a name to record.
            print(f"[{i}/{len(pending)}] no ESPN id for {candidate.known!r}; skipped")
            continue
        print(f"[{i}/{len(pending)}] same team?")
        print(_evidence(candidate, namer))
        answer = decisions.ask("  record as the same team?")
        if answer == "q":
            print("  stopping; everything answered so far is kept")
            break
        if answer == "n":
            decisions.rejected.add((candidate.name, candidate.known))
            continue
        # One observation per season the name was actually seen, which is
        # what `name_in` reads to answer what a team was called back then.
        # Squashing them to one year would answer it wrong.
        accepted.extend(
            (espn_id, TeamName(candidate.name, year, ESPN, registry_league))
            for year in sorted(activity[candidate.name].years)
        )
    return accepted


def _current(team: Team, league: str) -> str | None:
    try:
        return team.current_name(league=league)
    except (NoNamesError, AmbiguousNameError):
        return None


def _duplicate_ids(teams: Teams, league: str) -> list[tuple[Team, Team]]:
    """Teams whose current names are the same string under different ids.

    ESPN occasionally files one school under two records. That's what the
    registry's `same_as` column is for, and it can't be guessed at from
    names alone -- two schools genuinely sharing a name is normal.
    """
    by_name: dict[str, list[Team]] = {}
    for team in teams:
        name = _current(team, league)
        if name is not None:
            by_name.setdefault(normalize(name), []).append(team)
    return [
        (group[0], other)
        for group in by_name.values()
        if len(group) > 1
        for other in group[1:]
    ]


def _review_duplicate_ids(
    teams: Teams, league: str, namespace: str, decisions: Decisions
) -> int:
    """Ask whether two ESPN records are one school, and record `same_as`."""
    pairs = [
        (left, right)
        for left, right in _duplicate_ids(teams, league)
        if (left.espn_id, right.espn_id) not in decisions.distinct_ids
    ]
    if not pairs:
        return 0

    print(f"\n{len(pairs)} pair(s) of ESPN records with the same name:\n")
    merged = 0
    for left, right in pairs:
        print(f"  {_current(left, league)!r}")
        for team in (left, right):
            years = sorted({n.year for n in team.names})
            print(
                f"    id {team.espn_id}: {years[0]}-{years[-1]}, "
                f"{len(team.names)} observations"
            )
        answer = decisions.ask("  one school under two ESPN records?")
        if answer == "q":
            break
        if answer == "n":
            decisions.distinct_ids.add((left.espn_id, right.espn_id))
            continue
        # The older record becomes the duplicate: the newer id is the one
        # ESPN is still filing games under.
        older, newer = sorted((left, right), key=lambda t: max(n.year for n in t.names))
        _record_same_as(namespace, older, newer)
        merged += 1
    return merged


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _decisions_file(namespace: str) -> Path:
    return local_path(namespace).with_name(f"{namespace}_decisions.json")


def _sync(
    league: str,
    first_year: int | None,
    last_year: int,
    pull: bool,
    check: bool,
    interactive: bool,
    min_similarity: float,
    max_gap_years: int,
) -> None:
    namespace = namespace_for(league)
    reg_league = registry_league(league)
    if namespace is None or reg_league is None:
        raise ValueError(
            f"League {league!r} has no ESPN id namespace in cassandra.aliases. "
            "The NFL numbers its teams separately from college and collides "
            "with it, so it needs its own registry rather than this one."
        )
    print(f"{len(default_teams(namespace))} teams in the registry")

    if pull:
        if league not in _GRIDS:
            raise ValueError(
                f"No ESPN request grid for league {league!r}. "
                f"Pullable: {', '.join(sorted(_GRIDS))}. Pass --pull=False to "
                "reconcile against the seasons using what's already recorded."
            )
        if first_year is None:
            first_year = _GRIDS[league].first_year
        pulled = asyncio.run(_pull(league, reg_league, first_year, last_year))
        unrecorded = _unrecorded(default_teams(namespace), pulled)
        print(
            f"  {len(pulled)} observation(s) pulled, "
            f"{len(unrecorded.observations)} not on file"
        )
        if unrecorded.new_ids:
            print(
                f"  {len(unrecorded.new_ids)} ESPN id(s) new to the registry: "
                f"{', '.join(unrecorded.new_ids[:10])}"
            )
        if check:
            print("  --check: not recording")
        else:
            _record_all(unrecorded.observations, namespace)

    namer = TeamNamer(default_teams(namespace), league)
    decisions = Decisions(_decisions_file(namespace), interactive)
    activity = asyncio.run(_read_activity(league))

    if activity is not None:
        unresolved = unresolved_names(activity, namer)
        placed = {
            espn_id for name in activity if (espn_id := namer.espn_id(name)) is not None
        }
        print(f"{len(activity)} names in the seasons -> {len(placed)} teams")

        missing = [n for n in unresolved if n not in namer]
        both = [n for n in unresolved if n in namer]
        if missing:
            print(f"{len(missing)} name(s) the registry has never seen")
        if both:
            print(f"{len(both)} name(s) that two teams both answer to: {both}")

        problems = check_registry(namer, activity)
        if problems:
            print(f"\n{len(problems)} problem(s) -- two programs sharing an id:")
            print("\n".join(f"  {p}" for p in problems))

        if not check:
            accepted = _review_names(
                propose_names(
                    activity,
                    namer=namer,
                    min_similarity=min_similarity,
                    max_gap_years=max_gap_years,
                ),
                activity,
                namer,
                decisions,
                reg_league,
            )
            if accepted:
                _record_all(accepted, namespace)

    if check:
        print("\nNothing recorded; drop --check to record.")
        return

    merged = _review_duplicate_ids(
        default_teams(namespace), reg_league, namespace, decisions
    )
    if merged:
        print(f"{merged} ESPN record(s) marked as duplicates")
    decisions.save()

    print(f"\n{len(default_teams(namespace))} teams in the registry")
    print(f"Staged in {local_path(namespace)}. To get them into the package:")
    print(
        f"  ciwyw show --namespace {namespace} "
        f"--output call_it_what_you_want/data/{namespace}.csv"
    )


def main(
    league: str,
    first_year: int | None = None,
    last_year: int = 2025,
    pull: bool = True,
    check: bool = False,
    yes: bool = False,
    min_similarity: float = 0.75,
    max_gap_years: int = 2,
) -> None:
    # `--yes` is a misnomer kept for the shape people expect: there is no
    # answer that accepts every guess unseen, because accepting a wrong one
    # merges two schools. It means "don't stop to ask", and unanswered
    # guesses are left for a run with someone watching.
    interactive = not yes and sys.stdin.isatty()
    if not interactive and not check:
        print("Not interactive; debatable entries will be left undecided.")
    _sync(
        league,
        first_year,
        last_year,
        pull=pull,
        check=check,
        interactive=interactive,
        min_similarity=min_similarity,
        max_gap_years=max_gap_years,
    )


if __name__ == "__main__":
    fire.Fire(main)
