"""Find the names a team registry doesn't cover yet, and where they belong.

`call_it_what_you_want` answers *who a team is*; this answers *which names it
is still missing*, by looking at the seasons cassandra actually replays. The
output is rows in that package's format -- `espn_id,name,year,source` -- so a
finding here becomes either a local correction under `cassandra/data/` or a
pull request against the registry, with no translation in between.

Three signals, none of which needs anything the seasons don't already have:

*Temporal disjointness* is the strong one. A school cannot play itself, so two
names for the same school never appear in the same season -- the old one stops
the year the new one starts. This is what tells a rename apart from a
coincidence: `St. Thomas (MN) Tommies` and `St. Thomas University (Fl)
Bobcats` are two different schools and their seasons overlap, so the pair is
rejected without anyone having to know that one is in Florida.

*Name similarity* is the obvious one, and on its own it is not enough: it
can't distinguish `Miami Hurricanes` from `Miami RedHawks`, which are
different schools whose names differ by less than some real renames do.

*Opponent continuity* is the tiebreaker. A renamed school keeps its
conference, so the opponents in the old name's last season and the new name's
first season are largely the same teams. A coincidence shares nobody.

Every proposal carries all three numbers, for a human to accept or reject.
Nothing here writes to the registry.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from call_it_what_you_want import normalize
from endgame.types import Season, iter_weeks

from .aliases import TeamNamer

# Expanded before comparing, so `UW-Oshkosh` and `Wisconsin-Oshkosh` are the
# same string by the time similarity is measured. Deliberately short: every
# entry is a chance to collapse two schools that really are different, and the
# other two signals are what carry the decision.
_ABBREVIATIONS = {
    "uw": "wisconsin",
    "st": "saint",
    "univ": "university",
    "u": "university",
}

# Dropped as noise. Note what is *not* here: "state" stays, because `Michigan`
# and `Michigan State` are different schools and dropping it merges them.
_STOPWORDS = frozenset({"university", "college", "the", "of", "at"})

_NON_ALPHA = re.compile(r"[^a-z0-9]+")

# What a row proposed here is a sighting *by*. These names come out of the
# season data, which endgame built from ESPN's `displayName`, so they are
# ESPN's names even when ESPN's own team endpoint no longer serves them.
_SOURCE = "espn"

# One row of the registry's CSV: espn_id, name, year, source.
Row = tuple[str, str, int, str]


def normalize_name(
    name: str,
    abbreviations: Mapping[str, str] = _ABBREVIATIONS,
    stopwords: frozenset[str] = _STOPWORDS,
) -> str:
    """A team name reduced to what two spellings of it have in common.

    Built on the registry's own `normalize` rather than beside it, so case,
    whitespace and periods fold exactly the way a lookup folds them. The
    rest -- abbreviations, stopwords -- is fuzzy matching, which is this
    module's job and deliberately not the registry's: `normalize` is what
    decides whether two strings are the *same* name, and guessing there
    would resolve a name onto a team nobody claimed it was.
    """
    tokens = _NON_ALPHA.sub(" ", normalize(name)).split()
    expanded = [abbreviations.get(token, token) for token in tokens]
    return " ".join(token for token in expanded if token not in stopwords)


def name_similarity(left: str, right: str) -> float:
    """0-1 similarity of two team names, after normalization."""
    return SequenceMatcher(None, normalize_name(left), normalize_name(right)).ratio()


@dataclass
class TeamActivity:
    """When a name was in use, and who it played, per season."""

    name: str
    opponents_by_year: dict[int, set[str]] = field(default_factory=dict)

    @property
    def years(self) -> set[int]:
        return set(self.opponents_by_year)

    @property
    def first_year(self) -> int:
        return min(self.opponents_by_year)

    @property
    def last_year(self) -> int:
        return max(self.opponents_by_year)

    @property
    def n_games(self) -> int:
        return sum(len(o) for o in self.opponents_by_year.values())


def team_activity(seasons: Iterable[Season]) -> dict[str, TeamActivity]:
    """Index every name in the seasons by the years it played and against whom."""
    activity: dict[str, TeamActivity] = {}
    for season in seasons:
        for week in iter_weeks(season, validate=False):
            for game in week.games:
                for team, opponent in ((game.home, game.away), (game.away, game.home)):
                    entry = activity.setdefault(team, TeamActivity(team))
                    entry.opponents_by_year.setdefault(season.year, set()).add(opponent)
    return activity


def unresolved_names(
    activity: Mapping[str, TeamActivity], namer: TeamNamer
) -> dict[str, TeamActivity]:
    """The names in the seasons the registry can't place, worst first.

    This is the coverage number for a league: an empty result means every
    team in every game cassandra replays has an identity.
    """
    unresolved = {
        name: entry for name, entry in activity.items() if not namer.resolves(name)
    }
    return dict(sorted(unresolved.items(), key=lambda kv: kv[1].n_games, reverse=True))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass
class NameCandidate:
    """One unplaced name and the team it probably belongs to.

    `known` is whichever name was used more recently, since that's the one
    the registry is most likely to already have an id for.
    """

    name: str
    known: str
    similarity: float
    gap_years: int
    opponent_jaccard: float
    name_years: tuple[int, int]
    known_years: tuple[int, int]
    name_games: int
    known_games: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "known": self.known,
            "accept": True,
            "evidence": {
                "similarity": round(self.similarity, 3),
                "gap_years": self.gap_years,
                "opponent_jaccard": round(self.opponent_jaccard, 3),
                "name_years": list(self.name_years),
                "known_years": list(self.known_years),
                "name_games": self.name_games,
                "known_games": self.known_games,
            },
        }


def rows_for(
    candidate: NameCandidate,
    activity: Mapping[str, TeamActivity],
    namer: TeamNamer,
) -> list[Row]:
    """Registry rows for an accepted candidate: one per season it was seen.

    A row per year rather than one row for the name, because the registry
    stores observations -- "this source called this team this in this season"
    -- and `name_in` uses the years to answer what a team was called back
    then. Squashing them to a single year would answer that wrong.

    Returns nothing when the `known` side has no id, which means the whole
    group is new to the registry and needs a team created by hand rather
    than a name attached to one.
    """
    espn_id = namer.espn_id(candidate.known)
    if espn_id is None:
        return []
    return [
        (espn_id, candidate.name, year, _SOURCE)
        for year in sorted(activity[candidate.name].years)
    ]


def _candidate(
    older: TeamActivity, newer: TeamActivity, similarity: float
) -> NameCandidate:
    # The opponents either side of the handover: a rename keeps its
    # conference, so these two sets are mostly the same teams.
    return NameCandidate(
        name=older.name,
        known=newer.name,
        similarity=similarity,
        gap_years=newer.first_year - older.last_year - 1,
        opponent_jaccard=_jaccard(
            older.opponents_by_year[older.last_year],
            newer.opponents_by_year[newer.first_year],
        ),
        name_years=(older.first_year, older.last_year),
        known_years=(newer.first_year, newer.last_year),
        name_games=older.n_games,
        known_games=newer.n_games,
    )


def propose_names(
    activity: Mapping[str, TeamActivity],
    namer: TeamNamer | None = None,
    min_similarity: float = 0.75,
    max_gap_years: int = 2,
    min_opponent_jaccard: float = 0.0,
) -> list[NameCandidate]:
    """Every pair worth a human's attention, best evidence first.

    Pairs the registry already agrees about are skipped, so a run against a
    populated registry reports only what it is still missing.

    `min_opponent_jaccard` defaults to 0 so a school that changed conference
    in the same year it was renamed still gets proposed -- with a jaccard of
    0 attached, which is exactly the case a reviewer should look at rather
    than one the tool should silently drop.
    """
    namer = namer or TeamNamer.empty()
    candidates = []
    for left, right in combinations(sorted(activity), 2):
        if namer.same_team(left, right):
            continue
        first, second = activity[left], activity[right]
        # Two names in the same season are two teams, whatever they're
        # called. This is the check that does the most work.
        if first.years & second.years:
            continue
        similarity = name_similarity(left, right)
        if similarity < min_similarity:
            continue
        older, newer = (
            (first, second) if first.last_year < second.first_year else (second, first)
        )
        candidate = _candidate(older, newer, similarity)
        if candidate.gap_years > max_gap_years:
            continue
        if candidate.opponent_jaccard < min_opponent_jaccard:
            continue
        candidates.append(candidate)
    # Opponent overlap first: it's the signal least likely to be a
    # coincidence, so the top of the list is where review time pays off.
    return sorted(
        candidates,
        key=lambda c: (c.opponent_jaccard, c.similarity),
        reverse=True,
    )


def check_registry(namer: TeamNamer, activity: Mapping[str, TeamActivity]) -> list[str]:
    """Problems with what the registry says, checked against the seasons.

    The one that matters: two names the registry calls the same team, that
    played each other. That's two real programs sharing an id, and it
    silently destroys both teams' ratings while the replay runs happily.
    """
    problems = []
    by_team: dict[str, list[str]] = {}
    for name in activity:
        espn_id = namer.espn_id(name)
        if espn_id is not None:
            by_team.setdefault(espn_id, []).append(name)

    for espn_id, names in sorted(by_team.items()):
        for left, right in combinations(sorted(names), 2):
            overlap = activity[left].years & activity[right].years
            if overlap:
                problems.append(
                    f"{left!r} and {right!r} are both ESPN id {espn_id} but played "
                    f"in the same season(s): {sorted(overlap)}"
                )
    return problems


def next_renames(
    activity: Mapping[str, TeamActivity],
    namer: TeamNamer,
    year: int,
    min_similarity: float = 0.85,
) -> list[NameCandidate]:
    """Names new in `year` that look like a rename the registry hasn't got.

    A registry goes stale the season after it's pulled unless something
    looks for the next rename. Run this after a data refresh: an empty list
    means no team quietly reset to 1500.
    """
    debuted = {
        name: entry
        for name, entry in activity.items()
        if entry.first_year == year and not namer.resolves(name)
    }
    return propose_names(
        {**debuted, **{n: e for n, e in activity.items() if e.last_year < year}},
        namer=namer,
        min_similarity=min_similarity,
    )


def summarize(candidates: Sequence[NameCandidate]) -> str:
    """A table for reading in a terminal before opening the review file."""
    if not candidates:
        return "no candidates"
    width = max(len(c.name) for c in candidates)
    lines = [f"{'name':{width}}  ->  known as  (sim / jaccard / gap)"]
    lines += [
        f"{c.name:{width}}  ->  {c.known}  "
        f"({c.similarity:.2f} / {c.opponent_jaccard:.2f} / {c.gap_years}y)"
        for c in candidates
    ]
    return "\n".join(lines)
