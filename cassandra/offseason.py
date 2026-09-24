"""What changed about a team between seasons, as a rating model reads it.

The facts are say-youll-remember-me's: why the head coach who opened last
season is gone, who started this season's first game (with play-by-play) at
quarterback, and every passer's attempts and EPA. What those facts are worth
is a modeling question, and this is where it's answered.

**A quarterback's quality** is his EPA per pass attempt against the
season's FBS average, shrunk toward a prior by `PRIOR_ATTEMPTS` attempts, so
a backup's 40 good throws count for about a quarter of a starter's season.
The prior for the quarterback who left is average -- he was the starter, so
the team was rated with him. The prior for the one who arrives is what
first-time starters have turned out to be, measured off the data rather
than assumed (about 0.03 EPA per attempt below average): a starter nobody
has seen throw is not an average starter.

**The change at quarterback** is the week-one starter's quality last season,
at whatever school he threw for -- which is how a transfer's record follows
him -- minus the departing main starter's there. Zero when the starter
returns.

Measured against `glicko_margin_units`' residuals, FBS 2015-2025, with
`say-youll-remember-me` at 30465e7: in a team's first four games, a new
starter is -0.76 points (t -1.8) and each 0.1 EPA per attempt of quality
change +1.14 (t 4.3); over the season, -0.43 and +0.63 (t 3.9). A coach who
left for a better job is -2.5 early and -2.3 over the season (t -3.7); a
fired one is nothing (+0.1), and neither is one who retired (+0.3, 18 of
them), even from a strong team. Coordinator changes, measured the same way,
showed nothing on either side of the ball and aren't read.

**The NFL** reads the same facts from say-youll-remember-me's NFL files,
with every team setting a season's passing average rather than FBS. Its
residuals are too few to say much on their own -- a new starter -0.4 points
early (t -0.7), quality +0.7 per 0.1 (t 0.8) -- but replaying
`nfl/glicko_margin_units` with the college-sized shifts gains 0.0003 brier
on 2015-2025 and 0.0008 in teams' first four games. Only three NFL coaches
have left for another job, the last in 2007, so its coach shift has
nothing to read. Its eight retirements run -2.9 points over the next
season but +1.3 in the first four games: an old roster coming apart as the
season goes, not something the offseason knew.

Keyed by the team name a `Matchup` carries: call-it-what-you-want's current
name in college, which is also what the data files record; in the NFL,
endgame's nickname for the franchise ("raiders"), made from the files'
names by the rule endgame's stored seasons were named by.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from typing import NamedTuple, Self

import say_youll_remember_me as syrm
from call_it_what_you_want import TeamNamer
from endgame.nfl.games import _get_team as nfl_nickname

#: Attempts of the prior a quarterback's EPA per attempt is shrunk toward.
#: Measured flat between 75 and 300: the effect's size and t moved in the
#: second decimal across that range.
PRIOR_ATTEMPTS = 150.0

#: The fewest attempts in a first season as a starter for it to count
#: toward what first-time starters turn out to be.
NEWCOMER_MIN_ATTEMPTS = 100

#: The leagues the data files cover. Any other league has no facts, and a
#: model reading them sees every team as unchanged.
FACTS_LEAGUES = ("ncaafb", "nfl")


class OffseasonFact(NamedTuple):
    """One team-season: why last season's opening coach is gone, and the quarterback change.

    `coach_departure` is say-youll-remember-me's reason category (`fired`,
    `left_for_job`, ...), None when the coach who opened last season opens
    this one. `new_quarterback` is None when the plays don't say -- no
    week-one starter, or no main starter last season to compare with -- and
    `quality_change` is in EPA per attempt, 0 unless the starter is new.
    """

    coach_departure: str | None
    new_quarterback: bool | None
    quality_change: float


class QuarterbackRow(NamedTuple):
    """The fields of say-youll-remember-me's `QuarterbackSeason` this reads."""

    team: str
    season: int
    player_key: str
    starts: int
    started_week_one: bool
    attempts: int | None
    epa: float | None


def quarterback_facts(
    rows: Iterable[QuarterbackRow], peers: set[tuple[str, int]]
) -> dict[tuple[str, int], tuple[bool, float]]:
    """(new starter, quality change) per (team, season), from quarterback seasons.

    `peers` is the team-seasons whose passers set a season's average -- the
    level a quality is measured from: FBS in college, the whole NFL. Pure,
    so it's testable on rows built by hand.
    """
    rows = list(rows)
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    anywhere: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        if not row.attempts:
            continue
        anywhere[(row.player_key, row.season)][0] += row.epa or 0.0
        anywhere[(row.player_key, row.season)][1] += row.attempts
        if (row.team, row.season) in peers:
            totals[row.season][0] += row.epa or 0.0
            totals[row.season][1] += row.attempts
    average = {season: epa / att for season, (epa, att) in totals.items() if att}

    main: dict[tuple[str, int], QuarterbackRow] = {}
    for row in sorted(rows, key=lambda r: (r.starts, r.attempts or 0, r.player_key)):
        main[(row.team, row.season)] = row  # the most starts, then attempts, wins
    week_one = {(r.team, r.season): r for r in rows if r.started_week_one}

    first_seasons = [
        (starter.epa / starter.attempts - average[starter.season], starter.attempts)
        for starter in main.values()
        if starter.attempts
        and starter.epa is not None
        and starter.attempts >= NEWCOMER_MIN_ATTEMPTS
        and starter.season in average
        and starter.season - 1 in average
        and anywhere.get((starter.player_key, starter.season - 1), [0, 0])[1] == 0
    ]
    newcomer = (
        sum(q * a for q, a in first_seasons) / sum(a for _, a in first_seasons)
        if first_seasons
        else 0.0
    )

    def quality(epa: float, attempts: float, season: int, prior: float) -> float:
        above = epa - attempts * average[season]
        return (above + PRIOR_ATTEMPTS * prior) / (attempts + PRIOR_ATTEMPTS)

    found: dict[tuple[str, int], tuple[bool, float]] = {}
    for (team, season), starter in week_one.items():
        last = main.get((team, season - 1))
        if last is None or season - 1 not in average:
            continue
        if starter.player_key == last.player_key:
            found[(team, season)] = (False, 0.0)
            continue
        outgoing = quality(last.epa or 0.0, last.attempts or 0, season - 1, 0.0)
        epa, attempts = anywhere.get((starter.player_key, season - 1), [0.0, 0.0])
        incoming = quality(epa, attempts, season - 1, newcomer)
        found[(team, season)] = (True, incoming - outgoing)
    return found


class OffseasonFacts:
    """Offseason facts by (team, season), empty for a league the data doesn't cover."""

    def __init__(self, facts: Mapping[tuple[str, int], OffseasonFact] | None = None):
        self._facts = dict(facts or {})

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(_league_facts(league))

    def get(self, team: str, season: int | None) -> OffseasonFact | None:
        if season is None:
            return None
        return self._facts.get((team, season))

    def seasons(self, season: int | None) -> Iterable[tuple[str, OffseasonFact]]:
        """Every team with a fact for `season`; none before a model knows its season."""
        if season is None:
            return ()
        return (
            (team, fact) for (team, year), fact in self._facts.items() if year == season
        )

    def __len__(self) -> int:
        return len(self._facts)


@cache
def _league_facts(league: str) -> Mapping[tuple[str, int], OffseasonFact]:
    """The data files, read into facts once per process.

    Cached because an optimization run builds hundreds of predictors.
    """
    if league not in FACTS_LEAGUES:
        return {}
    name = _namer(league)
    # The staff file covers exactly the peers: every FBS team-season in
    # college, every franchise's in the NFL.
    peers = {(name(r.team), r.season) for r in syrm.coaching_staffs(league)}
    rows = [
        QuarterbackRow(
            team=name(r.team),
            season=r.season,
            player_key=r.player_key,
            starts=r.starts,
            started_week_one=r.started_week_one,
            attempts=r.attempts,
            epa=r.epa,
        )
        for r in syrm.quarterback_seasons(league)
    ]
    quarterbacks = quarterback_facts(rows, peers)
    team_seasons = {(r.team, r.season) for r in rows} | peers
    ids = {name(r.team): r.espn_id for r in syrm.coaching_staffs(league)}
    ids |= {name(r.team): r.espn_id for r in syrm.quarterback_seasons(league)}
    facts: dict[tuple[str, int], OffseasonFact] = {}
    for team, season in team_seasons:
        departed = syrm.departure(ids[team], season, league) if team in ids else None
        new_qb, change = quarterbacks.get((team, season), (None, 0.0))
        if departed is None and new_qb is None:
            continue
        facts[(team, season)] = OffseasonFact(
            coach_departure=departed.reason if departed is not None else None,
            new_quarterback=new_qb,
            quality_change=change,
        )
    return facts


def _namer(league: str) -> Callable[[str], str]:
    """How a data file's team name becomes the one a `Matchup` carries."""
    if league == "nfl":
        return _nfl_name
    return TeamNamer.for_league(league).canonical


def _nfl_name(name: str) -> str:
    """ "Las Vegas Raiders" -> "raiders", as endgame named its stored seasons.

    Its rule, not a copy of it: the files' names and the seasons' have to
    agree for every franchise, and a second rule would drift from the one
    that named the seasons.
    """
    nickname = nfl_nickname(name)
    if nickname is None:
        raise ValueError(f"not an NFL franchise: {name!r}")
    return nickname
