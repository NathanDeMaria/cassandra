"""What a rating model knows about a roster before the season.

Three measures, all filed by ESPN id and season by so-high-school -- a
private repo, since the talent numbers are 247's -- and copied to
`predictor/data/<league>_talent.csv`, `<league>_returning_offense.csv` and
`<league>_returning_defense.csv`, which every Batch job downloads with the
rest of that prefix. Nothing here imports it.

**What a model reads** is each FBS team's value in standard deviations from
that season's FBS mean. Only FBS teams: 247 rated about a hundred FCS teams
a year through 2023 and none since, so an average over everyone rated would
drop and recover between eras for no football reason, and the model's
lower-division teams have no counterpart to measure against anyway.

Talent
------

The numbers are 247Sports' Team Talent Composite: the top 85 players on a
team's active roster, each at their composite rating as a high school recruit,
weighted so the best 40% of the roster make up about 80% of the total.
Transfers count, at their high school rating, and leave the number when they
leave the roster -- in the change from one season to the next, star-weighted
transfers in and out move a team about as much as its freshman class and
its departing fifth-years do. Nothing is weighted by class year, and nobody's
rating moves after high school. The academies are left out: 247 lists them
at or near zero, and Air Force's 71 in 2015 against an FBS average near 540
says nothing about the team that took the field. Standardizing matters most
here: the spread of FBS talent was 180-200 points through 2023 and 149, 145
and 121 in 2024-2026, as backups who transfer down carry their high school
ratings with them, so a hundred points is a bigger gap now than it was in
2016.

Measured against `glicko_margin_units_offseason`'s residuals, FBS v FBS
2015-2025: at the same predicted margin, the side with 100 more talent
points beat the model by 0.84 points (t 6), positive in each of the eleven
seasons -- 0.91 in weeks 1-2 and about 0.4 after. The 2026 closing line sat
1.2 points per 100 above the model and the results agreed with the line, so
it's information the market already prices.

**Which players it's about.** Split into the five recruiting classes on a
roster, the newest class is the one whose strength the model misses in the
first four games (+3.0 points per sd of class strength, the four older
classes near 0): the model has watched the older players, through last
season's results, and hasn't seen the freshmen or the transfers. Whether
that is freshmen playing or a class signed on the program's momentum, the
composite carries it as well as the five classes do separately.

**Only a season's own number, or an earlier one.** The talent filed for the
season *after* a game explains that game's miss better than its own season's
does -- the class signed in December and the winter's transfers follow the
results -- so a later season's number is the results leaking back in.

Returning production
--------------------

How much of last season's production is on this season's roster, offense
and defense, built by so-high-school from CFBD's preseason rosters and last
season's player stats. Offense is the share of last season's plays (CFBD's
usage); defense the mean share of tackles, tackles for loss and passes
defended. Incoming transfers count at the new team on both sides of the
share, Connelly's rule -- CFBD's own measure counts only players who
stayed, which in 2026 leaves out over a quarter of what came back. The roster is
the preseason one: a starter hurt for the year is on it, and so are players
who left mid-season, so it is known before the season. Offense starts in
2014 and defense in 2017; a team with nothing last season has neither.

At the same predicted margin, 2017-2025, the side with a standard deviation
more of each beat `glicko_margin_units_offseason` by about a point -- 0.9
for offense and 1.0 for defense with talent and each other in, more in a
team's first four games. They are nearly independent of talent (gap
correlations under 0.1) and of each other (0.17). The academies are read
like anyone: their production is measured the same way.

Keyed by the team name a `Matchup` carries, as `cassandra.offseason` is.
"""

import csv
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import NamedTuple, Self

from call_it_what_you_want import TeamNamer, default_classifications, registry_league

from cassandra.constants import CASSANDRA_HOME

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

#: Air Force, Army and Navy, by ESPN id.
ACADEMIES = frozenset({"2005", "349", "2426"})

#: The division read, and averaged over.
DIVISION = "FBS"


class Measure(NamedTuple):
    """One of the roster files: its name, the column read, and who is left out."""

    name: str
    column: str
    excluded: frozenset[str] = frozenset()


TALENT = Measure("talent", "talent", ACADEMIES)
OFFENSE = Measure("returning_offense", "usage")
DEFENSE = Measure("returning_defense", "defense")


class SeasonValue(NamedTuple):
    """One team-season's number from a roster file."""

    season: int
    espn_id: str
    team: str
    value: float


def roster_path(league: str, measure: Measure) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_{measure.name}.csv"


def standardized(
    rows: Iterable[SeasonValue],
    in_division: Callable[[str, int], bool],
    excluded: frozenset[str] = frozenset(),
) -> dict[tuple[str, int], float]:
    """Each division team-season's value, in standard deviations from its season's mean.

    Keyed by (ESPN id, season). Pure, so it's testable on rows built by hand.
    """
    by_season: dict[int, list[SeasonValue]] = {}
    for row in rows:
        if row.espn_id not in excluded and in_division(row.espn_id, row.season):
            by_season.setdefault(row.season, []).append(row)
    out: dict[tuple[str, int], float] = {}
    for season, read in by_season.items():
        values = [row.value for row in read]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        if not sd:
            continue
        out |= {(row.espn_id, season): (row.value - mean) / sd for row in read}
    return out


@cache
def load_roster(league: str, measure: Measure) -> Mapping[tuple[str, int], float]:
    """A league's standardized measure by (team, season), empty if it has no file.

    Empty is an ordinary answer: only ncaafb has the files, and a model
    reading an empty one is the model without that measure.

    Cached for the reason `load_qb_out` is: an optimization run builds a
    predictor per probe and they would all read the same file.
    """
    path = roster_path(league, measure)
    registry = registry_league(league)
    if registry is None or not path.exists():
        return {}
    rows = [
        SeasonValue(
            season=int(r["season"]),
            espn_id=r["espn_id"],
            team=r["team"],
            value=float(r[measure.column]),
        )
        for r in csv.DictReader(path.read_text().splitlines())
    ]
    classifications = default_classifications()

    def in_division(espn_id: str, season: int) -> bool:
        # The last classification at or before the season, so a season the
        # pinned registry hasn't filed yet reads as the one before it.
        found = classifications.classification_in(espn_id, season, registry)
        return found is not None and found.division == DIVISION

    namer = TeamNamer.for_league(league)
    names = {row.espn_id: namer.canonical(row.team) for row in rows}
    return {
        (names[espn_id], season): z
        for (espn_id, season), z in standardized(
            rows, in_division, measure.excluded
        ).items()
    }


class RosterIndex:
    """One measure by (team, season), in standard deviations from the season's FBS mean.

    Constructible from a mapping for tests, and loadable per league for a
    replay, as `QbOutIndex` is.
    """

    def __init__(self, values: Mapping[tuple[str, int], float] | None = None):
        self._values = dict(values or {})
        self._seasons: dict[str, list[int]] = {}
        for team, season in sorted(self._values, key=lambda key: key[1]):
            self._seasons.setdefault(team, []).append(season)

    @classmethod
    def for_league(cls, league: str, measure: Measure) -> Self:
        return cls(load_roster(league, measure))

    def get(self, team: str, season: int | None) -> float | None:
        if season is None:
            return None
        return self._values.get((team, season))

    def season(self, season: int | None) -> Iterable[tuple[str, float]]:
        """Every team with a value in `season`; none before a model knows its season."""
        if season is None:
            return ()
        return (
            (team, value)
            for (team, year), value in self._values.items()
            if year == season
        )

    def previous(self, team: str, season: int) -> float:
        """The team's value in the latest season before `season` that has one; 0 if none.

        What a shift applied each summer has already added, when every
        summer's shift is the change since the one before: those changes sum
        to the latest level, gaps and all.
        """
        earlier = [y for y in self._seasons.get(team, ()) if y < season]
        return self._values[(team, earlier[-1])] if earlier else 0.0

    def __len__(self) -> int:
        return len(self._values)
