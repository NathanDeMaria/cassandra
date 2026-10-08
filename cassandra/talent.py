"""How talented each roster is, as a rating model reads it.

The numbers are 247Sports' Team Talent Composite, which rates a roster from
every player's composite rating as a recruit, so it follows transfers and
attrition where one recruiting class's ranking doesn't. so-high-school -- a
private repo, since the numbers are 247's -- files them by ESPN id and season
and copies them to `predictor/data/<league>_talent.csv`, which every Batch
job downloads with the rest of that prefix. Nothing here imports it.

**What a model reads** is a team's talent above that season's FBS average,
in hundreds of points. Only FBS teams: 247 rated about a hundred FCS teams
a year through 2023 and none since, so an average over everyone it rated
would drop and recover between eras for no football reason. Not the
academies either, which 247 lists at or near zero -- Air Force's 71 in 2015
against an FBS average near 540 says nothing about the team that took the
field.

Measured against `glicko_margin_units_offseason`'s residuals, FBS v FBS
2015-2025: at the same predicted margin, the side with 100 more talent
points beat the model by 0.84 points (t 6), positive in each of the eleven
seasons -- 0.91 in weeks 1-2 and about 0.4 after. The 2026 closing line sat
1.2 points per 100 above the model and the results agreed with the line, so
it's information the market already prices.

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


class TalentRow(NamedTuple):
    """One row of the data file."""

    season: int
    espn_id: str
    team: str
    talent: float


def talent_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_talent.csv"


def above_average(
    rows: Iterable[TalentRow], in_division: Callable[[str, int], bool]
) -> dict[tuple[str, int], float]:
    """Each division team-season's talent above its season's average, in hundreds.

    Keyed by (ESPN id, season). Pure, so it's testable on rows built by hand.
    """
    read = [
        row
        for row in rows
        if row.espn_id not in ACADEMIES and in_division(row.espn_id, row.season)
    ]
    totals: dict[int, list[float]] = {}
    for row in read:
        total = totals.setdefault(row.season, [0.0, 0.0])
        total[0] += row.talent
        total[1] += 1
    return {
        (row.espn_id, row.season): (
            row.talent - totals[row.season][0] / totals[row.season][1]
        )
        / 100
        for row in read
    }


@cache
def load_talent(league: str) -> Mapping[tuple[str, int], float]:
    """A league's talent above average by (team, season), empty if it has no file.

    Empty is an ordinary answer: only ncaafb has a file, and a model reading
    an empty one is the model without talent.

    Cached for the reason `load_qb_out` is: an optimization run builds a
    predictor per probe and they would all read the same file.
    """
    path = talent_path(league)
    registry = registry_league(league)
    if registry is None or not path.exists():
        return {}
    rows = [
        TalentRow(
            season=int(r["season"]),
            espn_id=r["espn_id"],
            team=r["team"],
            talent=float(r["talent"]),
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
        (names[espn_id], season): above
        for (espn_id, season), above in above_average(rows, in_division).items()
    }


class TalentIndex:
    """Talent above the season's FBS average by (team, season), in hundreds of points.

    Constructible from a mapping for tests, and loadable per league for a
    replay, as `QbOutIndex` is.
    """

    def __init__(self, talent: Mapping[tuple[str, int], float] | None = None):
        self._talent = dict(talent or {})

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_talent(league))

    def get(self, team: str, season: int | None) -> float | None:
        if season is None:
            return None
        return self._talent.get((team, season))

    def season(self, season: int | None) -> Iterable[tuple[str, float]]:
        """Every rated team in `season`; none before a model knows its season."""
        if season is None:
            return ()
        return (
            (team, above)
            for (team, year), above in self._talent.items()
            if year == season
        )

    def __len__(self) -> int:
        return len(self._talent)
