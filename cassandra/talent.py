"""How talented each roster is, as a rating model reads it.

The numbers are 247Sports' Team Talent Composite: the top 85 players on a
team's active roster, each at their composite rating as a high school recruit,
weighted so the best 40% of the roster make up about 80% of the total.
Transfers count, at their high school rating, and leave the number when they
leave the roster -- in the change from one season to the next, star-weighted
transfers in and out move a team about as much as its freshman class and
its departing fifth-years do. Nothing is weighted by class year, and nobody's
rating moves after high school. so-high-school -- a private repo, since the
numbers are 247's -- files them by ESPN id and season and copies them to
`predictor/data/<league>_talent.csv`, which every Batch job downloads with
the rest of that prefix. Nothing here imports it.

**What a model reads** is a team's talent in standard deviations above that
season's FBS average. Standardized because the scale drifts: the spread of
FBS talent was 180-200 points through 2023 and 149, 145 and 121 in
2024-2026, as backups who transfer down carry their high school ratings with
them, so a hundred points is a bigger gap now than it was in 2016. Only FBS
teams: 247 rated about a hundred FCS teams a year through 2023 and none
since, so an average over everyone it rated would drop and recover between
eras for no football reason. Not the academies either, which 247 lists at or
near zero -- Air Force's 71 in 2015 against an FBS average near 540 says
nothing about the team that took the field.

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


def standardized(
    rows: Iterable[TalentRow], in_division: Callable[[str, int], bool]
) -> dict[tuple[str, int], float]:
    """Each division team-season's talent, in standard deviations from its season's mean.

    Keyed by (ESPN id, season). Pure, so it's testable on rows built by hand.
    """
    by_season: dict[int, list[TalentRow]] = {}
    for row in rows:
        if row.espn_id not in ACADEMIES and in_division(row.espn_id, row.season):
            by_season.setdefault(row.season, []).append(row)
    out: dict[tuple[str, int], float] = {}
    for season, read in by_season.items():
        values = [row.talent for row in read]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        if not sd:
            continue
        out |= {(row.espn_id, season): (row.talent - mean) / sd for row in read}
    return out


@cache
def load_talent(league: str) -> Mapping[tuple[str, int], float]:
    """A league's standardized talent by (team, season), empty if it has no file.

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
        (names[espn_id], season): z
        for (espn_id, season), z in standardized(rows, in_division).items()
    }


class TalentIndex:
    """Talent by (team, season), in standard deviations from the season's FBS mean.

    Constructible from a mapping for tests, and loadable per league for a
    replay, as `QbOutIndex` is.
    """

    def __init__(self, talent: Mapping[tuple[str, int], float] | None = None):
        self._talent = dict(talent or {})
        self._seasons: dict[str, list[int]] = {}
        for team, season in sorted(self._talent, key=lambda key: key[1]):
            self._seasons.setdefault(team, []).append(season)

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

    def previous(self, team: str, season: int) -> float:
        """The team's talent in the latest season before `season` that has one; 0 if none.

        What a shift applied each summer has already added, when every
        summer's shift is the change since the one before: those changes sum
        to the latest level, gaps and all.
        """
        earlier = [y for y in self._seasons.get(team, ()) if y < season]
        return self._talent[(team, earlier[-1])] if earlier else 0.0

    def __len__(self) -> int:
        return len(self._talent)
