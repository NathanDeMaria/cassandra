"""What a game's players say its margin should be, from a model of players.

Every rating model here rates teams from their results. ten-minute-version
rates *players*, from who was on the floor for every possession (lineup
RAPM, chained season to season and seeded by recruiting ranks), and files
each team as of each week of a season: its players at their expected shares
of its floor time, its possessions per game. `{league}_player_ratings.csv`
under the predictor data prefix, mens and womens:

    league, season, as_of, team_id, team, offense, defense, net, pace,
    games, home_edge

`net` is offense plus defense per 100 possessions and `pace` possessions per
game; a row is fitted on the games before `as_of` and stands for games on
or after it, until the team's next row.

**What a model reads** is the neutral-site margin those say: the two teams'
paces averaged, times the gap in their nets. The home edge is left to the
model, which has its own. `GlickoPredictor` blends it into the prediction
alone -- its own rating gap at `glicko_share`, plus this margin at
`player_rating_weight` rating points per point -- and its ratings still
learn from the results alone.

Why the prediction alone: priced in the update as well, as the matchup
terms are, the ratings learn what the players' margin misses, but the
prediction still reads them at full weight on top of it -- and at no
weight did that come near the blend. Against `glicko_full`'s 2026-10-19
fits on every game the file covers (2014-2025, the whole season):

    Brier            glicko_full   in the update too   prediction only
    mens                 0.17313    0.17225 (w 0.3)     0.17102 (t -14)
    womens               0.15489    0.15463 (w 0.2)     0.15285 (t -12)

the blend a logistic fit with each season held out, at about 0.8 of
glicko's logit and 0.57 per 10 points of the players' margin, both
leagues. It helps in every stretch of the season but mens' first games.
Replayed here at that fit (`glicko_share` 0.82 and `player_rating_weight`
10 mens, 0.76 and 9.7 womens), every game the league has: mens 0.16987 ->
0.16836, womens 0.15454 -> 0.15310 -- the games the file doesn't cover
unchanged.

Measured before this existed (each season's first ten weeks, a normal on
the margin), the players' margin alone beat an older `glicko_full` on the
same games -- 0.1573 -> 0.1526 mens, 0.1395 -> 0.1349 womens.

**Who is likely to play** is in the margin: a row's `available` is the
team's net averaged over who might miss its next game -- each player
available at his chance of playing, from his miss streak and his last
game's minutes, each lineup its available players at their recent shares
rescaled (ten-minute-version's `availability`) -- and the margin reads it
in place of `net` (`TeamAsOf.strength`). A row without it is its net.

An absence is worth more than the margin's weight gives it, which is set by
the healthy margin -- mostly what the results already know, so discounted.
Fitting the residual of the replays on 2014+ games, who actually played,
an absence point is worth about twice a full point of margin on wins
(0.27 logit per point mens, 0.25 womens, beyond the margin's 0.06; a point
is ~0.15): none of it is the team column, and a looser player prior takes
a quarter of it at a cost in margin error. So the absences' shift
(`absences`) has its own searched weight, `absence_weight`, on top: in
the replays, -0.00027 mens and -0.00037 womens on 2014+ Brier against the
weekly rows, where the margin reading `available` alone is -0.00017 and
-0.00021.

**How sure each side is** decides the blend, where a model asks for it
(`player_variance_scale`, `player_floor`): Glicko's share of the teams' gap
is the players' margin's variance over both's -- Glicko's its two rating
deviations squared, the players' `net_sd` (ten-minute-version's posterior
of each net, at the row's shares) scaled, plus a floor for what the
posterior doesn't know: minutes, rotations, the model itself. The best
fixed mix moves through a season the way that says -- the players' margin
strongest in the first weeks, Glicko taking over late -- and fitted held
out by season it beats the fixed blend by 0.00017 mens, 0.00052 womens; a
floor of about 130 rating points, both leagues. Replayed at those fits,
2014+ Brier: 0.16732 -> 0.16717 mens, 0.14998 -> 0.14951 womens.

**A ranking reads the blend too.** A team's published rating is the
blend's view of it at a neutral site (`GlickoPredictor._with_players`), the
rating it learned from results alone beside it as `results`; a team the
file doesn't reach ranks on that alone.

**A row is stale after `MAX_AGE`.** The file is weekly; a game more than
two weeks after a team's last row -- the offseason, or a season the file
doesn't reach -- has no players' margin, and the model predicts it
unblended rather than from last March's team.
"""

import csv
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import NamedTuple, Self

from call_it_what_you_want import TeamNamer
from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

from .types import Matchup

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

#: The leagues ten-minute-version rates.
PLAYER_RATING_LEAGUES = ("mens", "womens")

#: Rating points per point of the players' margin. 0 is off.
DEFAULT_PLAYER_RATING_WEIGHT = 0.0

#: How long a team's row stands for its games.
MAX_AGE = timedelta(days=14)


def player_ratings_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_player_ratings.csv"


def validated_player_rating_weight(weight: float) -> float:
    """Non-negative: a players' margin that says the home side is better is
    not evidence it's worse. 0 is already "no effect"."""
    if weight < 0:
        raise ValueError(f"player_rating_weight must be non-negative, got {weight}")
    return weight


def season_for(day: date) -> int:
    """The basketball season `day` falls in, named for the year it starts,
    as ten-minute-version names them."""
    return day.year if day.month >= 8 else day.year - 1


class TeamAsOf(NamedTuple):
    """A team as of a date: net per 100 possessions, possessions per game."""

    as_of: date
    net: float
    pace: float
    # ten-minute-version's, the year the season starts; None where a caller
    # building rows by hand has no use for it
    season: int | None = None
    # the net averaged over who might miss the team's next game, each player
    # at his chance of playing (ten-minute-version's `availability`); None
    # for a row from before the file carried it
    available: float | None = None
    # per 100, how sure ten-minute-version's fit is of `net` (its posterior
    # sd at the row's shares); None for a row from before the file carried it
    net_sd: float | None = None

    @property
    def strength(self) -> float:
        """What the team's next game reads: its expected net over who might
        be out, or its net where the row doesn't say."""
        return self.net if self.available is None else self.available

    @property
    def shift(self) -> float:
        """How far who might be out moves the net: 0 where the row doesn't
        say."""
        return self.strength - self.net


@cache
def load_player_ratings(league: str) -> Mapping[str, Sequence[TeamAsOf]]:
    """A league's teams by name, each's rows in date order; empty if it has
    no file.

    Names go through the league's `TeamNamer`, as the games' do, so a team
    ESPN renamed is still one team. Cached for the reason `load_qb_out` is.
    """
    path = player_ratings_path(league)
    if not path.exists():
        return {}
    namer = TeamNamer.for_league(league)
    found: dict[str, list[TeamAsOf]] = {}
    for row in csv.DictReader(path.read_text().splitlines()):
        if not row["team"]:
            continue
        found.setdefault(namer.canonical(row["team"]), []).append(
            TeamAsOf(
                date.fromisoformat(row["as_of"]),
                float(row["net"]),
                float(row["pace"]),
                int(row["season"]),
                float(row["available"]) if row.get("available") else None,
                float(row["net_sd"]) if row.get("net_sd") else None,
            )
        )
    return {team: sorted(rows) for team, rows in found.items()}


class TeamRow(BaseModel):
    """`TeamAsOf`, as a published file carries it."""

    as_of: date
    season: int
    net: float
    pace: float
    available: float | None = None
    net_sd: float | None = None


class PlayerRatingsFile(BaseModel):
    """What a consumer predicting the games ahead needs of the players' file:
    each team's newest row, by canonical name.

    Newest only, because a game reads the latest row from before its day and
    the file is rewritten every morning: the games ahead all read today's.
    Published beside the releases (`publish.player_ratings_artifact`), since
    the file itself only exists where the sweep ran -- a model rebuilt from
    its release elsewhere reads an empty index, and predicts unblended.
    """

    league: str
    teams: dict[str, TeamRow]


def latest_player_ratings(league: str) -> PlayerRatingsFile | None:
    """The league's newest rows, or None if it has no file.

    Only the teams with a row as recent as `MAX_AGE` before the file's
    newest: one the file stopped rating -- a team gone from Division I, a
    name ESPN retired -- has nothing to say about a game ahead.
    """
    teams = load_player_ratings(league)
    if not teams:
        return None
    newest = max(rows[-1].as_of for rows in teams.values())
    teams = {
        team: rows for team, rows in teams.items() if newest - rows[-1].as_of <= MAX_AGE
    }
    return PlayerRatingsFile(
        league=league,
        teams={
            team: TeamRow(
                as_of=row.as_of,
                season=row.season if row.season is not None else season_for(row.as_of),
                net=row.net,
                pace=row.pace,
                available=row.available,
                net_sd=row.net_sd,
            )
            for team, rows in teams.items()
            for row in rows[-1:]
        },
    )


class PlayerRatingIndex:
    """Teams as their players rate them, by date.

    Constructible from a mapping for tests and for a caller who knows
    better, and loadable per league for a replay, as `QbOutIndex` is.
    """

    def __init__(self, teams: Mapping[str, Sequence[TeamAsOf]] | None = None):
        self._teams = {team: list(rows) for team, rows in (teams or {}).items()}
        self._dates = {
            team: [row.as_of for row in rows] for team, rows in self._teams.items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_player_ratings(league))

    @classmethod
    def from_file(cls, stored: PlayerRatingsFile) -> Self:
        """The index a published file says: a row per team."""
        return cls(
            {
                team: [
                    TeamAsOf(
                        row.as_of,
                        row.net,
                        row.pace,
                        row.season,
                        row.available,
                        row.net_sd,
                    )
                ]
                for team, row in stored.teams.items()
            }
        )

    def __len__(self) -> int:
        return len(self._teams)

    def team(self, team: str, day: date) -> TeamAsOf | None:
        """The team's latest row from before `day`, if one is recent enough.

        Strictly before: a row's `as_of` is the first day its fit left out,
        and a game's date here is UTC, a day after the local date of a late
        tip -- so a row dated the game's own day might have seen it.
        """
        dates = self._dates.get(team)
        if not dates:
            return None
        at = bisect_right(dates, day - timedelta(days=1))
        if not at:
            return None
        row = self._teams[team][at - 1]
        return row if day - row.as_of <= MAX_AGE else None

    def newest(self, team: str, season: int) -> TeamAsOf | None:
        """The team's latest row of `season`, however old: what a season
        nobody has played yet says of it, read before its first game."""
        for row in reversed(self._teams.get(team, ())):
            if row.season == season:
                return row
        return None

    def margin(self, matchup: Matchup) -> float | None:
        """The players' neutral-site margin for the home side; None if either
        team has no recent row."""
        day = matchup.date.date()
        home, away = self.team(matchup.home, day), self.team(matchup.away, day)
        if home is None or away is None:
            return None
        return (home.pace + away.pace) / 2 * (home.strength - away.strength) / 100

    def margin_variance(self, matchup: Matchup) -> float | None:
        """How unsure the fit says `margin` is, in points squared: both
        sides' `net_sd` at the pair's average pace. None if either team has
        no recent row; a row without `net_sd` counts as sure."""
        day = matchup.date.date()
        home, away = self.team(matchup.home, day), self.team(matchup.away, day)
        if home is None or away is None:
            return None
        pace = (home.pace + away.pace) / 2
        return (pace / 100) ** 2 * (
            (home.net_sd or 0.0) ** 2 + (away.net_sd or 0.0) ** 2
        )

    def absences(self, matchup: Matchup) -> float | None:
        """Points the two sides' likely absences move the margin, toward the
        home side; None if either team has no recent row.

        Each side's `available` less its `net` -- what the margin reads, less
        what it would at the shares so far -- at the pair's average pace. A
        row without `available` moves nothing.
        """
        day = matchup.date.date()
        home, away = self.team(matchup.home, day), self.team(matchup.away, day)
        if home is None or away is None:
            return None
        return (home.pace + away.pace) / 2 * (home.shift - away.shift) / 100
