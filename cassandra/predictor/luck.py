"""How lucky each basketball game's shooting was, for a rating to discount.

A final margin carries a lot of luck: whether the threes and free throws
went in, beyond what was expected of who took them and from where. Glicko
takes a margin as all signal, so a team that's been lucky comes out
overrated. ten-minute-version files each game's luck (`ten_minute_version.
luck`): each side's points on threes and free throws less what its shot
model expected of the shooters, the home side's less the away's, as known
the morning after -- `{league}_luck.csv` under the predictor data prefix,
mens and womens:

    league, season, game_id, date, home_team_id, home, away_team_id, away,
    luck, as_of

`GlickoPredictor` takes `luck_share` of a game's luck off its home margin
before scoring it (`_actual`), in the update alone: the prediction is of
the game as it'll be played, luck and all.

A game at a home site is filed less the home sides' average luck over the
same fit's games: the shot model knows nothing of venue, and home sides
shooting better than it expects is the home edge, which Glicko has its own
of.

Measured before this existed, each team's luck so far this season predicts
the replay's next residual against it (-0.0011 logit per point mens,
-0.0016 womens, t -6 each). `glicko_players` at its 2026-10-28 fit,
learning from margins with a share of the luck out, Brier on 2014+ games:

    share       0.3        0.4        0.5        0.6        1 (uncentered)
    mens     -0.00035   -0.00038   -0.00037   -0.00031   +0.0008
    womens   -0.00038   -0.00044   -0.00045   -0.00041   +0.0004

All of it is too much: the shot model's expectation misses some of what's
real -- the same reason ten-minute-version's ratings take out half.

The file also carries what the game's plays say of how it got to its margin
(`ten_minute_version.flow`), each in points toward the home side and empty
where the plays don't give it, and where its luck came from:

    jumpers, over_lead, closing, source

`jumpers` is two-point jump shots that fell beyond the league's rate,
`over_lead` the final margin less twice the game's average lead, `closing`
what the last five minutes added. Each has its own share in the update
(`jumper_share`, `lead_share`, `closing_share`), taken off the margin with
the luck's. A `source` of "box" is a game the shot model never saw -- every
one before 2013-14 -- whose luck is its box score's threes and free throws
against the season's rates: a best estimate, read like any other.

`glicko_players` at its 2026-11-08 fits, Brier over every game 2010-2026
(and in how many of 2014-2025's twelve seasons it's better), against the
file as it was before these:

                                               mens             womens
    box-score luck for the games without     -0.00010  (6)    -0.00008  (6)
    and jumper_share 0.4                     -0.00024 (11)    -0.00021 (12)
    and lead_share 0.25 (mens),
        closing_share 0.3 (womens)           -0.00039 (12)    -0.00026 (12)

The box-score luck is all in the seasons it fills. `closing_share` adds
nothing to the men's pair and `lead_share` 0.00001 to the women's.
"""

import csv
from functools import cache
from pathlib import Path
from typing import NamedTuple, Self

from call_it_what_you_want import TeamNamer
from endgame.types import Game

from cassandra.constants import CASSANDRA_HOME

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"


def luck_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_luck.csv"


def validated_share(name: str, share: float) -> float:
    """In [0, 1]: none of what a game's reading says taken out, or all of it."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {share}")
    return share


def validated_luck_share(share: float) -> float:
    """In [0, 1]: none of a game's luck taken out, or all of it."""
    return validated_share("luck_share", share)


class GameLuck(NamedTuple):
    """A game's luck in points, toward `home` (a canonical name), and the
    flow readings beside it, each None where the file has none: a game
    whose plays don't give them, or a file from before it carried them."""

    home: str
    away: str
    luck: float
    jumpers: float | None = None
    over_lead: float | None = None
    closing: float | None = None

    def turned(self) -> GameLuck:
        """The same game with the other side called home."""

        def other(value: float | None) -> float | None:
            return None if value is None else -value

        return GameLuck(
            self.away,
            self.home,
            -self.luck,
            other(self.jumpers),
            other(self.over_lead),
            other(self.closing),
        )


@cache
def load_luck(league: str) -> dict[str, GameLuck]:
    """A league's games' luck by game id; empty if it has no file.

    Names go through the league's `TeamNamer`, as the games' do. Cached for
    the reason `load_qb_out` is.
    """
    path = luck_path(league)
    if not path.exists():
        return {}
    namer = TeamNamer.for_league(league)

    def reading(row: dict[str, str], column: str) -> float | None:
        return float(row[column]) if row.get(column) else None

    return {
        row["game_id"]: GameLuck(
            namer.canonical(row["home"]) if row["home"] else "",
            namer.canonical(row["away"]) if row["away"] else "",
            float(row["luck"]),
            reading(row, "jumpers"),
            reading(row, "over_lead"),
            reading(row, "closing"),
        )
        for row in csv.DictReader(path.read_text().splitlines())
    }


class LuckIndex:
    """Each game's shooting luck, toward the side the game calls home."""

    def __init__(self, games: dict[str, GameLuck] | None = None) -> None:
        self._games = games or {}

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_luck(league))

    def toward_home(self, game: Game) -> GameLuck | None:
        """`game`'s luck and flow readings toward its home side, in points;
        None if the file doesn't have the game, or files it under teams the
        game doesn't name.

        The file's home is ten-minute-version's, which can be the other side
        at a neutral site: it's turned around where the names say so. Either
        side's name is enough, since a team outside Division I can be named
        differently by the two.
        """
        found = self._games.get(game.game_id)
        if found is None:
            return None
        if game.home == found.home or game.away == found.away:
            return found
        if game.home == found.away or game.away == found.home:
            return found.turned()
        return None

    def home_luck(self, game: Game) -> float | None:
        """`game`'s shooting luck alone, toward its home side."""
        found = self.toward_home(game)
        return None if found is None else found.luck
