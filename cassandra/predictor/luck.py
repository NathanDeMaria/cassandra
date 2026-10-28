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


def validated_luck_share(share: float) -> float:
    """In [0, 1]: none of a game's luck taken out, or all of it."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"luck_share must be in [0, 1], got {share}")
    return share


class GameLuck(NamedTuple):
    """A game's luck in points, toward `home` (a canonical name)."""

    home: str
    away: str
    luck: float


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
    return {
        row["game_id"]: GameLuck(
            namer.canonical(row["home"]) if row["home"] else "",
            namer.canonical(row["away"]) if row["away"] else "",
            float(row["luck"]),
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

    def home_luck(self, game: Game) -> float | None:
        """`game`'s luck toward its home side, in points; None if the file
        doesn't have it, or files it under teams the game doesn't name.

        The file's home is ten-minute-version's, which can be the other side
        at a neutral site: it's turned around where the names say so. Either
        side's name is enough, since a team outside Division I can be named
        differently by the two.
        """
        found = self._games.get(game.game_id)
        if found is None:
            return None
        if game.home == found.home or game.away == found.away:
            return found.luck
        if game.home == found.away or game.away == found.home:
            return -found.luck
        return None
