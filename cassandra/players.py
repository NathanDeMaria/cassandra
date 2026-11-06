"""Every player of a basketball season, as invisible-string's Players page
reads them: ten-minute-version's `{league}_players_{season}.csv`.

For reading, not predicting: no model loads this, so the files sit under
their own prefix rather than `predictor/data/`, which every optimize child
pulls. A morning rewrites the current season (`cassandra.player_ratings_build`);
finished seasons are a one-off upload. `publish` turns each into
`models/{league}/players_{season}.json.gz`.
"""

import csv
import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel, field_validator

from cassandra.constants import CASSANDRA_HOME

PLAYERS_PREFIX = "players/"

_PLAYERS_FILE = re.compile(r"^(?P<league>[a-z]+)_players_(?P<season>\d{4})\.csv$")


class PlayerRow(BaseModel):
    """A player, per 100 possessions; `vs_team` is his rating without his
    team's fifth. `projected`: he hasn't played, so a prior and a projected
    share. No share means the box scores don't have him."""

    athlete_id: str
    player: str | None = None
    team_id: str
    team: str | None = None
    projected: bool
    offense: float
    defense: float
    total: float
    offense_sd: float
    defense_sd: float
    total_sd: float
    vs_team: float
    games: int | None = None
    starts: int | None = None
    minutes: float | None = None
    share: float | None = None

    @field_validator(
        "player", "team", "games", "starts", "minutes", "share", mode="before"
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value == "" else value


class PlayersFile(BaseModel):
    """A season's table. `as_of` is the morning it stands for; a finished
    season has none."""

    league: str
    season: int
    as_of: date | None = None
    players: list[PlayerRow]


def players_dir() -> Path:
    return CASSANDRA_HOME / PLAYERS_PREFIX


def players_csv_path(league: str, season: int) -> Path:
    return players_dir() / f"{league}_players_{season}.csv"


def roster_path(league: str, season: int) -> Path:
    """Each team's last roster before its first game, which a morning can't
    fetch again once it's played: who's on the table without having played."""
    return players_dir() / f"{league}_{season}_roster.parquet"


def read_players_file(path: Path) -> PlayersFile:
    """A `{league}_players_{season}.csv`."""
    match = _PLAYERS_FILE.match(path.name)
    if match is None:
        raise ValueError(f"not a players' file: {path.name}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    days = {r["as_of"] for r in rows if r["as_of"]}
    return PlayersFile(
        league=match["league"],
        season=int(match["season"]),
        as_of=date.fromisoformat(max(days)) if days else None,
        players=[PlayerRow.model_validate(r) for r in rows],
    )
