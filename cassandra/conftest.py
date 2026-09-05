"""Play-by-play fixtures, shared by the two sweeps built on `cassandra.pbp_sweep`.

Here rather than in one of the test files because both of them need the same
thing and neither owns it: `game_control_build_test` and `epa_build_test`
differ in which number they check, exactly as the modules they test do.

The fixtures are deliberately crude -- a snap every thirty seconds at
midfield, one team ahead the whole way -- because what the build tests check
is the bookkeeping around the models, not the models. Whether a win
probability curve or an expected points fit is any good is `lucky_ones`' own
test suite's problem; whether a week's worth of games ends up in the right
file with the right header is theirs.
"""

from datetime import datetime, timezone
from typing import Iterable, NamedTuple, Sequence

from endgame.types import Game, Season, Week

LEAGUE = "nfl"
HOME_ID, AWAY_ID = "100", "200"


class Play(NamedTuple):
    """A row of endgame's play schema, in the columns `Play` reads."""

    league: str
    season: int
    week: int
    game_id: str
    play_id: str
    play_number: int
    period: int | None
    clock_seconds: int | None
    wallclock: datetime | None
    home_score: int | None
    away_score: int | None
    offense_team_id: str | None
    defense_team_id: str | None
    down: int | None
    distance: int | None
    yardline: int | None
    play_type: str | None
    text: str | None
    scoring_play: bool | None
    is_penalty: bool | None
    is_turnover: bool | None
    drive_id: str | None
    drive_number: int
    drive_team_id: str | None
    drive_result: str | None
    drive_is_score: bool | None


def plays_for(
    game_id: str,
    season: int = 2025,
    week: int = 1,
    periods: Iterable[int] = (1, 2, 3, 4),
    home_leads: bool = True,
    fumble_on: int | None = None,
) -> list[Play]:
    """A game as a snap every thirty seconds, with one side scoring first.

    The scoring plays are what `infer_home_team_id` votes on, so the game has
    to have some for `group_by_game` to keep it at all.
    """
    plays: list[Play] = []
    home_score = away_score = 0
    number = 0
    leader, trailer = (
        (HOME_ID, AWAY_ID) if home_leads else (AWAY_ID, HOME_ID)
    )
    for period in periods:
        for clock in range(900, 0, -30):
            number += 1
            # One touchdown for the leader early in each period, so the
            # scoring-drive vote has something to count and the leader is
            # ahead for essentially the whole game.
            scored = clock == 870
            if scored:
                if home_leads:
                    home_score += 7
                else:
                    away_score += 7
            plays.append(
                Play(
                    league=LEAGUE,
                    season=season,
                    week=week,
                    game_id=game_id,
                    play_id=f"{game_id}-{number}",
                    play_number=number,
                    period=period,
                    clock_seconds=clock,
                    wallclock=None,
                    home_score=home_score,
                    away_score=away_score,
                    offense_team_id=leader if period % 2 else trailer,
                    defense_team_id=trailer if period % 2 else leader,
                    down=1,
                    distance=10,
                    yardline=50,
                    play_type="Rush",
                    # `lucky_ones.luck` classifies off this sentence and
                    # nothing else, so a game with no text has no bounces in
                    # it and its adjusted control is its realized control.
                    text=(
                        "Smith rushes for 3 yards, FUMBLES, RECOVERED by Jones"
                        if number == fumble_on
                        else None
                    ),
                    scoring_play=scored,
                    is_penalty=False,
                    is_turnover=number == fumble_on,
                    drive_id=f"{game_id}-d{period}",
                    drive_number=period,
                    drive_team_id=leader,
                    drive_result="TD" if scored else None,
                    drive_is_score=scored,
                )
            )
    return plays


class FakeSource:
    """A `PlaySource` over plays held in a dict, keyed by (year, week)."""

    def __init__(self, weeks: dict[tuple[int, int], Sequence[Play]]) -> None:
        self._weeks = weeks
        self.loaded: list[tuple[int, int]] = []

    async def load_game(self, league, season, week, game_id):
        return [p for p in await self.load_week(league, season, week) if p.game_id == game_id]

    async def load_week(self, league: str, season: int, week: int):
        self.loaded.append((season, week))
        return list(self._weeks.get((season, week), []))

    async def load_weeks(self, league, season, weeks):
        out = []
        for week in weeks:
            out.extend(await self.load_week(league, season, week))
        return out


def game_for(game_id: str, home_score: int = 28, away_score: int = 0) -> Game:
    return Game(
        home="Home",
        away="Away",
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2025, 9, 7, tzinfo=timezone.utc),
        game_id=game_id,
        status="STATUS_FINAL",
    )


def season_for(year: int, *game_ids: str, week: int = 1) -> Season:
    return Season(
        year=year, weeks=[Week(games=[game_for(g) for g in game_ids], number=week)]
    )
