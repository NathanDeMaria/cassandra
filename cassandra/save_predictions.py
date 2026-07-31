import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator

import aiofiles
import pandas as pd
from endgame.types import Game, Season
from endgame_aws import Config, read_seasons
from endgame_aws.io import S3NotFoundException

from .odds import Odds, OddsDatabase
from .predictor import GameResult, Predictor

_EARLIEST_SEASON_YEAR = 1999


async def read_all_seasons(league: str, bucket: str) -> AsyncIterator[Season]:
    # Leagues don't all have data going back to the same year (e.g. ncaabb
    # seasons in S3 start around 2010, nfl/ncaafb around 1999), so just skip
    # missing years instead of hard-coding a start/end per league.
    latest_year = datetime.now(timezone.utc).year + 1
    for year in range(_EARLIEST_SEASON_YEAR, latest_year + 1):
        try:
            seasons = await read_seasons(bucket, f"seasons/{year}/{league}.pkl")
        except S3NotFoundException:
            continue
        for season in seasons:
            yield season


def generate_predictions(
    predictor: Predictor, seasons: Iterable[Season], post_callbacks: bool = False
) -> Iterator[GameResult]:
    for season in seasons:
        for week in season.weeks:
            for game in week.games:
                prediction = predictor.predict_game(game)
                yield GameResult(
                    prediction, game, year=season.year, week_number=week.number
                )
            predictor.pass_week()
        predictor.pass_season()
    if post_callbacks:
        predictor.postrun_callback()


@dataclass
class _Prediction:
    year: int
    week_number: int
    home_score: int
    away_score: int
    team1_win: bool
    team1_win_prob: float
    spread: float | None
    home_team: str
    away_team: str


def _build_prediction(result: GameResult, odds: Odds | None) -> _Prediction:
    team1_win = result.game.home_score > result.game.away_score
    spread = odds.spread if odds else None
    return _Prediction(
        year=result.year,
        week_number=result.week_number,
        home_score=result.game.home_score,
        away_score=result.game.away_score,
        team1_win=team1_win,
        team1_win_prob=result.prediction.team1_win_prob,
        spread=spread,
        home_team=result.game.home,
        away_team=result.game.away,
    )


async def _serialize_predictions(
    results: AsyncIterable[_Prediction], file_path: Path
) -> None:
    async with aiofiles.open(file_path, mode="w", encoding="utf-8", newline="") as f:
        await f.write(
            "year,week_number,home_score,away_score,team1_win,team1_win_prob,spread\n"
        )
        async for result in results:
            await f.write(",".join(asdict(result).values()))
            await f.write("\n")


async def save_predictions(
    predictor: Predictor, league: str, file_path: Path, post_callbacks: bool
) -> None:
    prediction_results = _get_results(predictor, league, post_callbacks=post_callbacks)
    await _serialize_predictions(prediction_results, file_path)


async def build_predictions_df(
    predictor: Predictor, league: str, post_callbacks: bool
) -> pd.DataFrame:
    prediction_results = _get_results(predictor, league, post_callbacks=post_callbacks)
    return pd.DataFrame([asdict(result) async for result in prediction_results])


async def _get_results(
    predictor: Predictor, league: str, post_callbacks: bool
) -> AsyncIterator[_Prediction]:
    config = Config.init_from_file()
    seasons = read_all_seasons(league, config.bucket)
    odds_db = await OddsDatabase.from_s3(config.bucket)
    seasons_now = [s async for s in seasons]
    predictions = join_with_odds(
        predictor, seasons_now, odds_db, post_callbacks=post_callbacks
    )
    for prediction in predictions:
        yield prediction


def join_with_odds(
    predictor: Predictor,
    seasons: Iterable[Season],
    odds_db: OddsDatabase,
    post_callbacks: bool,
) -> Iterator[_Prediction]:
    # like build_predictions_df, but meant for optimization that already has read seasons/odds into memory
    for result in generate_predictions(
        predictor, seasons, post_callbacks=post_callbacks
    ):
        odds = odds_db.get_odds(result.game.game_id)
        yield _build_prediction(result, odds)
