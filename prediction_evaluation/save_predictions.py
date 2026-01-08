import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, NamedTuple

import aiofiles
from endgame.ncaabb import NcaabbGender
from endgame.types import Game, Season
from endgame_aws import read_seasons, Config

from .odds import OddsDatabase
from .predictor import GameResult, Predictor


async def read_all_seasons(gender: NcaabbGender, bucket: str) -> AsyncIterator[Season]:
    # TODO: check S3 instead of this hard-coded start/end
    for year in range(2010, 2026):
        seasons = await read_seasons(
            bucket, f"seasons/{year}/{gender.name}.pkl"
        )
        for season in seasons:
            yield season


async def generate_predictions(
    predictor: Predictor, seasons: AsyncIterator[Season]
) -> AsyncIterator[GameResult]:
    async for season in seasons:
        for week in season.weeks:
            for game in week.games:
                prediction = predictor.predict_game(game)
                yield GameResult(
                    prediction, game, year=season.year, week_number=week.number
                )
            predictor.pass_week()
        predictor.pass_season()


async def _serialize_predictions(
    results: AsyncIterator[GameResult], odds_db: OddsDatabase, file_path: Path
) -> None:
    async with aiofiles.open(file_path, mode="w", encoding="utf-8", newline="") as f:
        await f.write(
            "year,week_number,home_score,away_score,team1_win,team1_win_prob,spread\n"
        )
        async for result in results:
            team1_win = result.game.home_score > result.game.away_score
            odds = odds_db.get_odds(result.game.game_id)
            spread = odds and odds.spread
            await f.write(
                ",".join(
                    [
                        str(result.year),
                        str(result.week_number),
                        str(result.game.home_score),
                        str(result.game.away_score),
                        str(team1_win),
                        str(result.prediction.team1_win_prob),
                        str(spread),
                    ]
                )
            )
            await f.write("\n")


async def save_predictions(
    predictor: Predictor, gender: NcaabbGender, file_path: Path
) -> None:
    config = Config.init_from_file()
    seasons = read_all_seasons(gender, config.bucket)
    odds = await OddsDatabase.from_s3(config.bucket)
    prediction_results = generate_predictions(predictor, seasons)
    await _serialize_predictions(prediction_results, odds, file_path)
