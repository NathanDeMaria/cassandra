import csv
import io
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator

import aiofiles
import pandas as pd
from endgame.types import Season, iter_weeks
from endgame_aws import Config, list_all_keys, read_seasons

from .odds import Odds, OddsDatabase
from .predictor import GameResult, Predictor

_SEASON_KEY_RE = re.compile(r"^seasons/(\d+)/([^/]+)\.pkl$")


async def read_all_seasons(league: str, bucket: str) -> AsyncIterator[Season]:
    async for key in list_all_keys(bucket, "seasons/"):
        match = _SEASON_KEY_RE.match(key)
        if match is None or match.group(2) != league:
            continue
        seasons = await read_seasons(bucket, key)
        for season in seasons:
            yield season


def generate_predictions(
    predictor: Predictor, seasons: Iterable[Season], post_callbacks: bool = False
) -> Iterator[GameResult]:
    # Chronological order matters here: update_game feeds each result back
    # into the predictor, so replaying games out of order trains it on
    # results from the future. iter_weeks raises if a season's weeks overlap
    # in time, which means its games are grouped into the wrong weeks and
    # sorting can't save us.
    for season in sorted(seasons, key=lambda s: s.year):
        for week in iter_weeks(season):
            for game in week.games_in_order:
                prediction = predictor.update_game(game)
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


def _to_csv_row(values: Iterable[Any]) -> str:
    # team names can contain commas, so let the csv module handle quoting
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue()


async def _serialize_predictions(
    results: AsyncIterable[_Prediction], file_path: Path
) -> None:
    async with aiofiles.open(file_path, mode="w", encoding="utf-8", newline="") as f:
        await f.write(_to_csv_row(field.name for field in fields(_Prediction)))
        async for result in results:
            await f.write(_to_csv_row(asdict(result).values()))


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
