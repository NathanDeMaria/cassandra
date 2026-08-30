import csv
import io
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator

import aiofiles
import pandas as pd
from call_it_what_you_want import TeamNamer
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
    predictor: Predictor,
    seasons: Iterable[Season],
    post_callbacks: bool = False,
    namer: TeamNamer | None = None,
    roll_over_final_season: bool = True,
) -> Iterator[GameResult]:
    """Replay every season in order, training the predictor as it goes.

    `roll_over_final_season` is about the last `pass_season` and nothing else.
    Between two seasons the rollover -- regression toward each team's anchor,
    and Glicko's season rd bump -- has to happen, or the next season's games
    are predicted by ratings that never cooled off. After the *last* season
    it's a bet about a season that hasn't started, applied to ratings that are
    about to be read as "where the teams stand". Scoring can't tell the
    difference, since no game follows it, so this defaults to True and only
    publish -- which does read the ratings afterward -- turns it off.
    """
    # Team names are canonicalized here, before anything sees a game, so the
    # predictor, the predictions and the release all agree on who a team is.
    # Defaulted rather than required: every caller wants the league's
    # registry, and a test that wants no renaming passes TeamNamer.empty().
    if namer is None:
        namer = TeamNamer.for_league(predictor.league)
    # Chronological order matters here: update_game feeds each result back
    # into the predictor, so replaying games out of order trains it on
    # results from the future. iter_weeks raises if a season's weeks overlap
    # in time, which means its games are grouped into the wrong weeks and
    # sorting can't save us.
    ordered = sorted(seasons, key=lambda s: s.year)
    for index, season in enumerate(ordered):
        for week in iter_weeks(season):
            played = [g for g in week.games_in_order if g.completed]
            if not played and week.games:
                # Every game in the week is a fixture, so the week hasn't
                # happened. Skipping it -- rather than falling through to an
                # empty `pass_week` -- is what keeps the clock from running
                # into the future: Glicko inflates every team's rd once per
                # week passed, and a season pickle now carries weeks of
                # games nobody has played.
                #
                # `week.games` is what tells the two empties apart. A week
                # with no games at all is one the source itself had nothing
                # for, and it passed before this existed; only a week
                # emptied by the filter is the future.
                continue
            for game in played:
                game = namer.apply(game)
                prediction = predictor.update_game(game)
                yield GameResult(
                    prediction, game, year=season.year, week_number=week.number
                )
            predictor.pass_week()
        if roll_over_final_season or index < len(ordered) - 1:
            # The season being entered, so a predictor with per-season
            # anchors regresses each team toward the division it is about to
            # play in. Past the last stored season that's a year nobody has
            # played, and the anchors -- fit from played seasons -- have
            # nothing for it; `anchor_in` holds the last one it does have.
            following = (
                ordered[index + 1].year
                if index + 1 < len(ordered)
                else season.year + 1
            )
            predictor.pass_season(following)
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
    # Which game this was. Carried so a consumer of the predictions can say
    # *which* games it has already seen: publish.py writes these into a
    # release's `trained_through`, and the incremental refresh uses them as
    # its idempotency watermark, since games get re-fetched and scores
    # corrected and ids are the only thing that makes a re-run exact.
    # Appended rather than placed up front so the csv's existing columns keep
    # their positions.
    game_id: str
    date: datetime


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
        game_id=result.game.game_id,
        date=result.game.date,
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
    roll_over_final_season: bool = True,
) -> Iterator[_Prediction]:
    # like build_predictions_df, but meant for optimization that already has read seasons/odds into memory
    for result in generate_predictions(
        predictor,
        seasons,
        post_callbacks=post_callbacks,
        roll_over_final_season=roll_over_final_season,
    ):
        odds = odds_db.get_odds(result.game.game_id)
        yield _build_prediction(result, odds)
