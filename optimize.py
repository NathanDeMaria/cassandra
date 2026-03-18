import asyncio
from dataclasses import asdict
from functools import partial
from pathlib import Path

import fire
import pandas as pd
from pydantic import BaseModel

from cassandra.brier import brier_score_df
from cassandra.optimize import optimize
from cassandra.predictor import (
    Predictor,
    PredictorConfig,
    load_predictor_class,
)
from cassandra.save_predictions import (
    Config,
    NcaabbGender,
    OddsDatabase,
    Season,
    join_with_odds,
    read_all_seasons,
)


class _OptimizationConfig(BaseModel):
    predictor_class: str
    league: str
    parameters: dict[str, tuple[float, float] | list[str]]
    n_iter: int = 100


def _negative_brier_score(
    league: str,
    seasons: list[Season],
    odds_db: OddsDatabase,
    predictor_class: type[Predictor],
    **kwargs,
) -> float:
    predictor = predictor_class(league, **kwargs)  # type: ignore[call-arg]
    prediction_results = join_with_odds(
        predictor, seasons, odds_db, post_callbacks=False
    )
    df = pd.DataFrame([asdict(result) for result in prediction_results])
    return -brier_score_df(df)


async def _run_optimization(config_file: str) -> None:
    config_path = Path(config_file)
    with open(config_path, "r") as f:
        config_model = _OptimizationConfig.model_validate_json(f.read())

    predictor_class = load_predictor_class(config_model.predictor_class)

    gender = NcaabbGender[config_model.league]

    aws_config = Config.init_from_file()
    seasons = [s async for s in read_all_seasons(gender, aws_config.bucket)]
    odds_db = await OddsDatabase.from_s3(aws_config.bucket)

    # Run once to make things like team priors
    predictor = predictor_class(gender.name)
    for _ in join_with_odds(predictor, seasons, odds_db, post_callbacks=True):
        pass

    target_function = partial(
        _negative_brier_score,
        league=gender.name,
        seasons=seasons,
        odds_db=odds_db,
        predictor_class=predictor_class,
    )
    target, params = optimize(
        target_function, config_model.parameters, config_model.n_iter
    )

    result_model = PredictorConfig(
        predictor_class=config_model.predictor_class,
        league=config_model.league,
        target=target,
        params=params,
    )

    output_path = config_path.parent / f"{config_path.stem}_result.json"
    output_path.write_text(result_model.model_dump_json(indent=4, by_alias=True))


def _main(config_file: str) -> None:
    asyncio.run(_run_optimization(config_file))


if __name__ == "__main__":
    fire.Fire(_main)
