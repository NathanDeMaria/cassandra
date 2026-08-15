import asyncio
from dataclasses import asdict
from functools import partial
from pathlib import Path

import fire
import pandas as pd

from cassandra.brier import brier_score_df
from cassandra.constants import CASSANDRA_HOME
from cassandra.optimize import optimize
from cassandra.predictor import (
    OptimizationConfig,
    Predictor,
    PredictorConfig,
    load_predictor_class,
)
from cassandra.save_predictions import (
    Config,
    OddsDatabase,
    Season,
    join_with_odds,
    read_all_seasons,
)


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
        config_model = OptimizationConfig.model_validate_json(f.read())

    predictor_class = load_predictor_class(config_model.predictor_class)

    league = config_model.league

    aws_config = Config.init_from_file()
    seasons = [s async for s in read_all_seasons(league, aws_config.bucket)]
    if not seasons:
        # Otherwise every probe scores an empty set of games and the search
        # dies inside brier_score_df, well away from the actual problem.
        raise ValueError(
            f"No seasons for league {league!r} in s3://{aws_config.bucket}/seasons/; "
            "the league's data has to be uploaded before it can be optimized"
        )
    odds_db = await OddsDatabase.from_s3(aws_config.bucket)

    # Run once to make things like team priors
    predictor = predictor_class(league)
    for _ in join_with_odds(predictor, seasons, odds_db, post_callbacks=True):
        pass

    target_function = partial(
        _negative_brier_score,
        league=league,
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

    # The config is a checked-in input; its result is generated, so it lands
    # under CASSANDRA_HOME with the rest of the run's output.
    output_path = CASSANDRA_HOME / "models" / league / f"{config_path.stem}_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_model.model_dump_json(indent=4, by_alias=True))


def _main(config_file: str) -> None:
    asyncio.run(_run_optimization(config_file))


if __name__ == "__main__":
    fire.Fire(_main)
