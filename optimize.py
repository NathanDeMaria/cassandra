import asyncio
from dataclasses import asdict
from functools import partial

import pandas as pd
from bayes_opt import BayesianOptimization

from prediction_evaluation.predictor import Elo538Predictor, Predictor
from prediction_evaluation.save_predictions import (
    Config,
    NcaabbGender,
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
    prediction_results = join_with_odds(predictor, seasons, odds_db, optimization=True)
    df = pd.DataFrame([asdict(result) for result in prediction_results])
    return -((df["team1_win_prob"] - df["team1_win"]) ** 2).mean()


async def _main() -> None:
    # TODO: make this configurable
    pbounds = {"home_advantage": (0, 300), "k": (1, 40)}
    predictor_class = Elo538Predictor
    gender = NcaabbGender.mens

    config = Config.init_from_file()
    seasons = [s async for s in read_all_seasons(gender, config.bucket)]
    odds_db = await OddsDatabase.from_s3(config.bucket)

    optimizer = BayesianOptimization(
        f=partial(
            _negative_brier_score,
            league=gender.name,
            seasons=seasons,
            odds_db=odds_db,
            predictor_class=predictor_class,
        ),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(n_iter=100)

    print(optimizer.max)


if __name__ == "__main__":
    asyncio.run(_main())
