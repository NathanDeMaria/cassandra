import asyncio
from datetime import datetime

import pandas as pd

from cassandra.constants import CASSANDRA_HOME
from cassandra.model_eval import DEFAULT_FITTERS, get_predictions, score_predictions

_MODELS_DIR = CASSANDRA_HOME / "models"
_METRICS_DIR = CASSANDRA_HOME / "evaluations"


async def _main():
    all_evaluations = []
    for league_path in _MODELS_DIR.iterdir():
        league = league_path.name
        for model_path in league_path.glob("*_result.json"):
            model_name = model_path.stem.replace("_result", "")
            print(f"Evaluating {league}: {model_name}")
            predictions_df = await get_predictions(model_path, league)
            for fitter_name, fitter in DEFAULT_FITTERS.items():
                evaluation_metrics = score_predictions(predictions_df, fitter)
                all_evaluations.append(
                    {
                        "league": league,
                        "model": model_name,
                        "fitter": fitter_name,
                        **evaluation_metrics,
                    }
                )

    df = pd.DataFrame(all_evaluations)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _METRICS_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(_METRICS_DIR / f"{timestamp}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(_main())
