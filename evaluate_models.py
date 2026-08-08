import asyncio
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from cassandra.constants import CASSANDRA_HOME
from cassandra.model_eval import DEFAULT_FITTERS, get_predictions, score_predictions

# Hand-written baselines are checked in next to the optimization configs they
# sit alongside; everything optimize.py produces lands under CASSANDRA_HOME.
_AUTHORED_DIR = Path(__file__).parent / "models"
_GENERATED_DIR = CASSANDRA_HOME / "models"
_METRICS_DIR = CASSANDRA_HOME / "evaluations"


def _models() -> Iterator[tuple[str, str, Path]]:
    """Every scoreable model, as (league, model name, config path)."""
    found: dict[tuple[str, str], Path] = {}
    for models_dir in (_AUTHORED_DIR, _GENERATED_DIR):
        if not models_dir.is_dir():
            continue
        for league_path in sorted(models_dir.iterdir()):
            if not league_path.is_dir():
                continue
            for path in sorted(league_path.glob("*_result.json")):
                # Generated results are read second on purpose: a freshly
                # optimized model wins over a baseline of the same name.
                found[(league_path.name, path.stem.removesuffix("_result"))] = path
    yield from (
        (league, model, path) for (league, model), path in sorted(found.items())
    )


async def _main():
    all_evaluations = []
    for league, model_name, model_path in _models():
        print(f"Evaluating {league}: {model_name}")
        predictions_df = await get_predictions(
            model_path,
            league,
            _GENERATED_DIR / league / f"{model_name}_state.json",
        )
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
