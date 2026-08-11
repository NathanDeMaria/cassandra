import argparse
import asyncio
import json
from collections.abc import Collection
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


def _models(leagues: Collection[str] | None = None) -> Iterator[tuple[str, str, Path]]:
    """Every scoreable model, as (league, model name, config path).

    Restricted to `leagues` when it's non-empty; otherwise everything.
    """
    found: dict[tuple[str, str], Path] = {}
    for models_dir in (_AUTHORED_DIR, _GENERATED_DIR):
        if not models_dir.is_dir():
            continue
        for league_path in sorted(models_dir.iterdir()):
            if not league_path.is_dir():
                continue
            if leagues and league_path.name not in leagues:
                continue
            for path in sorted(league_path.glob("*_result.json")):
                # Generated results are read second on purpose: a freshly
                # optimized model wins over a baseline of the same name.
                found[(league_path.name, path.stem.removesuffix("_result"))] = path
    yield from (
        (league, model, path) for (league, model), path in sorted(found.items())
    )


async def _main(leagues: Collection[str] | None = None):
    all_evaluations = []
    for league, model_name, model_path in _models(leagues):
        print(f"Evaluating {league}: {model_name}")
        try:
            predictions_df = await get_predictions(
                model_path,
                league,
                _GENERATED_DIR / league / f"{model_name}_state.json",
            )
        except Exception as e:
            # One unscoreable model shouldn't cost the whole run its table,
            # since the csv is only written once everything has finished.
            print(f"  FAILED {league}/{model_name}: {e!r}")
            continue
        for fitter_name, fitter in DEFAULT_FITTERS.items():
            scored = score_predictions(predictions_df, fitter)
            all_evaluations.append(
                {
                    "league": league,
                    "model": model_name,
                    "fitter": fitter_name,
                    **scored.metrics,
                }
            )
            # Saved next to the model's ratings, and in the same run that
            # scored it, so the mapping a consumer reads is the one the
            # numbers in the csv were computed from.
            calibration_path = (
                _GENERATED_DIR / league / f"{model_name}_{fitter_name}_calibration.json"
            )
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(scored.margin_predictor.to_dict()))

    df = pd.DataFrame(all_evaluations)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _METRICS_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(_METRICS_DIR / f"{timestamp}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score every model in models/.")
    parser.add_argument(
        "--league",
        action="append",
        dest="leagues",
        metavar="LEAGUE",
        help="only evaluate this league; repeatable (default: all leagues)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.leagues))
