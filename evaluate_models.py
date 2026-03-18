import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
from endgame.ncaabb import NcaabbGender

from cassandra import evaluate_model

_REPO_ROOT = Path(__file__).parent
_MODELS_DIR = _REPO_ROOT / "models"
_METRICS_DIR = _REPO_ROOT / "evaluations"


async def _main():
    all_evaluations = []
    for league_path in _MODELS_DIR.iterdir():
        league = NcaabbGender[league_path.name]
        for model_path in league_path.glob("*_result.json"):
            model_name = model_path.stem.replace("_result", "")
            print(f"Evaluating {league.name}: {model_name}")
            evaluation_metrics = await evaluate_model(model_path, league)
            all_evaluations.append(
                {
                    "league": league.name,
                    "model": model_name,
                    **evaluation_metrics,
                }
            )

    df = pd.DataFrame(all_evaluations)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(_METRICS_DIR / f"{timestamp}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(_main())
