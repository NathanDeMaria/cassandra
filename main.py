import asyncio
from datetime import datetime, timezone
from pathlib import Path

import fire

from cassandra.predictor import EloPredictor, FlatPredictor, Predictor
from cassandra.save_predictions import save_predictions

_PREDICTION_LOG_PATH = Path(__file__).parent / "predictions"


async def _main(league: str) -> None:
    prediction_path = _PREDICTION_LOG_PATH / datetime.now(timezone.utc).isoformat()
    prediction_path.mkdir(exist_ok=True, parents=True)

    predictors: list[Predictor] = [
        FlatPredictor(league),
        EloPredictor(league),
    ]
    for predictor in predictors:
        file_path = (
            prediction_path / f"{league}-{predictor.__class__.__name__}-results.csv"
        )
        await save_predictions(predictor, league, file_path, post_callbacks=True)


def main(league: str = "mens") -> None:
    asyncio.run(_main(league))


if __name__ == "__main__":
    fire.Fire(main)
