import asyncio
from datetime import datetime, timezone
from pathlib import Path

from endgame.ncaabb import NcaabbGender

from prediction_evaluation.predictor import EloPredictor, FlatPredictor, Predictor
from prediction_evaluation.save_predictions import save_predictions

_PREDICTION_LOG_PATH = Path(__file__).parent / "predictions"


async def _main() -> None:
    prediction_path = _PREDICTION_LOG_PATH / datetime.now(timezone.utc).isoformat()
    prediction_path.mkdir(exist_ok=True, parents=True)

    gender = NcaabbGender.mens
    predictors: list[Predictor] = [
        FlatPredictor(),
        EloPredictor(),
    ]
    for predictor in predictors:
        file_path = (
            prediction_path
            / f"{gender.name}-{predictor.__class__.__name__}-results.csv"
        )
        await save_predictions(predictor, gender, file_path)


if __name__ == "__main__":
    asyncio.run(_main())
