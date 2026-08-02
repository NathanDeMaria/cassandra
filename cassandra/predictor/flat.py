import json
from pathlib import Path
from typing import Self

from .base_predictor import Predictor
from .types import Matchup, Prediction


class FlatPredictor(Predictor):
    def predict_game(self, matchup: Matchup) -> Prediction:
        return Prediction(team1_win_prob=0.5)

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"league": self._league}))

    @classmethod
    def load_state(cls, path: Path) -> Self:
        data = json.loads(path.read_text())
        return cls(**data)
