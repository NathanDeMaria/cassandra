from endgame.types import Game

from .base_predictor import Predictor
from .types import Prediction


class FlatPredictor(Predictor):
    def predict_game(self, game: Game) -> Prediction:
        return Prediction(team1_win_prob=0.5)
