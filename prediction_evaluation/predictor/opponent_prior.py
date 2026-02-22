import json
from collections import Counter, defaultdict
from pathlib import Path

from endgame.types import Game

_PREDICTOR_DATA_DIR = Path(__file__).parent / "data"


class OpponentPriorManager:
    def __init__(self, league: str, model: str | None = None) -> None:
        self._prior_path = (
            _PREDICTOR_DATA_DIR / f"{league}_{model + '_' or ''}priors.json"
        )
        self._opponent_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
        if self._prior_path.exists():
            with self._prior_path.open("r") as f:
                self._ratings = json.load(f)
        else:
            self._ratings = {}

    def add_game(self, game: Game) -> None:
        self._opponent_counter[game.home][game.away] += 1
        self._opponent_counter[game.away][game.home] += 1

    def save(self, ratings: dict[str, float]) -> None:
        if self._prior_path.exists():
            raise ValueError(f"Priors already exist at {self._prior_path}")
        # Save off priors for each time, equal to the average rating of their top 10 opponents
        priors = {
            team: self._common_opponent_rating(team, ratings)
            for team in self._opponent_counter.keys()
            # There's some small schools that only play a few games
            # and only against top teams. Don't treat them as equal
            if sum(self._opponent_counter[team].values()) >= 50
        }
        with self._prior_path.open("w") as f:
            json.dump(priors, f)

    def _common_opponent_rating(self, team: str, ratings: dict[str, float]) -> float:
        opponents = self._opponent_counter[team]
        total_games = sum(opponents.values())
        return (
            sum(ratings.get(o, 1500) * count for o, count in opponents.items())
            / total_games
        )

    def get_ratings(self) -> dict[str, float]:
        return self._ratings
