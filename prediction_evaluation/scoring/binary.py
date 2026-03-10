from endgame.types import Game


def binary_score(game: Game) -> float:
    """Binary scoring: 1 for win, 0.5 for tie, 0 for loss (from home perspective)."""
    if game.home_score > game.away_score:
        return 1.0
    if game.home_score == game.away_score:
        return 0.5
    return 0.0
