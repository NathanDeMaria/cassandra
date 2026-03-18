from endgame.types import Game


def pythagorean_score(game: Game) -> float:
    """Pythagorean win percentage: home**2 / (home**2 + away**2)."""
    home = game.home_score
    away = game.away_score
    if home == 0 and away == 0:
        return 0.5
    return home**2 / (home**2 + away**2)
