from endgame.types import Game

from .elo import EloPredictor


def test_elo():
    predictor = EloPredictor("placeholder")
    team_a = "Team A"
    team_b = "Team B"
    original_rating_a = predictor.get_rating(team_a)
    original_rating_b = predictor.get_rating(team_b)
    predictor.predict_game(
        Game(
            home=team_a,
            away=team_b,
            home_score=1,
            away_score=0,
            neutral_site=False,
            completed=True,
            date="2023-01-01",
            game_id="1",
        )
    )
    assert (
        predictor.get_rating(team_a)
        > original_rating_a
        == original_rating_b
        > predictor.get_rating(team_b)
    )
