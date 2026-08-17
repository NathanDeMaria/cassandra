from datetime import datetime
from pathlib import Path

import pytest
from endgame.types import Game

from .elo import EloPredictor


def _game(home: str, away: str, home_score: int, away_score: int) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2023, 1, 1),
        game_id="1",
    )


def test_elo():
    predictor = EloPredictor("placeholder")
    team_a = "Team A"
    team_b = "Team B"
    original_rating_a = predictor.get_rating(team_a)
    original_rating_b = predictor.get_rating(team_b)
    predictor.update_game(_game(team_a, team_b, 1, 0))
    assert (
        predictor.get_rating(team_a)
        > original_rating_a
        == original_rating_b
        > predictor.get_rating(team_b)
    )


def test_no_regression_by_default() -> None:
    """The default has to be a no-op: every release published so far omits it."""
    predictor = EloPredictor("test_league")
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    before = predictor.get_rating("Team A")
    predictor.pass_season()
    assert predictor.get_rating("Team A") == before


def test_season_regression_pulls_both_directions_toward_the_mean() -> None:
    predictor = EloPredictor("test_league", season_regression=0.5)
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    winner = predictor.get_rating("Team A")
    loser = predictor.get_rating("Team B")

    predictor.pass_season()

    assert predictor.get_rating("Team A") == pytest.approx(1500 + (winner - 1500) / 2)
    assert predictor.get_rating("Team B") == pytest.approx(1500 + (loser - 1500) / 2)


def test_full_regression_forgets_the_season() -> None:
    predictor = EloPredictor("test_league", season_regression=1.0)
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    predictor.pass_season()
    assert predictor.get_rating("Team A") == pytest.approx(1500)


def test_regression_targets_the_anchor_when_there_is_one() -> None:
    """The seam per-division priors come in through.

    A team whose anchor is 1200 regresses toward 1200, not toward the 1500
    that only makes sense for a league whose teams all play each other.
    """
    predictor = EloPredictor(
        "test_league", season_regression=1.0, anchors={"Team A": 1200}
    )
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    predictor.pass_season()
    assert predictor.get_rating("Team A") == pytest.approx(1200)
    assert predictor.get_rating("Team B") == pytest.approx(1500)


def test_anchors_round_trip_through_the_state_dict() -> None:
    """A release replays against the anchors it was fit with.

    Recomputing them on load instead would silently re-rate a published
    model whenever the anchor file changed underneath it.
    """
    predictor = EloPredictor("test_league", anchors={"Team A": 1200})

    restored = EloPredictor.from_state_dict(predictor.state_dict())

    assert restored._anchors == {"Team A": 1200}


def test_explicit_empty_anchors_are_not_replaced_by_the_saved_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`{}` means no anchors, not "go find some"."""
    monkeypatch.setattr(
        "cassandra.predictor.base_predictor.load_anchors",
        lambda league: {"Team A": 1200},
    )

    assert EloPredictor("test_league", anchors={})._anchors == {}
    assert EloPredictor("test_league")._anchors == {"Team A": 1200}


def test_out_of_range_regression_is_rejected() -> None:
    with pytest.raises(ValueError):
        EloPredictor("test_league", season_regression=1.5)
    with pytest.raises(ValueError):
        EloPredictor("test_league", season_regression=-0.1)


def test_season_regression_round_trips(tmp_path: Path) -> None:
    predictor = EloPredictor("test_league", season_regression=0.25)
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    save_path = tmp_path / "elo.json"
    predictor.save_state(save_path)

    loaded = EloPredictor.load_state(save_path)
    loaded.pass_season()
    predictor.pass_season()
    assert loaded.get_rating("Team A") == pytest.approx(predictor.get_rating("Team A"))


def test_old_state_without_regression_still_loads() -> None:
    """A state file written before the parameter existed."""
    loaded = EloPredictor.from_state_dict(
        {
            "league": "test_league",
            "home_advantage": 80,
            "k": 15,
            "ratings": {"Team A": 1600},
        }
    )
    loaded.pass_season()
    assert loaded.get_rating("Team A") == 1600


def test_elo_save_load(tmp_path: Path) -> None:
    predictor = EloPredictor("test_league", home_advantage=80, k=15)
    predictor.update_game(_game("Team A", "Team B", 2, 1))
    predictor.update_game(_game("Team C", "Team A", 0, 3))

    save_path = tmp_path / "elo.json"
    predictor.save_state(save_path)

    loaded = EloPredictor.load_state(save_path)
    assert loaded.get_rating("Team A") == pytest.approx(predictor.get_rating("Team A"))
    assert loaded.get_rating("Team B") == pytest.approx(predictor.get_rating("Team B"))
    assert loaded.get_rating("Team C") == pytest.approx(predictor.get_rating("Team C"))
    assert loaded.get_rating("Unknown") == 1500


def test_save_load_predictions_match(tmp_path: Path) -> None:
    """After loading, predictions for new games match the original predictor."""
    predictor = EloPredictor("test_league")
    predictor.update_game(_game("Team A", "Team B", 1, 0))

    save_path = tmp_path / "elo.json"
    predictor.save_state(save_path)
    loaded = EloPredictor.load_state(save_path)

    new_game = _game("Team A", "Team B", 2, 1)
    assert loaded.predict_game(new_game).team1_win_prob == pytest.approx(
        predictor.predict_game(new_game).team1_win_prob
    )
