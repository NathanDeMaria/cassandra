import pytest

from .conftest import GameFactory
from .glicko import GlickoPredictor

# The rating half of Glicko's behavior is checked against every model in
# contract_test.py. What's here is the deviation, which only Glicko keeps.


def test_pass_season_widens_the_rating_deviation(game: GameFactory) -> None:
    """An offseason makes us less sure of a rating, whether or not it regresses."""
    predictor = GlickoPredictor(
        "test_league", initial_rd=350, season_rd_increase=100, season_regression=0.5
    )
    predictor.update_game(game("Team A", "Team B", 1, 0))
    before = predictor.get_rating("Team A")

    predictor.pass_season()

    assert predictor.get_rating("Team A").rating_deviation > before.rating_deviation


def test_an_anchored_team_is_no_better_measured_than_any_other(
    game: GameFactory,
) -> None:
    """The anchor sets the rating and leaves the deviation alone.

    Knowing a team's division says where its rating starts, not how sure we
    are of it -- an anchored team that hasn't played is as unmeasured as one
    with no anchor at all.
    """
    predictor = GlickoPredictor("test_league", initial_rd=200, anchors={"Team A": 1200})

    assert predictor.get_rating("Team A").rating_deviation == 200
    assert predictor.get_rating("Team B").rating_deviation == 200


def test_save_load_keeps_the_deviations(tmp_path, game: GameFactory) -> None:
    """The contract's round trip only pins the ratings; the rd has to survive too."""
    predictor = GlickoPredictor(
        "test_league", weekly_rd_increase=2, season_rd_increase=100, initial_rd=200
    )
    predictor.update_game(game("Team A", "Team B", 1, 0))
    predictor.update_game(game("Team C", "Team A", 2, 1))
    predictor.pass_week()

    save_path = tmp_path / "glicko.json"
    predictor.save_state(save_path)
    loaded = GlickoPredictor.load_state(save_path)

    for team in ("Team A", "Team B", "Team C"):
        assert loaded.get_rating(team).rating_deviation == pytest.approx(
            predictor.get_rating(team).rating_deviation
        )
    assert loaded.get_rating("Unknown").rating_deviation == 200
