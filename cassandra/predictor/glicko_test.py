from typing import Any

import pytest

from ..scoring import DEFAULT_SIGMOID_SCALE
from .conftest import GameFactory
from .glicko import DEFAULT_PREDICTION_SCALE, GlickoPredictor, _Rating
from .types import Rating

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


def test_the_sigmoid_scale_round_trips_through_the_state(game: GameFactory) -> None:
    """It changes what a game was worth, so a release has to carry it.

    A model refit at a searched scale and reloaded at the default 10 is a
    different model wearing the same ratings.
    """
    predictor = GlickoPredictor(
        "test_league", scoring_method="sigmoid", sigmoid_scale=1.5
    )

    state = predictor.state_dict()
    restored = GlickoPredictor.from_state_dict(state)

    assert state["sigmoid_scale"] == 1.5
    assert restored.state_dict() == state


def test_a_model_that_never_named_a_scale_still_gets_the_old_one() -> None:
    """Every release published before this existed replayed at 10."""
    predictor = GlickoPredictor("test_league", scoring_method="sigmoid")

    assert predictor.state_dict()["sigmoid_scale"] == DEFAULT_SIGMOID_SCALE


def test_a_smaller_scale_moves_a_rating_further(game: GameFactory) -> None:
    """What the parameter is for, seen through the ratings rather than the
    scorer: at a scale the score line is measured in, a sweep is a rout and
    Glicko moves accordingly."""
    sweep = game("Home", "Away", 3, 0)

    def _after(scale: float) -> float:
        predictor = GlickoPredictor(
            "test_league", scoring_method="sigmoid", sigmoid_scale=scale
        )
        predictor.update_game(sweep)
        return predictor.get_rating("Home").rating

    assert _after(1.0) > _after(DEFAULT_SIGMOID_SCALE)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_scale_of_zero_or_less_is_refused(scale: float) -> None:
    """0 divides by zero and a negative reads the score line backwards --
    the same rule `validated_scale` already holds the blend's scales to."""
    with pytest.raises(ValueError, match="sigmoid_scale must be positive"):
        GlickoPredictor("test_league", sigmoid_scale=scale)


def test_a_model_that_never_named_a_prediction_scale_predicts_at_400() -> None:
    """Every release published before this existed read a gap Elo's way."""
    predictor = GlickoPredictor("test_league", home_advantage=0)

    assert predictor.state_dict()["prediction_scale"] == DEFAULT_PREDICTION_SCALE
    assert predictor.win_prob(1900, 1500) == pytest.approx(10 / 11)


def test_a_wider_prediction_scale_reads_the_same_gap_as_a_closer_game(
    game: GameFactory,
) -> None:
    """What the parameter is for: the ratings stay put, the confidence moves."""

    def _prob(scale: float) -> float:
        predictor = GlickoPredictor(
            "test_league", prediction_scale=scale, ratings={"Home": _Rating(1700, 100)}
        )
        return predictor.predict_game(game("Home", "Away")).team1_win_prob

    assert _prob(800) < _prob(400) < _prob(200)


def test_the_prediction_scale_leaves_the_update_alone(game: GameFactory) -> None:
    """It is a readout, not a learning rate.

    The update keeps Glicko's own 400 whatever the predictions are read at,
    so two models that differ only in this parameter hold identical ratings
    after the same games. That is what makes it safe to search: it cannot
    trade against the deviations for how far a game moves a rating.
    """
    played = [game("A", "B", 21, 7), game("B", "C", 3, 30), game("C", "A", 14, 10)]

    def _ratings(scale: float) -> dict[str, Rating]:
        predictor = GlickoPredictor("test_league", prediction_scale=scale)
        for g in played:
            predictor.update_game(g)
        return predictor.ratings

    assert _ratings(250) == _ratings(DEFAULT_PREDICTION_SCALE)


def test_the_prediction_scale_round_trips_through_the_state() -> None:
    """It changes every probability a release gives, so the release carries it."""
    predictor = GlickoPredictor("test_league", prediction_scale=310.5)

    state = predictor.state_dict()
    restored = GlickoPredictor.from_state_dict(state)

    assert state["prediction_scale"] == 310.5
    assert restored.state_dict() == state


@pytest.mark.parametrize("scale", [0.0, -400.0])
def test_a_prediction_scale_of_zero_or_less_is_refused(scale: float) -> None:
    with pytest.raises(ValueError, match="prediction_scale must be positive"):
        GlickoPredictor("test_league", prediction_scale=scale)


# --- passes: the season re-rated in hindsight ---------------------------------


def _week(predictor: GlickoPredictor, game: GameFactory, *games: tuple) -> None:
    for index, (home, away, home_score, away_score) in enumerate(games):
        predictor.update_game(
            game(home, away, home_score, away_score, game_id=str(index))
        )
    predictor.pass_week()


def test_one_pass_is_the_filter(game: GameFactory) -> None:
    """The default replays every published model exactly as before."""
    filter_ = GlickoPredictor("test_league", scoring_method="binary")
    smoothed = GlickoPredictor("test_league", scoring_method="binary", passes=1)
    for predictor in (filter_, smoothed):
        _week(predictor, game, ("A", "B", 28, 7), ("C", "D", 10, 3))
        _week(predictor, game, ("B", "C", 3, 31), ("D", "A", 0, 14))

    assert smoothed.ratings == filter_.ratings


def test_a_win_over_a_team_that_was_then_exposed_gives_credit_back(
    game: GameFactory,
) -> None:
    """The case the knob is for.

    A beats B in week 1: a good win, priced against B's rating that day. In
    week 2 B is blown out by D, a nobody. The filter leaves A's week-1 credit
    where it was; the smoother re-rates A's win against what B turned out
    to be and takes some of it back.
    """
    filter_ = GlickoPredictor("test_league", scoring_method="binary", passes=1)
    smoothed = GlickoPredictor("test_league", scoring_method="binary", passes=3)
    for predictor in (filter_, smoothed):
        _week(predictor, game, ("A", "B", 21, 20))
        _week(predictor, game, ("D", "B", 49, 0))

    assert smoothed.get_rating("A").rating < filter_.get_rating("A").rating
    assert smoothed.get_rating("A").rating > filter_.anchor("A")


def test_only_the_means_move(game: GameFactory) -> None:
    """Every pass re-sees every game; the deviations must not shrink for it."""
    filter_ = GlickoPredictor("test_league", passes=1, initial_rd=300)
    smoothed = GlickoPredictor("test_league", passes=4, initial_rd=300)
    for predictor in (filter_, smoothed):
        _week(predictor, game, ("A", "B", 21, 20), ("C", "D", 3, 0))
        _week(predictor, game, ("D", "B", 49, 0), ("A", "C", 7, 6))

    for team in "ABCD":
        assert smoothed.get_rating(team).rating_deviation == pytest.approx(
            filter_.get_rating(team).rating_deviation
        )


def test_the_prediction_the_update_scores_is_the_filters(game: GameFactory) -> None:
    """A game is predicted before it is played, from the ratings as they stand.

    Smoothing runs after the week, so the first week's predictions are the
    same under any number of passes; only what the *next* week is predicted
    from differs.
    """
    filter_ = GlickoPredictor("test_league", passes=1)
    smoothed = GlickoPredictor("test_league", passes=3)
    first = game("A", "B", 21, 20)

    assert smoothed.update_game(first) == filter_.update_game(first)


def test_a_new_season_replays_from_its_own_opening(game: GameFactory) -> None:
    """Last season's games are not re-rated this season.

    The smoother replays from the ratings the season opened with -- after
    the rollover -- and only over this season's games; otherwise a team's
    rating in week 3 would carry a fresh re-reading of last October.
    """
    predictor = GlickoPredictor("test_league", passes=3, season_regression=0.5)
    _week(predictor, game, ("A", "B", 21, 20), ("C", "D", 3, 0))
    predictor.pass_season()
    opened = predictor.ratings

    assert predictor._weeks == [] and predictor._this_week == []
    assert predictor._preseason == predictor._ratings
    # A week with no games re-rates nothing: the opening stands.
    predictor.pass_week()
    for team, rating in opened.items():
        assert predictor.get_rating(team).rating == pytest.approx(rating.rating)


def test_passes_round_trips_through_the_state(tmp_path) -> None:
    predictor = GlickoPredictor("test_league", passes=3)
    save_path = tmp_path / "glicko.json"
    predictor.save_state(save_path)

    assert GlickoPredictor.load_state(save_path)._passes == 3


def test_a_search_can_hand_over_a_whole_float() -> None:
    passes: Any = 2.0
    assert GlickoPredictor("test_league", passes=passes)._passes == 2


@pytest.mark.parametrize("passes", [0, -1, 2.5])
def test_a_fraction_of_a_pass_or_none_at_all_is_refused(passes: Any) -> None:
    with pytest.raises(ValueError, match="passes"):
        GlickoPredictor("test_league", passes=passes)
