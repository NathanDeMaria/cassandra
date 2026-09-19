import math
from datetime import datetime
from typing import Any, NamedTuple

import pytest

from ..scoring import DEFAULT_SIGMOID_SCALE
from .base_predictor import MEAN_RATING
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


def test_the_home_edge_grows_with_the_home_team_s_anchor(game: GameFactory) -> None:
    """A tier above the mean gets more home advantage; a tier below, less.

    Anchors at 1900 and 1100 are one Elo decade either side of the mean, so
    at a slope of 10 the edges are 40 + 10 and 40 - 10. The away team's
    anchor is nobody's business here: the crowd is the home team's.
    """
    predictor = GlickoPredictor(
        "test_league",
        home_advantage=40,
        home_advantage_slope=10,
        anchors={"Upper": 1900, "Lower": 1100},
    )
    assert predictor.home_edge(game("Upper", "Lower")) == pytest.approx(50)
    assert predictor.home_edge(game("Lower", "Upper")) == pytest.approx(30)
    assert predictor.home_edge(game("Nobody", "Upper")) == pytest.approx(40)


def test_the_slope_reads_the_anchor_at_the_season_in_hand(game: GameFactory) -> None:
    """A program that moved up gets the tier it moved to, like its anchor does."""
    predictor = GlickoPredictor(
        "test_league",
        home_advantage=40,
        home_advantage_slope=10,
        anchors={"Riser": [(2010, 1100), (2015, 1900)]},
    )
    predictor.pass_season(2012)
    assert predictor.home_edge(game("Riser", "Other")) == pytest.approx(30)
    predictor.pass_season(2016)
    assert predictor.home_edge(game("Riser", "Other")) == pytest.approx(50)


def test_no_slope_is_the_constant_and_neutral_is_nothing(game: GameFactory) -> None:
    """The default replays every published model exactly as before."""
    predictor = GlickoPredictor("test_league", home_advantage=40, anchors={"A": 1900})
    assert predictor.home_edge(game("A", "B")) == 40
    sloped = GlickoPredictor(
        "test_league", home_advantage=40, home_advantage_slope=10, anchors={"A": 1900}
    )
    assert sloped.home_edge(_neutral("A", "B")) == 0.0


def test_the_slope_prices_the_update_as_well_as_the_prediction(
    game: GameFactory,
) -> None:
    """The update measures the result against the edge the prediction gave.

    Two upper-tier hosts win by the same score; the one whose model gives
    it more home edge was expected to do more with it, so it earns less.
    """
    flat = GlickoPredictor("test_league", home_advantage=40, anchors={"A": 1900})
    sloped = GlickoPredictor(
        "test_league", home_advantage=40, home_advantage_slope=40, anchors={"A": 1900}
    )
    for predictor in (flat, sloped):
        predictor.update_game(game("A", "B", 21, 14))
    assert sloped.get_rating("A").rating < flat.get_rating("A").rating


def test_the_slope_round_trips_through_the_state() -> None:
    predictor = GlickoPredictor("test_league", home_advantage_slope=7.5)
    state = predictor.state_dict()
    assert state["home_advantage_slope"] == 7.5
    assert GlickoPredictor.from_state_dict(state).state_dict() == state


def _season_of_unfiled_losses(predictor: GlickoPredictor, game: GameFactory) -> None:
    """Five unfiled teams each lose badly to a filed one, then the season rolls."""
    for i in range(5):
        predictor.update_game(game("Filed", f"Unfiled {i}", 49, 0))
    predictor.pass_week()
    predictor.pass_season()


def test_an_unfiled_team_enters_where_the_last_ones_ended_up(
    game: GameFactory,
) -> None:
    """Nothing seen, the league mean; a season of unfiled teams seen, their level.

    The estimate is the *end-of-season* rating of every unfiled team that
    played, so after five of them were routed the sixth enters below the
    mean, and an anchored team is untouched throughout.
    """
    predictor = GlickoPredictor("test_league", anchors={"Filed": 1800})
    assert predictor.get_rating("Unfiled 0").rating == MEAN_RATING
    assert predictor.unanchored_rd() == predictor.get_rating("Anyone").rating_deviation

    _season_of_unfiled_losses(predictor, game)

    total, squares, count = predictor.state_dict()["unanchored_seen"]
    assert count == 5
    prior = predictor.unanchored_prior()
    assert prior == pytest.approx(total / 5)
    assert prior < MEAN_RATING
    assert predictor.get_rating("Unfiled 5").rating == pytest.approx(prior)
    assert predictor.get_rating("Filed").rating > 1800


def test_the_estimate_waits_for_a_handful_of_teams(game: GameFactory) -> None:
    """One routed team is one team, not the population."""
    predictor = GlickoPredictor("test_league", anchors={"Filed": 1800})
    predictor.update_game(game("Filed", "Unfiled", 49, 0))
    predictor.pass_week()
    predictor.pass_season()
    assert predictor.state_dict()["unanchored_seen"][2] == 1
    assert predictor.unanchored_prior() == MEAN_RATING
    assert predictor.get_rating("Another").rating == MEAN_RATING


def test_an_unfiled_team_is_less_sure_than_a_filed_one(game: GameFactory) -> None:
    """The spread of where unfiled teams landed rides on top of initial_rd."""
    predictor = GlickoPredictor(
        "test_league", initial_rd=200, anchors={"Filed": 1800, "Other": 1300}
    )
    # Two kinds of unfiled team, so the population has a spread.
    for i in range(3):
        predictor.update_game(game("Filed", f"Weak {i}", 49, 0))
        predictor.update_game(game(f"Strong {i}", "Other", 49, 0))
    predictor.pass_week()
    predictor.pass_season()
    total, squares, count = predictor.state_dict()["unanchored_seen"]
    spread = math.sqrt(squares / count - (total / count) ** 2)
    assert spread > 0
    assert predictor.unanchored_rd() == pytest.approx(math.sqrt(200**2 + spread**2))
    assert predictor.get_rating("New").rating_deviation == predictor.unanchored_rd()
    # The filed team is back at initial_rd after the rollover; the unfiled
    # one is wider than that.
    assert predictor.get_rating("Filed").rating_deviation == 200


def test_an_unfiled_team_regresses_toward_its_own_kind(game: GameFactory) -> None:
    predictor = GlickoPredictor(
        "test_league", anchors={"Filed": 1800}, season_regression=1.0
    )
    _season_of_unfiled_losses(predictor, game)
    prior = predictor.unanchored_prior()
    predictor.update_game(game("Unfiled 0", "Filed", 30, 0))
    assert predictor.get_rating("Unfiled 0").rating > prior
    predictor.pass_week()
    predictor.pass_season()
    # Back to what unfiled teams are (the estimate moved a little for the
    # season just folded in), not to the league mean.
    assert predictor.get_rating("Unfiled 0").rating == pytest.approx(
        predictor.unanchored_prior()
    )


def test_a_team_is_counted_once_per_season_it_played(game: GameFactory) -> None:
    """A 2007 one-off isn't folded in again every offseason after."""
    predictor = GlickoPredictor("test_league", anchors={"Filed": 1800})
    predictor.update_game(game("Filed", "Once", 49, 0))
    predictor.pass_week()
    predictor.pass_season()
    predictor.update_game(game("Filed", "Someone Else", 49, 0))
    predictor.pass_week()
    predictor.pass_season()
    assert predictor.state_dict()["unanchored_seen"][2] == 2


def test_the_unanchored_estimate_round_trips(game: GameFactory) -> None:
    predictor = GlickoPredictor("test_league", anchors={"Filed": 1800})
    _season_of_unfiled_losses(predictor, game)
    state = predictor.state_dict()
    restored = GlickoPredictor.from_state_dict(state)
    assert restored.unanchored_prior() == predictor.unanchored_prior()
    assert restored.unanchored_rd() == predictor.unanchored_rd()
    assert restored.state_dict() == state


class _neutral(NamedTuple):
    home: str
    away: str
    neutral_site: bool = True
    date: datetime = datetime(2023, 1, 1)
    game_id: str = "1"


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
