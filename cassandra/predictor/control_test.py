import json

import pytest

from .conftest import GameFactory
from .control import ControlGlickoPredictor
from .game_control import GameControlIndex
from .glicko import GlickoPredictor
from .types import GameControl

_LEAGUE = "test_league"
# Pinned on both sides of every comparison below rather than left to the
# defaults: this model defaults to sigmoid and the Glicko it subclasses to
# binary, so a test that leaned on them matching would be comparing two
# different scoring functions and calling the difference "control".
_SCORING = "sigmoid"


def _index(home: float) -> GameControlIndex:
    return GameControlIndex({"g1": GameControl(home=home, seconds=3600)})


def _rating(predictor: GlickoPredictor, team: str) -> float:
    return predictor.get_rating(team).rating


def test_control_can_take_a_win_away_from_the_winner(game: GameFactory) -> None:
    """The whole point, in one assertion.

    A team that won 20-17 while being behind all night is not a team that
    beat anybody, and this is the model that can say so: at full weight the
    line it learns from is 7-30, so it loses rating for a game the scoreboard
    says it won.

    Asserted against plain Glicko rather than in the abstract, because "the
    rating went down" only means something next to what the same game does
    without control.
    """
    played = game("Team A", "Team B", 20, 17, game_id="g1")

    controlled = ControlGlickoPredictor(
        _LEAGUE, scoring_method=_SCORING, game_control=_index(0.2), control_weight=1.0
    )
    controlled.update_game(played)
    plain = GlickoPredictor(_LEAGUE, scoring_method=_SCORING)
    plain.update_game(played)

    assert _rating(plain, "Team A") > _rating(controlled, "Team A")
    assert _rating(controlled, "Team A") < _rating(controlled, "Team B")


def test_a_weight_of_zero_is_plain_glicko(game: GameFactory) -> None:
    """What makes the parameter safe to search.

    A search that finds control worthless has to land back on the model this
    subclasses -- exactly on it, not near it, because at weight 0 the
    alternate line is the game itself and no arithmetic happened.
    """
    played = game("Team A", "Team B", 20, 17, game_id="g1")

    controlled = ControlGlickoPredictor(
        _LEAGUE, scoring_method=_SCORING, game_control=_index(0.2), control_weight=0.0
    )
    controlled.update_game(played)
    plain = GlickoPredictor(_LEAGUE, scoring_method=_SCORING)
    plain.update_game(played)

    assert _rating(controlled, "Team A") == _rating(plain, "Team A")
    assert _rating(controlled, "Team B") == _rating(plain, "Team B")


def test_the_weight_moves_the_rating_smoothly_between_the_two(
    game: GameFactory,
) -> None:
    """The other half of searchable: the range in between has to mean something.

    A parameter whose only two useful values were its endpoints would be a
    flag, and the search would spend its budget discovering that.
    """
    played = game("Team A", "Team B", 20, 17, game_id="g1")

    ratings = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        predictor = ControlGlickoPredictor(
            _LEAGUE,
            scoring_method="pythagorean",
            game_control=_index(0.2),
            control_weight=weight,
        )
        predictor.update_game(played)
        ratings.append(_rating(predictor, "Team A"))

    assert all(earlier > later for earlier, later in zip(ratings, ratings[1:])), ratings


def test_a_game_the_sweep_never_saw_is_the_baseline_too(game: GameFactory) -> None:
    """Most of an NCAAFB schedule takes this path, at full weight."""
    played = game("Team A", "Team B", 20, 17, game_id="no_plays_for_this_one")

    controlled = ControlGlickoPredictor(
        _LEAGUE, scoring_method=_SCORING, game_control=_index(0.2), control_weight=1.0
    )
    controlled.update_game(played)
    plain = GlickoPredictor(_LEAGUE, scoring_method=_SCORING)
    plain.update_game(played)

    assert _rating(controlled, "Team A") == _rating(plain, "Team A")


@pytest.mark.parametrize("scoring_method", ["pythagorean", "sigmoid"])
def test_the_size_of_the_control_number_reaches_the_rating(
    scoring_method: str, game: GameFactory
) -> None:
    """Which scoring methods can hear the magnitude, and it isn't all of them.

    A team that controlled 0.9 of a three-point win did something a team that
    controlled 0.55 of the same win didn't, and these two scoring functions
    read the margin, so they can tell.
    """
    played = game("Team A", "Team B", 20, 17, game_id="g1")

    comfortable = ControlGlickoPredictor(
        _LEAGUE, scoring_method=scoring_method, game_control=_index(0.9)
    )
    comfortable.update_game(played)
    narrow = ControlGlickoPredictor(
        _LEAGUE, scoring_method=scoring_method, game_control=_index(0.55)
    )
    narrow.update_game(played)

    assert _rating(comfortable, "Team A") > _rating(narrow, "Team A")


def test_binary_scoring_only_hears_the_crossover(game: GameFactory) -> None:
    """The limit of the default, stated so a search result reads correctly.

    `binary_score` is 1/0.5/0 off the sign, so two control numbers on the
    same side of even produce the same update. Control reaches a binary
    Glicko only when the blend hands the game to the other team -- which is
    real, and is exactly the case the first test covers.
    """
    played = game("Team A", "Team B", 20, 17, game_id="g1")

    comfortable = ControlGlickoPredictor(
        _LEAGUE, scoring_method="binary", game_control=_index(0.9)
    )
    comfortable.update_game(played)
    narrow = ControlGlickoPredictor(
        _LEAGUE, scoring_method="binary", game_control=_index(0.55)
    )
    narrow.update_game(played)

    assert _rating(comfortable, "Team A") == _rating(narrow, "Team A")


def test_the_control_weight_round_trips(game: GameFactory) -> None:
    """A release replays at the weight it was fit with."""
    predictor = ControlGlickoPredictor(
        _LEAGUE, game_control=_index(0.2), control_weight=0.25
    )
    predictor.update_game(game("Team A", "Team B", 20, 17, game_id="g1"))

    state = predictor.state_dict()
    restored = ControlGlickoPredictor.from_state_dict(json.loads(json.dumps(state)))

    assert state["control_weight"] == 0.25
    assert restored._control_weight == 0.25


def test_an_out_of_range_control_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        ControlGlickoPredictor(_LEAGUE, control_weight=1.5)
    with pytest.raises(ValueError):
        ControlGlickoPredictor(_LEAGUE, control_weight=-0.1)


def test_the_index_is_not_carried_in_the_state() -> None:
    """Tens of thousands of floats would dominate a release of ratings.

    The cost is deliberate and worth seeing in a test: a replay picks up
    whatever sweep is on disk, not the one the fit ran against.
    """
    state = ControlGlickoPredictor(_LEAGUE, game_control=_index(0.2)).state_dict()

    assert "game_control" not in state


def test_a_league_with_no_sweep_builds_an_empty_index() -> None:
    """Constructing one has to work without an artifact anywhere.

    Every contract test builds this for a league nobody has swept, so does a
    laptop running the model for the first time, and so does `load_manifest`
    when it instantiates each config to find its priors path.
    """
    assert len(ControlGlickoPredictor(_LEAGUE)._game_control) == 0


def test_the_default_scoring_method_is_the_one_the_search_settled_on() -> None:
    """The config omits `scoring_method`, so this default is what a probe runs.

    Glicko's own default is binary, which is the one scoring function that
    can't hear a blend that doesn't change the winner -- so inheriting it
    would have made every probe of `control_weight` a near-no-op.
    """
    assert ControlGlickoPredictor(_LEAGUE)._scoring_method == "sigmoid"
