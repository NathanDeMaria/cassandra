import json
from pathlib import Path

import pytest

from .conftest import GameFactory
from .game_control import (
    GameControlIndex,
    game_control_path,
    load_game_control,
    read_game_control_file,
    validated_control_weight,
)
from .types import GameControl

# A game the home team won by three while being behind on the field all
# night: 0.80 of the way to controlled, and a 20-17 line that says nothing
# about it. This is the case the whole feature exists for.
_CLOSE_WIN = GameControl(home=0.8, seconds=3600)


def _index(**control: GameControl) -> GameControlIndex:
    return GameControlIndex(control)


def test_the_alternate_line_re_splits_the_real_total(game: GameFactory) -> None:
    index = _index(g1=_CLOSE_WIN)

    alternate = index.alternate(game("A", "B", 20, 17, game_id="g1"), weight=1.0)

    assert (alternate.home_score, alternate.away_score) == (30, 7)


def test_the_total_survives_the_substitution(game: GameFactory) -> None:
    """Every consumer downstream is on the league's own scoring scale.

    A 6-3 defensive game has to stay a low-scoring game, or `pythagorean_score`
    and the margin-of-victory multiplier are both reading a different sport.
    """
    index = _index(g1=_CLOSE_WIN)

    for home_score, away_score in ((20, 17), (6, 3), (52, 49), (35, 0)):
        played = game("A", "B", home_score, away_score, game_id="g1")
        alternate = index.alternate(played, weight=1.0)

        assert alternate.home_score + alternate.away_score == home_score + away_score


def test_an_even_game_comes_out_even(game: GameFactory) -> None:
    index = _index(g1=GameControl(home=0.5, seconds=3600))

    alternate = index.alternate(game("A", "B", 24, 10, game_id="g1"), weight=1.0)

    assert alternate.home_score == alternate.away_score == 17


def test_a_weight_of_zero_is_the_game_as_played(game: GameFactory) -> None:
    """The property that lets a search recover the plain model exactly.

    Not "close to" the plain model -- the same object, so there is no rounding
    between a control predictor at weight 0 and the one it subclasses.
    """
    index = _index(g1=_CLOSE_WIN)
    played = game("A", "B", 20, 17, game_id="g1")

    assert index.alternate(played, weight=0.0) is played


def test_a_weight_between_the_two_blends_them(game: GameFactory) -> None:
    index = _index(g1=_CLOSE_WIN)

    alternate = index.alternate(game("A", "B", 20, 17, game_id="g1"), weight=0.5)

    # Halfway between the 20 that was scored and the 29.6 control implies.
    assert (alternate.home_score, alternate.away_score) == (25, 12)


def test_a_game_with_no_control_is_left_alone(game: GameFactory) -> None:
    """Most of an NCAAFB schedule, and all of it before 2006.

    The missing half needs no fallback rule of its own -- it is the weight-0
    case, reached without the predictor having to ask.
    """
    index = _index(g1=_CLOSE_WIN)
    played = game("A", "B", 20, 17, game_id="not_in_the_index")

    assert index.alternate(played, weight=1.0) is played


def test_an_empty_index_leaves_every_game_alone(game: GameFactory) -> None:
    """A league with no sweep, which is four of the six cassandra rates."""
    played = game("A", "B", 20, 17, game_id="g1")

    assert GameControlIndex().alternate(played, weight=1.0) is played


def test_a_scoreless_game_stays_scoreless(game: GameFactory) -> None:
    """There is no total to re-split, so control has nothing to say."""
    index = _index(g1=_CLOSE_WIN)

    alternate = index.alternate(game("A", "B", 0, 0, game_id="g1"), weight=1.0)

    assert (alternate.home_score, alternate.away_score) == (0, 0)


def test_the_alternate_line_is_whole_points(game: GameFactory) -> None:
    """`Game` scores are ints, and a synthetic line has to be one too."""
    index = _index(g1=GameControl(home=0.61, seconds=3600))

    alternate = index.alternate(game("A", "B", 20, 17, game_id="g1"), weight=1.0)

    assert isinstance(alternate.home_score, int)
    assert isinstance(alternate.away_score, int)


def test_nothing_but_the_score_changes(game: GameFactory) -> None:
    """The teams, the site and the id are what the replay matches on."""
    index = _index(g1=_CLOSE_WIN)
    played = game("A", "B", 20, 17, game_id="g1")

    alternate = index.alternate(played, weight=1.0)

    assert alternate._replace(home_score=20, away_score=17) == played


def test_an_out_of_range_weight_is_rejected() -> None:
    for weight in (1.5, -0.1):
        with pytest.raises(ValueError):
            validated_control_weight(weight)


def test_the_ends_of_the_range_are_allowed() -> None:
    assert validated_control_weight(0.0) == 0.0
    assert validated_control_weight(1.0) == 1.0


def test_a_league_with_no_file_has_no_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cassandra.predictor.game_control.game_control_path",
        lambda league: tmp_path / f"{league}_game_control.json",
    )
    load_game_control.cache_clear()

    assert load_game_control("a_league_nobody_swept") == {}


def test_a_saved_sweep_reads_back_as_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The games come back; the header the build stage reads stays behind it."""
    path = tmp_path / "ncaafb_game_control.json"
    path.write_text(
        json.dumps(
            {
                "league": "ncaafb",
                "fit": {"lucky_ones": "d253c4b", "run_id": "20260901-005010"},
                "games": {"401628319": [0.804, 3600], "401628320": [0.5, 1800]},
            }
        )
    )
    monkeypatch.setattr(
        "cassandra.predictor.game_control.game_control_path", lambda league: path
    )
    load_game_control.cache_clear()

    control = load_game_control("ncaafb")

    assert control == {
        "401628319": GameControl(0.804, 3600),
        "401628320": GameControl(0.5, 1800),
    }
    stored = read_game_control_file("ncaafb")
    assert stored is not None
    assert stored.fit.run_id == "20260901-005010"


def test_the_path_names_the_league() -> None:
    assert game_control_path("ncaafb").name == "ncaafb_game_control.json"


@pytest.fixture(autouse=True)
def _clear_the_cache():
    """`load_game_control` is cached, and these tests move the file under it.

    Cleared on the way out as well as in, so a test that monkeypatches the
    path can't leave a tmp_path read cached for whatever runs next.
    """
    load_game_control.cache_clear()
    yield
    load_game_control.cache_clear()
