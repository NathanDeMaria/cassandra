import json
from pathlib import Path

import pytest

from .game_control import (
    GameControlIndex,
    game_control_path,
    load_game_control,
    read_game_control_file,
    validated_control_weight,
)
from .types import GameControl

# A game the home team *lost* on the field and won on the scoreboard: it
# controlled 0.2 of the sixty minutes and still came out ahead. The scoring
# functions say 1.0; this is what the plays say instead, and the gap between
# them is the case the whole feature exists for.
_CLOSE_WIN = GameControl(home=0.2, seconds=3600)


def _index(**control: GameControl) -> GameControlIndex:
    return GameControlIndex(control)


def test_the_blend_moves_the_result_toward_control() -> None:
    """A home win the plays say was a coin flip counts for less than a win."""
    index = _index(g1=_CLOSE_WIN)

    assert index.blend(1.0, "g1", weight=1.0) == pytest.approx(0.2)


def test_a_weight_of_zero_is_the_result_as_scored() -> None:
    """The property that lets a search recover the plain model exactly."""
    index = _index(g1=_CLOSE_WIN)

    assert index.blend(1.0, "g1", weight=0.0) == 1.0


def test_a_weight_between_the_two_blends_them() -> None:
    index = _index(g1=_CLOSE_WIN)

    assert index.blend(1.0, "g1", weight=0.5) == pytest.approx(0.6)


def test_the_blend_stays_inside_the_scoring_range() -> None:
    """Both ends are shares of a game, so no weight can leave [0, 1]."""
    index = _index(g1=_CLOSE_WIN)

    for result in (0.0, 0.25, 0.5, 1.0):
        for weight in (0.0, 0.3, 0.7, 1.0):
            assert 0.0 <= index.blend(result, "g1", weight) <= 1.0


def test_a_game_with_no_control_keeps_its_result() -> None:
    """Most of an NCAAFB schedule, and all of it before 2006.

    The missing half needs no fallback rule of its own -- it is the weight-0
    case, reached without the predictor having to ask.
    """
    index = _index(g1=_CLOSE_WIN)

    assert index.blend(1.0, "not_in_the_index", weight=1.0) == 1.0


def test_an_empty_index_leaves_every_result_alone() -> None:
    """A league with no sweep, which is four of the six cassandra rates."""
    assert GameControlIndex().blend(1.0, "g1", weight=1.0) == 1.0


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
