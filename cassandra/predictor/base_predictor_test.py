import json

import pytest

from .base_predictor import anchor_in, load_anchors


def test_a_league_without_an_anchor_file_gets_no_anchors() -> None:
    assert load_anchors("league_that_has_never_been_fit") == {}


def test_anchors_are_read_from_the_league_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr("cassandra.predictor.base_predictor._ANCHOR_DIR", tmp_path)
    (tmp_path / "ncaafb_division_anchors.json").write_text(json.dumps({"Team A": 1200}))
    load_anchors.cache_clear()

    assert load_anchors("ncaafb") == {"Team A": 1200}

    load_anchors.cache_clear()


def test_a_flat_anchor_is_the_same_in_every_season() -> None:
    """The shape every release published before histories existed carries."""
    assert anchor_in(1200, None) == 1200
    assert anchor_in(1200.0, 2015) == 1200


def test_a_history_holds_each_step_until_the_next_one() -> None:
    steps = [[2002, 1200.0], [2004, 1500.0], [2012, 1800.0]]

    assert [anchor_in(steps, y) for y in (2002, 2003, 2004, 2011, 2012)] == [
        1200.0,
        1200.0,
        1500.0,
        1500.0,
        1800.0,
    ]


def test_a_history_clamps_at_both_ends() -> None:
    """Before the first step a team hadn't reached a later division yet, and
    after the last there is no played season left to fit one from."""
    steps = [[2002, 1200.0], [2004, 1500.0]]

    assert anchor_in(steps, 1990) == 1200.0
    assert anchor_in(steps, None) == 1200.0
    assert anchor_in(steps, 2099) == 1500.0
