import json

import pytest

from .base_predictor import load_anchors


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
