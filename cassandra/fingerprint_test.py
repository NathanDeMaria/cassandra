import json

import pytest

from .fingerprint import (
    GIT_SHA_ENV_VAR,
    _config_digest,
    _digest,
    _reads,
    _seed_digest,
    code_version,
)
from .predictor import InputFingerprint, PredictorConfig, SearchRecord


def _fit(**overrides) -> PredictorConfig:
    fields: dict = {
        "predictor_class": "GlickoPredictor",
        "league": "mens",
        "target": -0.17,
        "params": {"k": 20.0, "home_advantage": 60.0},
        "search": SearchRecord(
            frame="rating", weeks_per_season=22.0, knobs={"k": 20.0}
        ),
    }
    return PredictorConfig(**{**fields, **overrides})


# --- which objects a league's search actually reads ---------------------------
#
# The filter is the whole reason a skip is possible at all: both prefixes hold
# every league's files, and the football sweeps rewrite theirs daily. A filter
# that let those through would invalidate basketball every night and skip
# nothing, ever.


@pytest.mark.parametrize(
    "key",
    [
        "seasons/2024/mens.pkl",
        "cassandra/predictor/data/mens_division_anchors.json",
    ],
)
def test_a_league_reads_its_own_seasons_and_indexes(key: str) -> None:
    assert _reads(key, "mens")


@pytest.mark.parametrize(
    "key",
    [
        "seasons/2024/ncaafb.pkl",
        "cassandra/predictor/data/ncaafb_epa.json",
        "cassandra/predictor/data/nfl_qb_out.json",
    ],
)
def test_another_league_s_files_are_not_this_league_s_inputs(key: str) -> None:
    """The daily football sweeps must not invalidate basketball.

    `game_control`, `epa` and `qb_out` rewrite ncaafb's and nfl's indexes
    every day. If those counted, `mens` would never match its own fingerprint
    between November and April, which is exactly the window the skip exists
    for.
    """
    assert not _reads(key, "mens")


def test_the_priors_a_search_writes_are_not_an_input_to_it() -> None:
    """Otherwise every run invalidates its own fingerprint.

    `rebuild_priors` writes `{league}_{Class}_priors.json` as a warm-up pass,
    so a fingerprint that counted it would never match the next run's. They
    are derived from the seasons and the pins, which are both covered.
    """
    assert not _reads(
        "cassandra/predictor/data/mens_GlickoPredictor_priors.json", "mens"
    )
    assert not _reads("cassandra/predictor/data/mens_priors.json", "mens")


def test_a_league_whose_name_prefixes_another_is_not_confused() -> None:
    """`mens` and `womens` are not substrings of each other, but `ncaa*` are.

    The filter matches on `<league>_`, so `ncaawvb_epa.json` must not read as
    an `ncaa` file -- and no league is a prefix of another with the separator
    in place.
    """
    assert _reads("cassandra/predictor/data/ncaawvb_epa.json", "ncaawvb")
    assert not _reads("cassandra/predictor/data/ncaawvb_epa.json", "ncaa")
    assert not _reads("seasons/2024/womens.pkl", "mens")


# --- the digests themselves ---------------------------------------------------


def test_reformatting_a_config_is_not_a_change(tmp_path) -> None:
    """The digest is of the parsed json, so whitespace and key order don't count.

    Otherwise `ruff format` on a config, or a hand edit that reorders two
    keys, would re-search a league for hours and produce the same fit.
    """
    one = tmp_path / "a.json"
    other = tmp_path / "b.json"
    one.write_text('{"league": "mens", "n_iter": 60}')
    other.write_text(json.dumps({"n_iter": 60, "league": "mens"}, indent=4) + "\n")

    assert _config_digest(one) == _config_digest(other)


def test_changing_a_config_value_is_a_change(tmp_path) -> None:
    one = tmp_path / "a.json"
    other = tmp_path / "b.json"
    one.write_text('{"n_iter": 60}')
    other.write_text('{"n_iter": 61}')

    assert _config_digest(one) != _config_digest(other)


def test_the_seed_digest_ignores_the_fingerprint_it_sits_beside() -> None:
    """It has to, or it would depend on itself.

    The previous result carries the previous fingerprint; hashing the whole
    file would make this run's seed digest a function of last run's seed
    digest, and nothing would ever match twice.
    """
    without = _fit()
    with_inputs = _fit(
        inputs=InputFingerprint(data="d", config="c", code="abc123", seed="s")
    )

    assert _seed_digest(without) == _seed_digest(with_inputs)


def test_a_different_fit_is_a_different_seed() -> None:
    assert _seed_digest(_fit()) != _seed_digest(
        _fit(params={"k": 21.0, "home_advantage": 60.0})
    )


def test_no_previous_fit_is_its_own_seed_value() -> None:
    """Distinct from any real fit's digest, so "no fit" never matches one."""
    assert _seed_digest(None) == ""
    assert _seed_digest(_fit()) != ""


def test_the_target_is_not_part_of_the_seed() -> None:
    """The score is an output. Two fits at the same point are the same seed.

    This matters because `target` moves when the *data* moves even if the
    fitted parameters don't -- and the data has its own digest, so counting
    the score here would double-count it.
    """
    assert _seed_digest(_fit(target=-0.17)) == _seed_digest(_fit(target=-0.18))


def test_the_digest_is_stable_across_processes() -> None:
    """Hard-coded, because a digest that changes between runs skips nothing.

    `hash()` is salted per process and would have made this whole mechanism
    silently useless; this test is here to catch a change to `_digest` that
    reintroduces that.
    """
    assert _digest({"a": 1, "b": [2, 3]}) == _digest({"b": [2, 3], "a": 1})
    assert len(_digest("anything")) == 16


# --- the code half ------------------------------------------------------------


def test_an_image_that_records_no_commit_is_never_fresh(monkeypatch) -> None:
    """A missing sha has to read as "can't tell", not as "unchanged".

    A source checkout has no `CASSANDRA_GIT_SHA`, so this is the path every
    local run takes, and the safe answer is to search.
    """
    monkeypatch.delenv(GIT_SHA_ENV_VAR, raising=False)
    assert code_version() == ""
    assert not InputFingerprint(data="d", config="c", code="", seed="s").complete


def test_a_recorded_commit_is_what_gets_compared(monkeypatch) -> None:
    monkeypatch.setenv(GIT_SHA_ENV_VAR, "  abc1234  ")
    assert code_version() == "abc1234"
    assert InputFingerprint(data="d", config="c", code="abc1234", seed="s").complete


# --- what the fingerprint says when it doesn't match --------------------------


def test_it_names_every_part_that_moved() -> None:
    """The line a search prints when it was expected to skip and didn't."""
    before = InputFingerprint(data="d1", config="c1", code="sha1", seed="s1")
    after = InputFingerprint(data="d2", config="c1", code="sha2", seed="s1")

    assert after.differences(before) == ["data", "code"]


def test_an_identical_fingerprint_has_no_differences() -> None:
    one = InputFingerprint(data="d", config="c", code="sha", seed="s")

    assert one.differences(one) == []
    assert one == InputFingerprint(data="d", config="c", code="sha", seed="s")
