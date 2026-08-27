import json
from pathlib import Path

import pytest

from . import manifest
from .manifest import UnknownWork


def _write_config(
    models_dir: Path,
    league: str,
    name: str,
    *,
    predictor_class: str = "FlatPredictor",
    n_iter: int = 100,
    config_league: str | None = None,
) -> Path:
    league_dir = models_dir / league
    league_dir.mkdir(parents=True, exist_ok=True)
    path = league_dir / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "predictor_class": predictor_class,
                "league": config_league or league,
                "parameters": {"k": [1.0, 2.0]},
                "n_iter": n_iter,
            }
        )
    )
    return path


@pytest.fixture(name="models_dir")
def _models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(manifest, "_MODELS_DIR", models_dir)
    # Both env vars leak between the launcher and the child paths, so every
    # test starts from neither being set.
    monkeypatch.delenv(manifest.MANIFEST_ENV_VAR, raising=False)
    monkeypatch.delenv(manifest.ARRAY_INDEX_ENV_VAR, raising=False)
    return models_dir


def test_orders_cheap_configs_first(models_dir: Path) -> None:
    _write_config(models_dir, "nfl", "expensive", n_iter=500)
    _write_config(models_dir, "mens", "cheap", n_iter=10)
    _write_config(models_dir, "nfl", "cheap", n_iter=10)

    assert [work.name for work in manifest.load_manifest()] == [
        "mens/cheap",
        "nfl/cheap",
        "nfl/expensive",
    ]


def test_skips_result_and_state_files(models_dir: Path) -> None:
    _write_config(models_dir, "mens", "elo")
    # Checked-in baselines and predictor state sit in the same directory and
    # are not searchable configs.
    (models_dir / "mens" / "flat_result.json").write_text("{}")
    (models_dir / "mens" / "elo_state.json").write_text("{}")

    assert [work.name for work in manifest.load_manifest()] == ["mens/elo"]


def test_rejects_league_that_disagrees_with_its_directory(models_dir: Path) -> None:
    _write_config(models_dir, "mens", "elo", config_league="nfl")

    # evaluate_models.py labels by directory, so a config that disagrees would
    # be optimized as one league and scored as another.
    with pytest.raises(ValueError, match="does not match directory"):
        manifest.load_manifest()


def test_filters_by_league_and_model(models_dir: Path) -> None:
    _write_config(models_dir, "mens", "elo")
    _write_config(models_dir, "mens", "glicko")
    _write_config(models_dir, "nfl", "elo")

    assert [w.name for w in manifest.load_manifest(leagues=["mens"])] == [
        "mens/elo",
        "mens/glicko",
    ]
    assert [w.name for w in manifest.load_manifest(models=["elo"])] == [
        "mens/elo",
        "nfl/elo",
    ]


def test_prior_path_set_only_for_predictors_that_build_priors(
    models_dir: Path,
) -> None:
    _write_config(models_dir, "mens", "flat", predictor_class="FlatPredictor")
    _write_config(models_dir, "mens", "glicko", predictor_class="GlickoPredictor")

    by_name = {work.name: work for work in manifest.load_manifest()}
    assert by_name["mens/flat"].prior_path is None
    assert by_name["mens/glicko"].prior_path is not None


def test_resolve_index_uses_the_pinned_manifest_not_local_order(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(models_dir, "mens", "cheap", n_iter=10)
    _write_config(models_dir, "nfl", "expensive", n_iter=500)

    # The launcher's list, in an order the local sort would not produce. The
    # child has to follow it, or index 0 optimizes the wrong model.
    monkeypatch.setenv(
        manifest.MANIFEST_ENV_VAR, json.dumps(["nfl/expensive", "mens/cheap"])
    )

    assert manifest.resolve_index(0).name == "nfl/expensive"
    assert manifest.resolve_index(1).name == "mens/cheap"


def test_resolve_index_reads_the_array_index_from_the_environment(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(models_dir, "mens", "a", n_iter=10)
    _write_config(models_dir, "mens", "b", n_iter=20)
    monkeypatch.setenv(manifest.ARRAY_INDEX_ENV_VAR, "1")

    assert manifest.resolve_index().name == "mens/b"


def test_resolve_index_fails_loudly_when_the_launcher_is_ahead_of_the_image(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(models_dir, "mens", "elo")
    monkeypatch.setenv(
        manifest.MANIFEST_ENV_VAR, json.dumps(["mens/elo", "mens/added_later"])
    )

    # The alternative is silently running a 2-child array against a 1-item
    # list, which optimizes mens/elo twice and reports success.
    with pytest.raises(UnknownWork, match="different commits"):
        manifest.resolve_index(0)


def test_resolve_index_rejects_an_out_of_range_index(models_dir: Path) -> None:
    _write_config(models_dir, "mens", "elo")

    with pytest.raises(IndexError):
        manifest.resolve_index(1)


def test_resolve_index_without_an_index_or_environment_says_so(
    models_dir: Path,
) -> None:
    _write_config(models_dir, "mens", "elo")

    with pytest.raises(ValueError, match="array job"):
        manifest.resolve_index()


def test_find_names_the_known_configs_when_it_misses(models_dir: Path) -> None:
    _write_config(models_dir, "mens", "elo")

    assert manifest.find("mens", "elo").name == "mens/elo"
    with pytest.raises(UnknownWork, match="mens/elo"):
        manifest.find("mens", "nope")


def test_encode_round_trips_through_resolve_index(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(models_dir, "mens", "a", n_iter=10)
    _write_config(models_dir, "nfl", "b", n_iter=20)

    work = manifest.load_manifest()
    monkeypatch.setenv(manifest.MANIFEST_ENV_VAR, manifest.encode(work))

    assert [manifest.resolve_index(i).name for i in range(len(work))] == [
        item.name for item in work
    ]


def test_array_index_falls_back_to_what_batch_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job definitions carry a bare command, so the index is in the env."""
    monkeypatch.setenv(manifest.ARRAY_INDEX_ENV_VAR, "4")

    assert manifest.array_index() == 4
    # An explicit --index wins, which is how a single model gets run by hand.
    assert manifest.array_index(1) == 1


def test_array_index_is_none_outside_an_array_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not an error: it's how a stage knows to do its whole-run default."""
    monkeypatch.delenv(manifest.ARRAY_INDEX_ENV_VAR, raising=False)

    assert manifest.array_index() is None
