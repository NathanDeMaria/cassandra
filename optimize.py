import asyncio
from dataclasses import asdict
from functools import partial
from pathlib import Path

import fire
import pandas as pd

from cassandra.brier import brier_score_df
from cassandra.constants import CASSANDRA_HOME
from cassandra.optimize import optimize
from cassandra.predictor import (
    OptimizationConfig,
    Predictor,
    PredictorConfig,
    load_predictor_class,
)
from cassandra.save_predictions import (
    Config,
    OddsDatabase,
    Season,
    join_with_odds,
    read_all_seasons,
)


def _negative_brier_score(
    league: str,
    seasons: list[Season],
    odds_db: OddsDatabase,
    predictor_class: type[Predictor],
    **kwargs,
) -> float:
    predictor = predictor_class(league, **kwargs)  # type: ignore[call-arg]
    prediction_results = join_with_odds(
        predictor, seasons, odds_db, post_callbacks=False
    )
    df = pd.DataFrame([asdict(result) for result in prediction_results])
    return -brier_score_df(df)


def _pinned_notice(fixed: dict[str, float | str], config_name: str) -> str | None:
    """The run report line that keeps a pinned parameter from going quiet.

    A pin is a decision made from one run's diagnostics, and the evidence for
    it decays: a parameter held at 0 because the search wanted it off may be
    worth searching again once the model around it changes. Nothing else would
    ever raise the question -- a pinned parameter produces no probes, so it
    produces no bound-hit diagnostic either, which is exactly how a decision
    becomes permanent by accident.

    Carries the `[optimize]` prefix the run report already collects into its
    tuning diagnostics, so every weekly report restates the open question
    against the config that has to change to answer it.
    """
    if not fixed:
        return None
    pinned = ", ".join(f"{name}={value}" for name, value in sorted(fixed.items()))
    return (
        f"[optimize] pinned, not searched: {pinned} -- re-open in "
        f"{config_name} if a change could make one of them matter again"
    )


async def _run_optimization(config_file: str) -> None:
    config_path = Path(config_file)
    with open(config_path, "r") as f:
        config_model = OptimizationConfig.model_validate_json(f.read())

    predictor_class = load_predictor_class(config_model.predictor_class)

    league = config_model.league

    aws_config = Config.init_from_file()
    seasons = [s async for s in read_all_seasons(league, aws_config.bucket)]
    if not seasons:
        # Otherwise every probe scores an empty set of games and the search
        # dies inside brier_score_df, well away from the actual problem.
        raise ValueError(
            f"No seasons for league {league!r} in s3://{aws_config.bucket}/seasons/; "
            "the league's data has to be uploaded before it can be optimized"
        )
    odds_db = await OddsDatabase.from_s3(aws_config.bucket)

    # Run once to make things like team priors
    predictor = predictor_class(league, **config_model.fixed)
    for _ in join_with_odds(predictor, seasons, odds_db, post_callbacks=True):
        pass

    notice = _pinned_notice(config_model.fixed, config_path.name)
    if notice is not None:
        print(notice)

    target_function = partial(
        _negative_brier_score,
        league=league,
        seasons=seasons,
        odds_db=odds_db,
        predictor_class=predictor_class,
        # The pinned arguments reach the constructor the same way a searched
        # one does; the optimizer simply never varies them. It is not told
        # about them at all, so they cost no dimension and appear in no probe.
        **config_model.fixed,
    )
    target, params = optimize(
        target_function, config_model.parameters, config_model.n_iter
    )

    result_model = PredictorConfig(
        predictor_class=config_model.predictor_class,
        league=config_model.league,
        target=target,
        # Merged, not just recorded: `load_predictor` rebuilds from `params`
        # alone, so a pinned argument left out here is one the published
        # model silently takes the constructor default for -- which for
        # `scoring_method` is `binary`, a different model than the one that
        # scored `target`. The two dicts are disjoint by construction; see
        # `OptimizationConfig._no_parameter_is_both`.
        params={**config_model.fixed, **params},
    )

    # The config is a checked-in input; its result is generated, so it lands
    # under CASSANDRA_HOME with the rest of the run's output.
    output_path = CASSANDRA_HOME / "models" / league / f"{config_path.stem}_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_model.model_dump_json(indent=4, by_alias=True))


def _main(config_file: str) -> None:
    asyncio.run(_run_optimization(config_file))


if __name__ == "__main__":
    fire.Fire(_main)
