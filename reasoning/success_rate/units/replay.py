"""Replay NFL glicko_margin_units_offseason under different readings and knobs.

One process, one seasons pickle, one private priors file (built once from the
search's pins with the league's own EPA index) for every variant -- see the
shared-priors note -- so the variants differ only in what they're given.

    python replay.py NAME BETA [SINCE] [knob=value ...]

writes out/NAME.parquet: the replay's predictions, one row per game.
"""

import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import cassandra
from cassandra.model_eval import rebuild_priors
from cassandra.odds import OddsDatabase
from cassandra.predictor import OptimizationConfig, PredictorConfig
from cassandra.predictor import frame as frames
from cassandra.predictor.opponent_prior import OpponentPriorManager
from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor
from cassandra.save_predictions import join_with_odds, prepared_for_replay

sys.path.insert(0, str(Path(__file__).parent.parent))
from pin import pin  # noqa: E402

# Every replay reads the pinned copy of the shared predictor data; see pin.py.
pin()

sys.path.insert(0, str(Path(__file__).parent))
from readings import build  # noqa: E402

DATA = Path.home() / ".cassandra/success_rate/units"
REPO = Path(cassandra.__file__).parent.parent
FITTED = Path.home() / ".cassandra/models/nfl/glicko_margin_units_offseason_result.json"
SEARCH = REPO / "models/nfl/glicko_margin_units_offseason.json"
PRIORS = DATA / "nfl_units_priors.json"
OUT = DATA / "out"


def seasons():
    raw = pickle.loads((DATA / "nfl_seasons.pkl").read_bytes())
    return prepared_for_replay("nfl", raw)


def ensure_priors(named) -> None:
    if PRIORS.exists():
        return
    config = OptimizationConfig.model_validate_json(SEARCH.read_text())
    weeks = frames.weeks_per_season([len(s.weeks) for s in named])
    rebuild_priors(UnitMarginGlickoPredictor, "nfl",
                   frames.to_params(config.frame, config.fixed, weeks),
                   named, OddsDatabase({}), priors_path=PRIORS)


def replay(name: str, beta: float, since: int = 2014, **knobs) -> pd.DataFrame:
    named, namer = seasons()
    ensure_priors(named)
    start = time.time()
    params = {**PredictorConfig.model_validate_json(FITTED.read_text()).params, **knobs}
    predictor = UnitMarginGlickoPredictor(
        "nfl",
        **params,
        opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
        game_epa=build(beta, since),
    )
    df = pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, named, OddsDatabase({}), post_callbacks=False, namer=namer)])
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    brier = ((df.team1_win_prob - df.team1_win) ** 2).mean()
    print(f"{name}: beta={beta} since={since} {knobs} brier {brier:.6f} ({time.time() - start:.0f}s)",
          flush=True)
    return df


if __name__ == "__main__":
    name, beta = sys.argv[1], float(sys.argv[2])
    rest = sys.argv[3:]
    since = int(rest.pop(0)) if rest and "=" not in rest[0] else 2014
    knobs = {k: float(v) for k, v in (arg.split("=") for arg in rest)}
    replay(name, beta, since, **knobs)
