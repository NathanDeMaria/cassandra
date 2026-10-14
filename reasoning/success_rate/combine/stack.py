"""How much a better combination of the record and the units could be worth.

    python stack.py ncaafb|nfl

The units model predicts from one blend: the record's rating and the units'
averaged by precision, `unit_weight` scaling the units' share. That is the
right average only if the two are independent estimates, and they learned
from the same games. This replays the production code with a subclass that
writes down, before each game, what the blend is made of -- each team's
record rating and deviation, its units' offsets and earned precision, the
edge -- for three unit models over one record:

- `published`: the published model, its units on EPA;
- `epa`: units on EPA with every reading knob refit through 2021;
- `success`: units on the success rate alone, refit the same way.

The record is the same in all three (the units never feed back into it),
so a logistic regression over their gaps, fit on games through 2021 and
scored on 2022 on, is a ceiling for re-weighting what the models already
know: with the published units alone, with the weights free to move with
how much each side has learned, and with efficiency and explosiveness apart
(EPA less success, as two gaps the regression can weigh separately).
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor

LEAGUE = sys.argv[1]
HERE = Path(__file__).parent
OUT = Path.home() / ".cassandra/success_rate/combine"
TRAIN_THROUGH = 2021


class Recorder(UnitMarginGlickoPredictor):
    """The production model, writing down a game's ingredients before it predicts."""

    rows: list[dict] = []

    def _update_game(self, game):
        edge = self.home_edge(game) + self.matchup_adjustment(game)
        row = {"game_id": game.game_id, "edge": edge}
        for side, team in (("h", game.home), ("a", game.away)):
            parent = self.get_rating(team)
            sides = self._offsets(team)
            row[f"r_{side}"] = parent.rating
            row[f"d_{side}"] = parent.rating_deviation
            row[f"u_{side}"] = self._rating_per_epa * (sides.offense.rating + sides.defense.rating)
            row[f"anchor_{side}"] = self.anchor(team)
            row[f"info_{side}"] = self.unit_information(team)
        self.rows.append(row)
        return super()._update_game(game)


def replay_ncaafb(knobs: dict) -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent / "ncaafb"))
    from replay import run_from_2014

    Recorder.rows = []
    df = run_from_2014(knobs, predictor_class=Recorder)
    return df.merge(pd.DataFrame(Recorder.rows), on="game_id")


def replay_nfl(knobs: dict) -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent / "units"))
    from prod import BASE, INDEX, NAMED, NAMER, PRIORS

    from cassandra.odds import OddsDatabase
    from cassandra.predictor.opponent_prior import OpponentPriorManager
    from cassandra.save_predictions import join_with_odds

    Recorder.rows = []
    predictor = Recorder(
        "nfl", **{**BASE, **knobs},
        opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
        game_epa=INDEX,
    )
    df = pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, NAMED, OddsDatabase({}), post_callbacks=False, namer=NAMER)])
    df["fbs"] = True
    return df.merge(pd.DataFrame(Recorder.rows), on="game_id")


#: Each league's two refits, through 2021: units on EPA and on success rate.
RUNS = {
    "ncaafb": ("ncaafb_out/nc_fit_epa.json", "ncaafb_out/nc_fit_free.json"),
    "nfl": ("out/fit_epa.json", "out/fit_sr.json"),
}


def knobs_of(path: str) -> dict:
    meta = json.loads((Path.home() / ".cassandra/success_rate/units" / path).read_text())
    knobs = dict(meta["knobs"])
    if "beta" in meta:
        knobs["success_weight"] = meta["beta"]
    return knobs


def frame() -> pd.DataFrame:
    replay = replay_ncaafb if LEAGUE == "ncaafb" else replay_nfl
    epa_path, success_path = RUNS[LEAGUE]
    runs = {"published": {}, "epa": knobs_of(epa_path), "success": knobs_of(success_path)}
    out = None
    for name, knobs in runs.items():
        df = replay(knobs)
        df = df[df.year >= (2014 if LEAGUE == "ncaafb" else 2006)]
        keep = ["u_h", "u_a", "info_h", "info_a"]
        if out is None:
            out = df.rename(columns={c: f"{c}_{name}" for c in keep + ["team1_win_prob"]})
        else:
            # The record is the units' parent, untouched by them: check it.
            joined = out.merge(df[["game_id", "r_h"] + keep + ["team1_win_prob"]], on="game_id",
                               suffixes=("", "_new"))
            assert np.allclose(joined.r_h, joined.r_h_new), f"{name}: the record moved"
            out = joined.drop(columns="r_h_new").rename(
                columns={c: f"{c}_{name}" for c in keep + ["team1_win_prob"]})
        print(name, "replayed", flush=True)
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Gaps, home less away, in the record's rating points.

    Per team, what the units say less what the record says (`anchor + u - r`),
    and the units' share of what the two know at `unit_weight` 1. The
    published prediction is the record's gap plus each team's disagreement
    times its share at the published weight, so `<units>_blend` with the
    record reproduces it nearly, and `<units>_gap` -- the disagreement at a
    constant weight -- is the blend that ignores how much either has learned.
    """
    f = pd.DataFrame({"year": df.year, "fbs": df.fbs, "win": df.team1_win.astype(float),
                      "tie": df.home_score == df.away_score})
    f["record"] = df.r_h - df.r_a + df.edge
    for name in ("published", "epa", "success"):
        blend = 0.0
        for side in ("h", "a"):
            disagree = df[f"anchor_{side}"] + df[f"u_{side}_{name}"] - df[f"r_{side}"]
            info = df[f"info_{side}_{name}"]
            share = info / (info + 1 / df[f"d_{side}"] ** 2)
            sign = 1.0 if side == "h" else -1.0
            f[f"{name}_gap"] = f.get(f"{name}_gap", 0.0) + sign * disagree
            blend = blend + sign * share * disagree
        f[f"{name}_blend"] = blend
    f["explosive_gap"] = f.epa_gap - f.success_gap
    f["explosive_blend"] = f.epa_blend - f.success_blend
    eps = 1e-6
    f["prod"] = np.log((df.team1_win_prob_published + eps) / (1 - df.team1_win_prob_published + eps))
    f["prob"] = df.team1_win_prob_published
    return f


SPECS = {
    "published, recalibrated": ["prod"],
    "record + published units, its blend": ["record", "published_blend"],
    "  + a constant-weight term": ["record", "published_blend", "published_gap"],
    "record + EPA units (refit)": ["record", "epa_blend", "epa_gap"],
    "record + success units (refit)": ["record", "success_blend", "success_gap"],
    "record + success + explosive": ["record", "success_blend", "success_gap",
                                     "explosive_blend", "explosive_gap"],
    "published + success + explosive": ["prod", "success_blend", "success_gap",
                                        "explosive_blend", "explosive_gap"],
}


def report(f: pd.DataFrame) -> None:
    f = f[f.fbs & ~f.tie]
    train, test = f[f.year <= TRAIN_THROUGH], f[f.year > TRAIN_THROUGH]
    base = (test.prob - test.win) ** 2
    print(f"\n{LEAGUE}: {len(train)} games through {TRAIN_THROUGH}, {len(test)} after;"
          f" published brier on those {base.mean():.6f}")
    print(f"{'combination':42s} {'held out':>9s} {'vs published':>13s} {'t':>6s}   coefficients")
    for name, cols in SPECS.items():
        scale = train[cols].std()
        model = LogisticRegression(C=1e6, max_iter=5000).fit(train[cols] / scale, train.win)
        p = model.predict_proba(test[cols] / scale)[:, 1]
        loss = (p - test.win) ** 2
        diff = loss.to_numpy() - base.to_numpy()
        t = diff.mean() / diff.std() * np.sqrt(len(diff))
        coefs = " ".join(f"{c} {b:+.2f}" for c, b in zip(cols, model.coef_[0]))
        print(f"{name:42s} {loss.mean():9.6f} {diff.mean() * 1e4:+12.2f}e-4 {t:+6.2f}   {coefs}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"stack_{LEAGUE}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = frame()
        df.to_parquet(path)
    report(features(df))


if __name__ == "__main__":
    main()
