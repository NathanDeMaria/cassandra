"""The record, updated on a margin with its least lasting points discounted.

    python discount.py LEAGUE NAME [--knobs lam_expl,lam_rest,obs_sd] [--from RUN]

`residual.py` splits a margin into efficiency points, explosiveness points
and the rest (special teams, the clip, the expected points model's error),
and finds the last two nearly pure noise at the team level -- 17 and 49
games of prior on ncaafb where the margin whole takes 1.6 -- while the
record banks them at full value. This updates the record on

    margin - lam_expl * (explosiveness points - their mean)
           - lam_rest * (rest - its mean)

in the forward step and in what the smoother replays, and nowhere else:
the prediction is scored against the game as played. `lam_*` 0 is the
published model. The split's constants -- EPA on success rate, points per
point of EPA margin, each part's mean for home and neutral games -- are
measured on games through 2021 only.

Nelder-Mead on the named knobs (`obs_sd` too, by default, since a cleaner
margin is a less noisy one), by brier through 2021 -- FBS v FBS on ncaafb,
2006 on for the NFL -- then the whole history saved for the comparisons:
`ncaafb_out/NAME` (so `ncaafb/compare.py` reads it) or `out/NAME`.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from persistence import TRAIN_THROUGH, load  # noqa: E402

DATA = Path.home() / ".cassandra/success_rate/units"


class Discounted(UnitMarginGlickoPredictor):
    """The production model, its record stepping on a discounted margin."""

    lam_expl = 0.0
    lam_rest = 0.0
    split: dict = {}

    _pending: list[float] | None = None

    def _discount(self, game) -> float:
        if not (self.lam_expl or self.lam_rest):
            return 0.0
        e = self._game_epa.get(game.game_id)
        if e is None or not e.home_plays or not e.away_plays:
            return 0.0
        if e.home_success is None or e.away_success is None:
            return 0.0
        s = self.split
        epa = e.home * e.home_plays - e.away * e.away_plays
        eff = ((s["alpha"] + s["beta"] * e.home_success) * e.home_plays
               - (s["alpha"] + s["beta"] * e.away_success) * e.away_plays)
        margin = float(game.home_score - game.away_score)
        key = "neutral" if game.neutral_site else "home"
        expl = s["k"] * (epa - eff) - s[f"expl_{key}"]
        rest = margin - s["k"] * epa - s[f"rest_{key}"]
        return self.lam_expl * expl + self.lam_rest * rest

    def _update_game(self, game):
        cut = self._discount(game)
        self._pending = [cut, -cut] if cut else None
        try:
            prediction = super()._update_game(game)
        finally:
            self._pending = None
        if cut:
            last = self._this_week[-1]
            assert last.home == game.home and last.away == game.away
            self._this_week[-1] = last._replace(actual=last.actual - cut)
        return prediction

    def _step(self, my, opp, margin, home_adjustment):
        # The record's two forward steps, home then away; the smoother
        # replays the stored (already discounted) margin with none pending.
        if self._pending:
            margin -= self._pending.pop(0)
        return super()._step(my, opp, margin, home_adjustment)


def measure_split(league: str) -> dict:
    """The split's constants, from games through 2021."""
    games, plays = load(league)
    plays = plays[plays.year <= TRAIN_THROUGH]
    w = np.sqrt(plays.snaps.to_numpy())
    design = np.c_[np.ones(len(plays)), plays.success]
    alpha, beta = np.linalg.lstsq(design * w[:, None], plays.epa * w, rcond=None)[0]
    plays = plays.assign(eff=alpha + beta * plays.success)
    rows = plays.groupby("game_id")
    first = rows.nth(0).set_index("game_id")
    second = rows.nth(1).set_index("game_id").loc[first.index]
    df = games.set_index("game_id").loc[first.index]
    epa = first.epa * first.snaps - second.epa * second.snaps
    eff = first.eff * first.snaps - second.eff * second.snaps
    design = np.c_[np.ones(len(df)), df.home, epa]
    k = np.linalg.lstsq(design, df.margin, rcond=None)[0][2]
    expl = k * (epa - eff)
    rest = df.margin - k * epa
    home = df.home.to_numpy() > 0
    split = {"alpha": float(alpha), "beta": float(beta), "k": float(k),
             "expl_home": float(expl[home].mean()), "expl_neutral": float(expl[~home].mean()),
             "rest_home": float(rest[home].mean()), "rest_neutral": float(rest[~home].mean())}
    print("split:", {key: round(v, 3) for key, v in split.items()}, flush=True)
    return split


def ncaafb_runner():
    sys.path.insert(0, str(HERE.parent / "ncaafb"))
    from replay import BASE, run_from_2014, save, scores

    def run(knobs: dict, through: int = 2026) -> pd.DataFrame:
        return run_from_2014(knobs, through=through, predictor_class=Discounted)

    def train(df: pd.DataFrame) -> float:
        return scores(df)["train"]

    def store(name: str, df: pd.DataFrame, knobs: dict, extra: dict) -> None:
        save(name, df, knobs, **extra)
        print(name, {k: round(v, 6) for k, v in scores(df).items()}, flush=True)

    return BASE, run, train, store


def nfl_runner():
    sys.path.insert(0, str(HERE.parent / "units"))
    from prod import BASE, INDEX, NAMED, NAMER, OUT, PRIORS

    from cassandra.odds import OddsDatabase
    from cassandra.predictor.opponent_prior import OpponentPriorManager
    from cassandra.save_predictions import join_with_odds

    def run(knobs: dict, through: int = 2026) -> pd.DataFrame:
        predictor = Discounted(
            "nfl", **{**BASE, **knobs},
            opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
            game_epa=INDEX,
        )
        seasons = [s for s in NAMED if s.year <= through]
        return pd.DataFrame([asdict(p) for p in join_with_odds(
            predictor, seasons, OddsDatabase({}), post_callbacks=False, namer=NAMER)])

    def window(df: pd.DataFrame, lo: int, hi: int) -> float:
        loss = (df.team1_win_prob - df.team1_win) ** 2
        return float(loss[(df.year >= lo) & (df.year <= hi)].mean())

    def train(df: pd.DataFrame) -> float:
        return window(df, 2006, TRAIN_THROUGH)

    def store(name: str, df: pd.DataFrame, knobs: dict, extra: dict) -> None:
        df.to_parquet(OUT / f"{name}.parquet")
        result = {"train": train(df), "held_out": window(df, TRAIN_THROUGH + 1, 2026),
                  "since_2014": window(df, 2014, 2026)}
        (OUT / f"{name}.json").write_text(json.dumps({"knobs": knobs, **result, **extra}, indent=2))
        print(name, {k: round(v, 6) for k, v in result.items()}, flush=True)

    return BASE, run, train, store


#: knob: (where it's searched, how far the first simplex steps)
LAMBDAS = {"lam_expl": 0.3, "lam_rest": 0.3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("league")
    parser.add_argument("name")
    parser.add_argument("--knobs", default="lam_expl,lam_rest,obs_sd")
    parser.add_argument("--from", dest="start_from",
                        help="a run whose knobs to start from and hold, as `search.py --from`")
    args = parser.parse_args()
    Discounted.split = measure_split(args.league)
    base, run_knobs, train, store = (ncaafb_runner if args.league == "ncaafb" else nfl_runner)()
    held = {}
    if args.start_from:
        out = DATA / ("ncaafb_out" if args.league == "ncaafb" else "out")
        held = json.loads((out / f"{args.start_from}.json").read_text())["knobs"]
        base = {**base, **held}

    def run(knobs: dict, through: int = 2026) -> pd.DataFrame:
        return run_knobs({**held, **knobs}, through=through)

    names = args.knobs.split(",")
    start = {"lam_expl": 0.0, "lam_rest": 0.0, **{n: base[n] for n in names if n not in LAMBDAS}}

    def decode(z: np.ndarray) -> tuple[dict, dict]:
        lams = {k: 0.0 for k in LAMBDAS}
        knobs = {}
        for name, v in zip(names, z):
            if name in LAMBDAS:
                lams[name] = float(v)
            else:
                knobs[name] = float(np.exp(v))
        return lams, knobs

    def apply(lams: dict) -> None:
        Discounted.lam_expl, Discounted.lam_rest = lams["lam_expl"], lams["lam_rest"]

    x0 = np.array([start[n] if n in LAMBDAS else np.log(start[n]) for n in names])
    simplex = [x0] + [x0 + np.eye(len(names))[i] * (LAMBDAS.get(n, 0.15))
                      for i, n in enumerate(names)]
    seen = []

    def objective(z: np.ndarray) -> float:
        lams, knobs = decode(z)
        apply(lams)
        loss = train(run(knobs, through=TRAIN_THROUGH))
        seen.append(loss)
        if len(seen) % 10 == 0:
            print(args.name, len(seen), f"{loss:.6f} best {min(seen):.6f}", lams,
                  {k: round(v, 3) for k, v in knobs.items()}, flush=True)
        return loss

    fit = optimize.minimize(objective, x0, method="Nelder-Mead",
                            options={"maxfev": 40 * len(names), "initial_simplex": np.array(simplex),
                                     "fatol": 2e-7, "xatol": 1e-3})
    lams, knobs = decode(fit.x)
    apply(lams)
    store(args.name, run(knobs), {**held, **knobs},
          {**lams, "evals": len(seen), "split": Discounted.split, "from": args.start_from})
    print(args.name, "done", lams, knobs, flush=True)


if __name__ == "__main__":
    main()
