"""Whether success rate's own posterior says anything the units don't.

    python posterior.py ncaafb|nfl

Every units model keeps, beside its sides, a second pair per team on the
success rate alone -- binomial noise per down, its own clocks, an average
opponent -- for the ratings page (`unit_glicko`'s "What a release
carries"). It is the estimate-with-a-deviation of success rate that the
blend this README's "Combining the signals" weighed didn't have: the units
read success rate and EPA mixed before anything learns from them, so their
deviation is one number for the mixture.

This replays the published default -- `glicko_margin_units_success` -- with
a subclass that writes down, before each game, the published probability and
both teams' displayed rates and deviations, then asks two things.

- The question `stack.py` asked of its ingredients: fit through 2021 beside
  the published probability, scored on 2022 on, does the rates' gap add
  anything, and does how much to trust it move with its deviation? The
  rates are posterior means, shrunk already, so the deviation can only say
  the gap is worth more or less than its mean claims.
- Whether the deviation a reader is shown is honest: each offense's rate in
  the game against what the two teams' displayed rates expected of it,
  over the deviation they claimed plus the game's own binomial noise.
"""

import asyncio
import pickle
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import publish
from cassandra.constants import CASSANDRA_HOME
from cassandra.objective import _in_fbs
from cassandra.odds import OddsDatabase
from cassandra.predictor.epa import load_epa
from cassandra.predictor.unit_glicko import (
    SUCCESS_SCALE,
    UNWEIGHTED,
    UnitMarginGlickoPredictor,
)
from cassandra.save_predictions import join_with_odds

LEAGUE = sys.argv[1]
MODEL = "glicko_margin_units_success"
OUT = CASSANDRA_HOME / "success_rate" / "combine"
TRAIN_THROUGH = 2021


class Recorder(UnitMarginGlickoPredictor):
    """The production model, writing down both teams' displayed rates before a game."""

    rows: list[dict] = []

    def _update_game(self, game):
        per_rating = (
            self._anchor_epa_scale
            * self.points_per_rating
            / (2 * self._points_per_epa)
            / SUCCESS_SCALE[self._league][UNWEIGHTED][0]
        )
        row: dict = {"game_id": game.game_id, "center": self._success_rates.center}
        for side, team in (("h", game.home), ("a", game.away)):
            rates = self._success_rates.rates(team, self.anchor, per_rating)
            if rates is None:
                break
            row[f"off_{side}"], row[f"off_sd_{side}"] = rates.offense
            row[f"def_{side}"], row[f"def_sd_{side}"] = rates.defense
        else:
            self.rows.append(row)
        return super()._update_game(game)


def _seasons() -> list:
    path = OUT / f"{LEAGUE}_seasons.pkl"
    if not path.exists():
        from endgame_aws import Config

        from cassandra.save_predictions import read_rated_seasons

        bucket = Config.init_from_file().bucket
        path.write_bytes(pickle.dumps(asyncio.run(read_rated_seasons(LEAGUE, bucket))))
    return pickle.loads(path.read_bytes())


def frame() -> pd.DataFrame:
    job = publish._job(CASSANDRA_HOME / "models" / LEAGUE / f"{MODEL}_result.json", LEAGUE)
    Recorder.rows = []
    predictor = Recorder(LEAGUE, **job.config.params)
    df = pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, _seasons(), OddsDatabase({}), post_callbacks=False,
        roll_over_final_season=False)])
    df["fbs"] = [
        LEAGUE != "ncaafb" or (_in_fbs(h, int(y)) and _in_fbs(a, int(y)))
        for h, a, y in zip(df.home_team, df.away_team, df.year)
    ]
    return df.merge(pd.DataFrame(Recorder.rows), on="game_id")


def features(df: pd.DataFrame) -> pd.DataFrame:
    """The rates' gaps, home less away, in success-rate points.

    `net` is each team's rate won less its rate allowed, against an average
    opponent; the gap is the two teams' nets apart, and `net_var` the sum of
    the four deviations squared.
    """
    f = pd.DataFrame({"year": df.year, "week": df.week_number, "fbs": df.fbs,
                      "win": df.team1_win.astype(float),
                      "tie": df.home_score == df.away_score})
    f["prob"] = df.team1_win_prob
    eps = 1e-6
    f["prod"] = np.log((f.prob + eps) / (1 - f.prob + eps))
    f["off_gap"] = df.off_h - df.off_a
    f["def_gap"] = df.def_h - df.def_a
    f["net"] = f.off_gap - f.def_gap
    f["net_var"] = df.off_sd_h**2 + df.def_sd_h**2 + df.off_sd_a**2 + df.def_sd_a**2
    f["net_x_sd"] = f.net * np.sqrt(f.net_var)
    return f


SPECS = {
    "published, recalibrated": ["prod"],
    "rates' gap alone": ["net"],
    "published + rates' gap": ["prod", "net"],
    "published + gap, gap x its sd": ["prod", "net", "net_x_sd"],
    "published + offense, defense apart": ["prod", "off_gap", "def_gap"],
}


def report(f: pd.DataFrame) -> None:
    f = f[f.fbs & ~f.tie]
    train, test = f[f.year <= TRAIN_THROUGH], f[f.year > TRAIN_THROUGH]
    early = (test.week <= 4).to_numpy()
    base = ((test.prob - test.win) ** 2).to_numpy()
    print(f"\n{LEAGUE}: {len(train)} games through {TRAIN_THROUGH}, {len(test)} after"
          f" ({early.sum()} in weeks 1-4); published brier on those {base.mean():.6f}")
    print(f"{'combination':36s} {'held out':>9s} {'vs published':>13s} {'t':>6s}"
          f" {'weeks 1-4':>10s} {'t':>6s}   coefficients")
    for name, cols in SPECS.items():
        scale = train[cols].std()
        model = LogisticRegression(C=1e6, max_iter=5000).fit(train[cols] / scale, train.win)
        p = model.predict_proba(test[cols] / scale)[:, 1]
        diff = (p - test.win.to_numpy()) ** 2 - base

        def t(d: np.ndarray) -> float:
            return d.mean() / d.std() * np.sqrt(len(d))

        coefs = " ".join(f"{c} {b:+.2f}" for c, b in zip(cols, model.coef_[0]))
        print(f"{name:36s} {base.mean() + diff.mean():9.6f} {diff.mean() * 1e4:+12.2f}e-4"
              f" {t(diff):+6.2f} {diff[early].mean() * 1e4:+8.2f}e-4 {t(diff[early]):+6.2f}"
              f"   {coefs}")
    for season in sorted(test.year.unique()):
        cols = SPECS["published + rates' gap"]
        scale = train[cols].std()
        model = LogisticRegression(C=1e6, max_iter=5000).fit(train[cols] / scale, train.win)
        mask = (test.year == season).to_numpy()
        p = model.predict_proba(test[cols][mask] / scale)[:, 1]
        d = (p - test.win.to_numpy()[mask]) ** 2 - base[mask]
        print(f"  {season}: rates' gap {d.mean() * 1e4:+.2f}e-4 over {mask.sum()} games")


def honesty(df: pd.DataFrame) -> None:
    """Each offense's rate in a game, in deviations from what was expected.

    Expected: the offense's displayed rate and the defense's rate allowed,
    each against an average opponent, less the center they both count from
    once. `excess` is the residual's spread beyond the downs' own noise --
    what the expected rate actually missed by, which the deviations
    claimed to be `claimed`. Neither field nor the gap between tiers in that
    particular game is in the expectation, so `excess` is an upper bound on
    what they should have claimed. A z spread near 1 is a deviation that
    means what it says; well under 1, the ratings page shows more doubt
    than there is.
    """
    games = load_epa(LEAGUE)
    rows = []
    for row in df.itertuples():
        game = games.get(row.game_id)
        if game is None or row.center is None:
            continue
        for offense, defense, rate, downs in (
            ("h", "a", game.home_success, game.home_success_plays),
            ("a", "h", game.away_success, game.away_success_plays),
        ):
            if rate is None or not downs:
                continue
            expected = getattr(row, f"off_{offense}") + getattr(row, f"def_{defense}") - row.center
            claimed = getattr(row, f"off_sd_{offense}") ** 2 + getattr(row, f"def_sd_{defense}") ** 2
            noise = expected * (1 - expected) / downs
            rows.append({"year": row.year, "week": row.week_number, "fbs": row.fbs,
                         "resid": rate - expected, "claimed": claimed, "noise": noise})
    z = pd.DataFrame(rows)
    z = z[z.fbs & (z.year > TRAIN_THROUGH)]
    print(f"\nhonesty, {len(z)} offenses since {TRAIN_THROUGH + 1}:")
    print(f"{'weeks':>8s} {'n':>6s} {'mean resid':>11s} {'sd resid':>9s} {'claimed':>8s}"
          f" {'noise':>7s} {'excess':>7s} {'z sd':>6s} {'within 1':>9s} {'within 2':>9s}")
    for label, mask in (("all", z.week > 0), ("1-4", z.week <= 4), ("5+", z.week > 4)):
        part = z[mask]
        scaled = part.resid / np.sqrt(part.claimed + part.noise)
        excess = np.sqrt(max(part.resid.var() - part.noise.mean(), 0.0))
        print(f"{label:>8s} {len(part):6d} {part.resid.mean():+11.4f} {part.resid.std():9.4f}"
              f" {np.sqrt(part.claimed.mean()):8.4f} {np.sqrt(part.noise.mean()):7.4f}"
              f" {excess:7.4f}"
              f" {scaled.std():6.3f} {(scaled.abs() < 1).mean():9.3f} {(scaled.abs() < 2).mean():9.3f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"posterior_{LEAGUE}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = frame()
        df.to_parquet(path)
    report(features(df))
    honesty(df)


if __name__ == "__main__":
    main()
