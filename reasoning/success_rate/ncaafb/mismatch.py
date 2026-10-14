"""Does success rate hold up across ncaafb's spread of opponents?

Two checks on the games with play-by-play (2014 on, every game with an FBS
side), each against EPA per play on the same games:

1. Same-game: within bins of the pregame favorite's win probability (the
   published roster model's replay), how closely each reading's
   differential follows the margin, and at what slope. A reading that
   saturates in mismatches flattens there.
2. Additivity: what the units assume, a reading = league + offense +
   defense. Fit per season leaving each week out, predict that week, and
   bin the errors by the predicted reading. Success rate is fit on its own
   scale and on logits -- the odds-ratio rule -- and both are scored as
   success rates, so the two are compared in the same units.

    python mismatch.py

Reads `~/.cassandra/success_rate/units/ncaafb_index.pkl` (`sweep_index.py
ncaafb`) and the roster model's cached replay.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit, logit

from cassandra.objective import _in_fbs

DATA = Path.home() / ".cassandra/success_rate/units"
REPLAY = Path.home() / ".cassandra/replays/ncaafb/glicko_margin_units_roster.parquet"
#: Ridge on each team's offset; the FCS-and-below offsets get their own
#: group mean, so a one-game FCS team shrinks toward FCS and not the league.
RIDGE = 2.0
FAVORITE_BINS = [0.5, 0.65, 0.8, 0.9, 0.97, 1.0]


def games() -> pd.DataFrame:
    index = pickle.loads((DATA / "ncaafb_index.pkl").read_bytes())
    replay = pd.read_parquet(REPLAY)
    replay = replay[replay.game_id.isin(list(index))].copy()
    for side in ("home", "away"):
        replay[f"epa_{side}"] = [getattr(index[g], side) for g in replay.game_id]
        replay[f"sr_{side}"] = [getattr(index[g], f"{side}_success") for g in replay.game_id]
        replay[f"n_{side}"] = [getattr(index[g], f"{side}_success_plays") for g in replay.game_id]
        replay[f"fbs_{side}"] = [
            _in_fbs(t, int(y)) for t, y in zip(replay[f"{side}_team"], replay.year)
        ]
    replay["margin"] = replay.home_score - replay.away_score
    replay["favorite"] = np.maximum(replay.team1_win_prob, 1 - replay.team1_win_prob)
    return replay.dropna(subset=["sr_home", "sr_away"]).reset_index(drop=True)


def same_game(df: pd.DataFrame) -> None:
    clipped = df.assign(
        lsr_home=logit(df.sr_home.clip(0.02, 0.98)), lsr_away=logit(df.sr_away.clip(0.02, 0.98))
    )
    print("\n1. same game: margin on each reading's differential, by the favorite's win probability")
    print(f"{'favorite':>12} {'games':>6} {'FBSvFBS':>8}  {'epa r':>6} {'slope':>6}  "
          f"{'sr r':>6} {'slope':>6}  {'logit r':>7} {'slope':>6}  {'fav sr':>6} {'dog sr':>6}")
    cuts = pd.cut(clipped.favorite, FAVORITE_BINS, include_lowest=True)
    for label, g in [*clipped.groupby(cuts, observed=True), ("all", clipped)]:
        row = [f"{str(label):>12} {len(g):6d} {(g.fbs_home & g.fbs_away).mean():8.0%} "]
        for col in ("epa", "sr", "lsr"):
            x = g[f"{col}_home"] - g[f"{col}_away"]
            row.append(f"{np.corrcoef(x, g.margin)[0, 1]:6.3f} {np.polyfit(x, g.margin, 1)[0]:6.1f}")
        home_fav = g.team1_win_prob >= 0.5
        fav = np.where(home_fav, g.sr_home, g.sr_away)
        dog = np.where(home_fav, g.sr_away, g.sr_home)
        row.append(f"{fav.mean():6.3f} {dog.mean():6.3f}")
        print("  ".join(row))


def _sides(df: pd.DataFrame) -> pd.DataFrame:
    """One row per offense per game."""
    home = pd.DataFrame({
        "game_id": df.game_id, "year": df.year, "week": df.week_number,
        "offense": df.home_team, "defense": df.away_team,
        "home": np.where(df.neutral_site, 0.0, 1.0),
        "off_fbs": df.fbs_home, "def_fbs": df.fbs_away,
        "epa": df.epa_home, "sr": df.sr_home, "n": df.n_home, "favorite": df.favorite,
        "fav": df.team1_win_prob >= 0.5,
    })
    away = pd.DataFrame({
        "game_id": df.game_id, "year": df.year, "week": df.week_number,
        "offense": df.away_team, "defense": df.home_team,
        "home": np.where(df.neutral_site, 0.0, -1.0),
        "off_fbs": df.fbs_away, "def_fbs": df.fbs_home,
        "epa": df.epa_away, "sr": df.sr_away, "n": df.n_away, "favorite": df.favorite,
        "fav": df.team1_win_prob < 0.5,
    })
    return pd.concat([home, away], ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, y: str) -> np.ndarray:
    """reading = mu + home + FCS offense + FCS defense + team offense + team defense."""
    teams = sorted(set(train.offense) | set(train.defense) | set(test.offense) | set(test.defense))
    at = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)

    def design(d: pd.DataFrame) -> sparse.csr_matrix:
        rows = np.arange(len(d))
        fixed = np.column_stack([
            np.ones(len(d)), d.home.to_numpy(),
            (~d.off_fbs.to_numpy()).astype(float), (~d.def_fbs.to_numpy()).astype(float),
        ])
        off = sparse.csr_matrix((np.ones(len(d)), (rows, d.offense.map(at))), shape=(len(d), n_t))
        dfn = sparse.csr_matrix((np.ones(len(d)), (rows, d.defense.map(at))), shape=(len(d), n_t))
        return sparse.hstack([sparse.csr_matrix(fixed), off, dfn]).tocsr()

    x = design(train)
    penalty = sparse.hstack([
        sparse.csr_matrix((2 * n_t, 4)), np.sqrt(RIDGE) * sparse.identity(2 * n_t),
    ]).tocsr()
    a = sparse.vstack([x, penalty]).tocsr()
    b = np.concatenate([train[y].to_numpy(), np.zeros(2 * n_t)])
    coef = lsqr(a, b, atol=1e-10, btol=1e-10)[0]
    return design(test) @ coef


def additivity(df: pd.DataFrame) -> None:
    sides = _sides(df)
    sides["lsr"] = logit(sides.sr.clip(0.02, 0.98))
    preds = []
    for _, season in sides.groupby("year"):
        for week in season.week.unique():
            test = season[season.week == week]
            train = season[season.week != week]
            out = test[["game_id", "offense", "sr", "epa", "favorite", "fav", "off_fbs", "def_fbs"]].copy()
            out["sr_id"] = _fit_predict(train, test, "sr")
            out["sr_logit"] = expit(_fit_predict(train, test, "lsr"))
            out["epa_hat"] = _fit_predict(train, test, "epa")
            preds.append(out)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(DATA / "ncaafb_additivity.parquet")

    print("\n2. additivity, leave-one-week-out: observed minus predicted, by the predicted reading")
    bins = pd.qcut(p.sr_id, 10)
    print(f"{'predicted sr (identity)':>24} {'sides':>6}  {'identity':>8} {'logit':>8}   "
          f"{'epa bias':>8} {'epa bias/sd':>11}")
    epa_sd = p.epa.std()
    for label, g in p.groupby(bins, observed=True):
        print(f"{str(label):>24} {len(g):6d}  {(g.sr - g.sr_id).mean():+8.4f} {(g.sr - g.sr_logit).mean():+8.4f}"
              f"   {(g.epa - g.epa_hat).mean():+8.4f} {(g.epa - g.epa_hat).mean() / epa_sd:+11.3f}")
    sr_sd = p.sr.std()
    for name, mask in (("all", p.sr == p.sr), ("FBS v FBS", p.off_fbs & p.def_fbs),
                       ("favorite 97%+", p.favorite > 0.97), ("FBS v FCS-", p.off_fbs != p.def_fbs)):
        g = p[mask]
        rmse = {k: np.sqrt(((g.sr - g[k]) ** 2).mean()) for k in ("sr_id", "sr_logit")}
        slope = {k: np.polyfit(g[k], g.sr, 1)[0] for k in ("sr_id", "sr_logit")}
        e_slope = np.polyfit(g.epa_hat, g.epa, 1)[0]
        print(f"  {name:>14}: {len(g):6d} sides; sr rmse identity {rmse['sr_id']:.4f} logit {rmse['sr_logit']:.4f}"
              f" (sd {sr_sd:.3f}); observed-on-predicted slope identity {slope['sr_id']:.3f}"
              f" logit {slope['sr_logit']:.3f}, epa {e_slope:.3f}")


if __name__ == "__main__":
    df = games()
    print(f"{len(df)} games with a success rate, {df.year.min()}-{df.year.max()}; "
          f"{(df.fbs_home & df.fbs_away).mean():.0%} FBS v FBS; "
          f"success rate per side mean {pd.concat([df.sr_home, df.sr_away]).mean():.3f}, "
          f"sd {pd.concat([df.sr_home, df.sr_away]).std():.3f}, "
          f"5-95% {pd.concat([df.sr_home, df.sr_away]).quantile([0.05, 0.95]).round(3).tolist()}")
    same_game(df)
    additivity(df)
