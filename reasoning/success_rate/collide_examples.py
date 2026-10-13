"""What the collision does: the fit, what it buys, and pictures of it working."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).parent))
from collide import Params, betabinom_logpmf, collide, replay  # noqa: E402
from fit import FIT, load_games  # noqa: E402
from fit import OUT as DATA
from style import (  # noqa: E402
    AQUA,
    BLUE,
    GRID,
    INK,
    INK2,
    ORANGE,
    RAMP,
    SURFACE,
    pct,
    plt,
)

FIG = Path(__file__).parent / "fig"
params = Params(**json.loads((DATA / "collide_params.json").read_text()))
print(params)
games = load_games()
history, snaps, league = replay(games, params, size=48, snapshots=True)
history.to_parquet(DATA / "collide_history.parquet")
snaps.to_parquet(DATA / "collide_snapshots.parquet")


# ------------------------------------------------------------ what it buys
def league_so_far(games: pd.DataFrame, prior_plays: float = 2000.0) -> pd.DataFrame:
    """Baseline: the league's success rate before the week, nothing about teams."""
    weekly = games.groupby(["season", "week"])[["n", "k"]].sum().reset_index()
    rows, last = [], None
    for season, s in weekly.groupby("season"):
        n0 = prior_plays if last is not None else 0.0
        k0 = n0 * (last if last is not None else 0.0)
        cn, ck = n0, k0
        for _, w in s.iterrows():
            rows.append((season, w.week, ck / cn if cn else np.nan))
            cn, ck = cn + w.n, ck + w.k
        last = s.k.sum() / s.n.sum()
    rate = pd.DataFrame(rows, columns=["season", "week", "league"])
    out = games.merge(rate, on=["season", "week"])
    out["p"] = expit(logit(out.league) + params.home * out.home)
    out["loglik"] = betabinom_logpmf(out.k, out.n, out.p, params.kappa_game)
    return out


def summary(h: pd.DataFrame, seasons) -> dict:
    h = h[h.season.isin(seasons)]
    rate = h.k / h.n
    return {
        "loglik/meeting": h.loglik.mean(),
        "rmse (pts)": 100 * np.sqrt(np.average((rate - h.p) ** 2, weights=h.n)),
        "corr": np.corrcoef(rate, h.p)[0, 1],
    }


BIG = 1e7
variants = {
    "league rate so far": league_so_far(games),
    "offense betas only": replay(games, replace(params, kappa_def=BIG), size=32),
    "defense betas only": replay(games, replace(params, kappa_off=BIG), size=32),
    "no carryover (flat prior each season)": replay(
        games, replace(params, carry_off=0.0, carry_def=0.0), size=32),
    "full collision": history,
}
base = {name: summary(variants["league rate so far"], s) for name, s in
        (("2007-25", FIT), ("2026", [2026]))}
table = []
for name, h in variants.items():
    for label, seasons in (("2007-25", FIT), ("2026", [2026])):
        s = summary(h, seasons)
        table.append({"model": name, "seasons": label,
                      "Δ loglik/meeting": s["loglik/meeting"] - base[label]["loglik/meeting"],
                      "rmse (pts)": s["rmse (pts)"], "corr": s["corr"]})
table = pd.DataFrame(table)
print(table.round(4).to_string(index=False))

# Calibration: does a predicted 50% succeed half the time?
h = history[history.season.isin(FIT)]
bins = pd.qcut(h.p, 10)
cal = h.groupby(bins, observed=True).apply(
    lambda g: pd.Series({"predicted": np.average(g.p, weights=g.n), "observed": g.k.sum() / g.n.sum()}))
print(cal.round(4).to_string())

# Does λ track the league? End-of-season λ against the season's actual rate.
end = snaps.groupby("season").lam.last()
actual = games.groupby("season").apply(lambda g: g.k.sum() / g.n.sum())
print(pd.DataFrame({"model league (end)": expit(end), "actual": actual}).round(3).T.to_string())


# --------------------------------------------- one collision, before and after
SEASON, WEEK, OFF, DEF = 2026, 1, "jaguars", "browns"
row = history[(history.season == SEASON) & (history.week == WEEK) & (history.offense == OFF)
              & (history.defense == DEF)].iloc[0]
off0, def0 = (np.array([row.off_a]), np.array([row.off_b])), (np.array([row.def_a]), np.array([row.def_b]))
edge = np.array([params.home * row.home])
after = collide(off0, def0, row.lam, edge, np.array([row.n]), np.array([row.k]), params.kappa_game, size=96)


def mean(a, b):
    return float(a / (a + b))


shift_off = mean(after.off[0][0], after.off[1][0]) - mean(row.off_a, row.off_b)
shift_def = mean(after.deff[0][0], after.deff[1][0]) - mean(row.def_a, row.def_b)
print(f"example: {SEASON} wk{WEEK} {OFF} O vs {DEF} D: {row.k}/{row.n}, predicted {row.p:.3f}; "
      f"offense moved {shift_off * 100:+.2f} pts (κ {row.off_a + row.off_b:.0f}), "
      f"defense moved {shift_def * 100:+.2f} pts (κ {row.def_a + row.def_b:.0f})")

rng = np.random.default_rng(0)
o = rng.beta(row.off_a, row.off_b, 200_000)
d = rng.beta(row.def_a, row.def_b, 200_000)
p = expit(logit(o) + logit(d) - row.lam + edge[0])
game_rate = rng.binomial(int(row.n), rng.beta(p * params.kappa_game, (1 - p) * params.kappa_game)) / row.n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.25})
x = np.linspace(0.36, 0.62, 500)
for (a, b), (a2, b2), color, name in (
    ((row.off_a, row.off_b), (after.off[0][0], after.off[1][0]), BLUE,
     f"{OFF.title()} offense ({row.off_a + row.off_b:.0f} plays of weight): {shift_off * 100:+.1f} pts"),
    ((row.def_a, row.def_b), (after.deff[0][0], after.deff[1][0]), ORANGE,
     f"{DEF.title()} defense ({row.def_a + row.def_b:.0f}): {shift_def * 100:+.1f} pts allowed"),
):
    ax1.plot(x, stats.beta.pdf(x, a, b), color=color, linewidth=1.4, linestyle=(0, (4, 2)))
    ax1.plot(x, stats.beta.pdf(x, a2, b2), color=color, label=name)
ax1.plot([], [], color=INK2, linewidth=1.4, linestyle=(0, (4, 2)), label="before the game (dashed)")
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)
ax1.axvline(expit(row.lam), ymax=0.72, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
ax1.annotate("league", (expit(row.lam), 0), xytext=(-3, 3), textcoords="offset points", color=INK2,
             fontsize=8, ha="right")
ax1.set_yticks([])
ax1.xaxis.set_major_formatter(pct)
ax1.set_title("Each side's beta, before and after", fontsize=11)
ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

edges = np.arange(0.25, 0.80, 1 / row.n) - 0.5 / row.n
ax2.hist(game_rate, bins=edges, density=True, color="#cdcbc4", edgecolor=SURFACE, linewidth=0.5,
         label="this game's success rate (beta-binomial)")
xs = np.linspace(0.25, 0.75, 400)
ax2.plot(xs, stats.gaussian_kde(p[:20000])(xs), color=AQUA, label="matchup rate p")
ax2.axvline(row.k / row.n, color=INK, linewidth=1.5)
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.25)
ax2.annotate(f"actual {row.k}/{row.n} = {row.k / row.n:.0%}", (row.k / row.n, ax2.get_ylim()[1] * 0.55),
             xytext=(5, 0), textcoords="offset points", color=INK, fontsize=9)
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(pct)
ax2.set_title(f"The collision: predicted {row.p:.1%}", fontsize=11)
ax2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
fig.suptitle(f"{OFF.title()} offense meets {DEF.title()} defense, {SEASON} week {WEEK}: "
             "the looser beta takes more of the surprise", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(FIG / "6_collision.png", dpi=160)
plt.close(fig)


# ------------------------------------------------------ one team, many seasons
def team_path(team: str, since: int) -> None:
    s = snaps[(snaps.team == team) & (snaps.season >= since)].copy()
    order = s[["season", "week"]].drop_duplicates().reset_index(drop=True)
    order["t"] = np.arange(len(order))
    s = s.merge(order, on=["season", "week"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for side, color, name in (("offense", BLUE, "offense"), ("defense", ORANGE, "defense (allowed)")):
        z = s[s.side == side]
        ax.fill_between(z.t, stats.beta.ppf(0.1, z.a, z.b), stats.beta.ppf(0.9, z.a, z.b),
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(z.t, z["mean"], color=color)
        ax.annotate(name, (z.t.iloc[-1], z["mean"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    lam = s.drop_duplicates("t")
    ax.plot(lam.t, expit(lam.lam), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("league", (lam.t.iloc[-1], expit(lam.lam.iloc[-1])), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK2, fontsize=8)
    starts = order[order.week == 0]
    for t, season in zip(starts.t, starts.season):
        ax.axvline(t, color=GRID, linewidth=1.2)
        ax.annotate(str(season), (t, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, order.t.max() + 14)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("success rate vs a league-average opponent")
    ax.set_title(f"{team.title()}, {since}–{order.season.max()}: posterior mean and 80% band, week by week; "
                 "each season opens regressed and wider", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"7_{team}_seasons.png", dpi=160)
    plt.close(fig)


team_path("lions", 2020)


# -------------------------------------------- 2026 now, with last year carried
latest = int(snaps.season.max())
now = snaps[(snaps.season == latest) & (snaps.week == snaps[snaps.season == latest].week.max())]
pre = snaps[(snaps.season == latest) & (snaps.week == 0)]
fig, axes = plt.subplots(1, 2, figsize=(10, 8.5))
for ax, side, color in ((axes[0], "offense", BLUE), (axes[1], "defense", ORANGE)):
    z = now[now.side == side].merge(pre[pre.side == side][["team", "mean"]], on="team", suffixes=("", "_pre"))
    z = z.sort_values("mean", ascending=(side == "defense"))
    y = np.arange(len(z))
    ax.hlines(y, stats.beta.ppf(0.05, z.a, z.b), stats.beta.ppf(0.95, z.a, z.b), color=color,
              linewidth=2, alpha=0.45)
    ax.scatter(z.mean_pre, y, facecolors=SURFACE, edgecolors=color, s=26, linewidths=1.2, zorder=3)
    ax.scatter(z["mean"], y, color=color, s=24, zorder=4)
    ax.set_yticks(y, [t.title() for t in z.team], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(expit(now.lam.iloc[0]), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.xaxis.set_major_formatter(pct)
    ax.grid(axis="y", visible=False)
    ax.set_title("offense (higher is better)" if side == "offense"
                 else "defense, allowed (lower is better)", fontsize=10)
weeks_in = int(now.week.iloc[0])
fig.suptitle(f"{latest} after week {weeks_in}, opponent-adjusted, last season carried in",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.text(0.01, 0.005, "filled: now · hollow: preseason (2025 regressed) · bar: 90% interval · "
         "dashed: league", color=INK2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / f"8_now_{latest}.png", dpi=160)
plt.close(fig)

for side in ("offense", "defense"):
    z = now[now.side == side].sort_values("mean", ascending=(side == "defense"))
    print(side, "top 5:", ", ".join(f"{t} {m:.3f}" for t, m in zip(z.team[:5], z["mean"][:5])))
