"""Examples of a team's success-rate distribution, as a beta posterior."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from success import fit_concentration, load_plays, team_games, team_seasons

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "fig")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "lines.linewidth": 2,
    }
)


def pct(x, _=None):
    return f"{x:.0%}"


plays = load_plays()
games = team_games(plays)
# Exhibitions (Pro Bowl AFC/NFC): a franchise has played 8+ games in some season.
counts = games.groupby(["season", "offense"]).size()
franchises = set(counts[counts >= 8].index.get_level_values("offense"))
games = games[games.offense.isin(franchises) & games.defense.isin(franchises)]
print("franchises:", len(franchises), "dropped:", sorted(set(counts.index.get_level_values("offense")) - franchises))

league = games.groupby("season")[["n", "k"]].sum()
league["rate"] = league.k / league.n
print("league success rate by season\n", league.rate.round(3).to_string())

complete = games[games.season <= 2025]
fits = {}
for side in ("offense", "defense"):
    ts = team_seasons(complete, side)
    mu = league.rate.reindex(ts.season).to_numpy()
    kappa = fit_concentration(ts.n.to_numpy(), ts.k.to_numpy(), mu)
    rate = ts.k / ts.n
    resid = rate - mu
    binom_var = float(np.mean(mu * (1 - mu) / ts.n))
    talent_sd = float(np.sqrt(np.mean(mu * (1 - mu)) / (kappa + 1)))
    fits[side] = kappa
    print(
        f"{side}: kappa={kappa:.0f} plays, talent sd={talent_sd:.4f}, "
        f"observed sd={resid.std():.4f}, binomial-only sd={np.sqrt(binom_var):.4f}, "
        f"plays/season median={ts.n.median():.0f}"
    )

KAPPA = fits["offense"]


def prior(season: int, kappa: float = KAPPA) -> tuple[float, float]:
    mu = league.rate.get(season - 1, league.rate.iloc[-1])
    return mu * kappa, (1 - mu) * kappa


def path(team: str, season: int, side: str = "offense") -> pd.DataFrame:
    """Posterior after each game of `team`'s season, plus the raw rate."""
    g = games[(games.season == season) & (games[side] == team)].reset_index(drop=True)
    a0, b0 = prior(season, fits[side])
    a = a0 + g.k.cumsum()
    b = b0 + (g.n - g.k).cumsum()
    out = pd.DataFrame(
        {
            "game": np.arange(1, len(g) + 1),
            "opp": g["defense" if side == "offense" else "offense"],
            "n": g.n,
            "k": g.k,
            "game_rate": g.k / g.n,
            "raw": g.k.cumsum() / g.n.cumsum(),
            "a": a,
            "b": b,
        }
    )
    out["mean"] = out.a / (out.a + out.b)
    out["lo"] = stats.beta.ppf(0.10, out.a, out.b)
    out["hi"] = stats.beta.ppf(0.90, out.a, out.b)
    head = pd.DataFrame(
        {"game": [0], "a": [a0], "b": [b0], "mean": [a0 / (a0 + b0)],
         "lo": [stats.beta.ppf(0.10, a0, b0)], "hi": [stats.beta.ppf(0.90, a0, b0)]}
    )
    return pd.concat([head, out], ignore_index=True)


# ---------------------------------------------------------------- 1. spread
ts = team_seasons(complete, "offense")
mu = league.rate.reindex(ts.season).to_numpy()
resid = (ts.k / ts.n - mu).to_numpy()
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.arange(-0.12, 0.121, 0.01)
ax.hist(resid, bins=bins, density=True, color="#b7d3f6", edgecolor=SURFACE, linewidth=2,
        label="observed team-seasons, 2006–2025")
x = np.linspace(-0.12, 0.12, 400)
m = float(np.mean(mu))
n_med = float(np.median(ts.n))
talent_sd = np.sqrt(m * (1 - m) / (KAPPA + 1))
noise_sd = np.sqrt(m * (1 - m) / n_med)
ax.plot(x, stats.norm.pdf(x, 0, noise_sd), color=ORANGE, label=f"luck alone ({n_med:.0f} plays, every team average)")
ax.plot(x, stats.beta.pdf(x + m, m * KAPPA, (1 - m) * KAPPA), color=BLUE,
        label=f"talent alone: Beta fit, κ = {KAPPA:.0f} plays")
ax.plot(x, stats.norm.pdf(x, 0, np.hypot(talent_sd, noise_sd)), color=INK, linewidth=1.2,
        linestyle=(0, (4, 2)), label="talent + luck (what the beta-binomial predicts)")
ax.xaxis.set_major_formatter(lambda v, _: f"{v*100:+.0f} pts")
ax.set_xlabel("offensive success rate minus that season's league rate")
ax.set_yticks([])
share = talent_sd**2 / resid.var()
ax.set_title(
    f"A season of offensive success rate is {share:.0%} team (talent + schedule), "
    f"{1 - share:.0%} luck"
)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / "1_spread.png", dpi=160)
plt.close(fig)


# ---------------------------------------------- 2/3. one team through a season
season_2025 = team_seasons(games[games.season == 2025], "offense")
season_2025["rate"] = season_2025.k / season_2025.n
season_2025 = season_2025.sort_values("rate")
best, worst = season_2025.team.iloc[-1], season_2025.team.iloc[0]
print("2025 best/worst offense:", best, worst)

fig, ax = plt.subplots(figsize=(8, 4.4))
for team, color in ((best, BLUE), (worst, ORANGE)):
    p = path(team, 2025)
    ax.fill_between(p.game, p.lo, p.hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(p.game, p["mean"], color=color)
    ax.scatter(p.game[1:], p.game_rate[1:], s=18, color=color, alpha=0.55, linewidths=0)
    ax.annotate(team.title(), (p.game.iloc[-1], p["mean"].iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(league.rate[2025], color=INK2, linewidth=1, linestyle=(0, (3, 3)))
ax.annotate("2025 league", (23, league.rate[2025]), xytext=(0, 3), textcoords="offset points",
            color=INK2, fontsize=8, va="bottom", ha="right")
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("games played (0 = preseason prior: Beta at 2024's league rate, κ plays)")
ax.set_ylabel("offensive success rate")
ax.set_xlim(-0.3, 23)
ax.set_xticks(range(0, 21, 2))
ax.set_title("2025's best and worst offenses: posterior mean, 80% band, and each game")
fig.tight_layout()
fig.savefig(OUT / "2_season_path.png", dpi=160)
plt.close(fig)

# The distribution itself, narrowing.
p = path(best, 2025)
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.linspace(0.38, 0.62, 600)
marks = [0, 4, 10, int(p.game.iloc[-1])]
for color, g in zip(["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], marks):
    row = p[p.game == g].iloc[0]
    y = stats.beta.pdf(x, row.a, row.b)
    plays_seen = row.a + row.b - (p.a.iloc[0] + p.b.iloc[0])
    label = "prior" if g == 0 else f"after {g} games ({plays_seen:.0f} plays)"
    ax.plot(x, y, color=color, label=label)
    ax.annotate(label.split(" (")[0], (x[np.argmax(y)], y.max()), xytext=(0, 4),
                textcoords="offset points", ha="center", color=INK2, fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
ax.xaxis.set_major_formatter(pct)
ax.set_yticks([])
ax.set_xlabel("offensive success rate")
ax.set_title(f"{best.title()} offense, 2025: Beta(α + successes, β + failures) after n games")
fig.tight_layout()
fig.savefig(OUT / "3_densities.png", dpi=160)
plt.close(fig)


# -------------------------------------------------- 4. the league at a moment
def forest(season: int, fname: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 8.5), sharey=False)
    for ax, side, color in ((axes[0], "offense", BLUE), (axes[1], "defense", ORANGE)):
        ts = team_seasons(games[games.season == season], side)
        a0, b0 = prior(season, fits[side])
        ts["a"], ts["b"] = a0 + ts.k, b0 + ts.n - ts.k
        ts["mean"] = ts.a / (ts.a + ts.b)
        ts["lo"] = stats.beta.ppf(0.05, ts.a, ts.b)
        ts["hi"] = stats.beta.ppf(0.95, ts.a, ts.b)
        ts = ts.sort_values("mean", ascending=(side == "defense"))
        y = np.arange(len(ts))
        ax.hlines(y, ts.lo, ts.hi, color=color, linewidth=2, alpha=0.45)
        ax.scatter(ts["mean"], y, color=color, s=22, zorder=3)
        ax.scatter(ts.k / ts.n, y, color=INK2, marker="|", s=40, zorder=4)
        ax.set_yticks(y, [t.title() for t in ts.team], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(a0 / (a0 + b0), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax.xaxis.set_major_formatter(pct)
        ax.grid(axis="y", visible=False)
        ax.set_title(
            "offense (higher is better)" if side == "offense" else "defense, allowed (lower is better)",
            fontsize=10,
        )
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "dot: posterior mean · bar: 90% interval · tick: raw rate · dashed: prior mean",
             color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


forest(2025, "4_forest_2025.png", "End of 2025: every team's success-rate posterior")
latest = int(games.season.max())
n_games = games[games.season == latest].groupby("offense").size().median()
forest(latest, f"5_forest_{latest}.png",
       f"{latest} so far (~{n_games:.0f} games each): same prior, far less data")

# A few numbers worth quoting.
g = games[games.season <= 2025]
rate = g.k / g.n
print(f"per game: median plays {g.n.median():.0f}, sd of game rate {rate.std():.3f}, "
      f"binomial sd at median n {np.sqrt(m*(1-m)/g.n.median()):.3f}")
print(path(best, 2025)[["game", "opp", "n", "k", "game_rate", "raw", "mean", "lo", "hi"]].round(3).to_string())
