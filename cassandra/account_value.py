"""What the Kalshi account is worth once its open bets settle, by the model.

Every open position is a contract that pays $1 if one outcome happens.
Football has no ties, so a YES on a team and a NO on its opponent pay on the
same result, and the payout is the contract count either way. What was paid
is already out of the cash, so the account ends at

    cash + sum(count_i * [bet i wins])

and, with the games independent, each bet is a Bernoulli draw at the
model's probability. The mean is a sum; the spread comes from drawing the
whole slate many times (`simulate`).
"""

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
from endgame.types import Season

#: How long after kickoff a game is taken to be over, as `picks.py` waits.
GAME_LENGTH = timedelta(hours=4)

#: Percentiles of the final balance that `summary` reports.
PERCENTILES = (5, 25, 50, 75, 95)


class Bet(NamedTuple):
    """One open position, with the probability it pays."""

    label: str
    #: Dollars paid for it, fees included.
    cost: float
    #: Dollars it pays if it wins: the contract count.
    payout: float
    #: The probability it wins; 0 or 1 once the game is decided.
    probability: float
    #: The same under the market's no-vig price, where one is known.
    market: float | None = None

    @property
    def expected(self) -> float:
        return self.payout * self.probability

    @property
    def at_market(self) -> float:
        """The expected payout at the market's price; the model's where there's none."""
        return self.payout * (self.probability if self.market is None else self.market)


def held_probability(yes: bool, team_wins: float) -> float:
    """The chance a position pays, given the chance its ticker's team wins.

    A YES pays when the team wins; a NO pays when it doesn't.
    """
    return team_wins if yes else 1 - team_wins


def simulate(
    bets: list[Bet], cash: float, draws: int = 200_000, seed: int = 0
) -> np.ndarray:
    """The account's balance after every bet settles, `draws` times over."""
    rng = np.random.default_rng(seed)
    payouts = np.array([bet.payout for bet in bets])
    wins = rng.random((draws, len(bets))) < np.array([b.probability for b in bets])
    return cash + wins @ payouts


def summary(bets: list[Bet], cash: float, balances: np.ndarray) -> dict:
    """The expected balance and its spread, against what's staked.

    `staked` is the account before the bets were paid for, so `profit` is
    what the pending slate adds to it.
    """
    staked = sum(bet.cost for bet in bets)
    expected = cash + sum(bet.expected for bet in bets)
    variance = sum(
        bet.payout**2 * bet.probability * (1 - bet.probability) for bet in bets
    )
    return {
        "cash": cash,
        "staked": staked,
        "expected": expected,
        "profit": expected - cash - staked,
        "roi": (expected - cash) / staked - 1 if staked else 0.0,
        "sd": variance**0.5,
        "best": cash + sum(bet.payout for bet in bets),
        "percentiles": dict(
            zip(PERCENTILES, np.percentile(balances, PERCENTILES).tolist())
        ),
        "p_profit": float((balances > cash + staked).mean()),
        "p_loss_over_10pct": float((balances < cash + 0.9 * staked).mean()),
        "market_expected": cash + sum(bet.at_market for bet in bets),
    }


def histogram(balances: np.ndarray, bins: int = 14, width: int = 40) -> list[str]:
    """A text histogram of the balances, one line per dollar bin."""
    counts, edges = np.histogram(balances, bins=bins)
    top = counts.max()
    return [
        f"  ${low:>7,.0f} - ${high:>7,.0f}  {count / len(balances):6.1%}  "
        + "#" * round(width * count / top)
        for count, low, high in zip(counts, edges[:-1], edges[1:])
    ]


def plot(balances: np.ndarray, s: dict, path: str) -> None:
    """A density histogram of the final balances, with the break-even and the mean."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, series = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160, facecolor=surface)
    ax.set_facecolor(surface)
    ax.hist(
        balances,
        bins=60,
        density=True,
        color=series,
        edgecolor=surface,
        linewidth=0.8,
    )
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    breakeven = s["cash"] + s["staked"]
    # Labels sit above the bars, break-even to the left of its line, the mean to the right.
    for x, label, style, align in (
        (breakeven, f"break-even \\${breakeven:,.0f}", "--", "right"),
        (s["expected"], f"expected \\${s['expected']:,.0f}", "-", "left"),
    ):
        ax.axvline(x, color=ink, linewidth=1.2, linestyle=style)
        ax.text(
            x + (2 if align == "left" else -2),
            top * 0.97,
            label,
            color=ink,
            fontsize=9,
            va="top",
            ha=align,
        )
    ax.set_xlabel("account balance after every bet settles (\\$)", color=muted)
    ax.set_ylabel("probability density (per \\$)", color=muted)
    ax.set_title(
        f"Expected balance \\${s['expected']:,.0f} ± \\${s['sd']:,.0f}; "
        f"{s['p_profit']:.0%} chance the bets net a gain",
        color=ink,
        loc="left",
        fontsize=11,
    )
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def as_of(season: Season, cutoff: datetime) -> Season:
    """The season as it stood at `cutoff`: a game not over by then has no result.

    Its scores and status are cleared, so a replay treats it as unplayed
    and can predict it.
    """

    def unplayed(game):
        if game.completed and game.date + GAME_LENGTH > cutoff:
            return game._replace(
                completed=False, home_score=0, away_score=0, status="STATUS_SCHEDULED"
            )
        return game

    return season._replace(
        weeks=[
            week._replace(games=[unplayed(g) for g in week.games])
            for week in season.weeks
        ]
    )


def expectation(bets: list[Bet], actual_payout: float, draws: int = 100_000) -> dict:
    """What a slate of settled bets was expected to net, and where the result fell.

    `percentile` is the share of draws that paid less than the actual
    payout, counting a tie as half.
    """
    staked = sum(bet.cost for bet in bets)
    payouts = simulate(bets, 0.0, draws)
    below = float((payouts < actual_payout - 0.005).mean())
    tied = float((abs(payouts - actual_payout) <= 0.005).mean())
    variance = sum(b.payout**2 * b.probability * (1 - b.probability) for b in bets)
    return {
        "staked": staked,
        "expected": sum(bet.expected for bet in bets) - staked,
        "sd": variance**0.5,
        "p_gain": float((payouts > staked).mean()),
        "actual": actual_payout - staked,
        "percentile": below + tied / 2,
        "pnl": payouts - staked,
    }


def plot_weeks(panels: list[dict], path: str) -> None:
    """One density histogram of net P&L per week, with the expected and actual marked.

    Each panel: `label`, `pnl` (the draws), `expected`, `actual`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grey, accent = (
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#c3c2b7",
        "#2a78d6",
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 3.2 * len(panels)),
        dpi=160,
        facecolor=surface,
        squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        ax.set_facecolor(surface)
        ax.hist(
            panel["pnl"],
            bins=50,
            density=True,
            color=grey,
            edgecolor=surface,
            linewidth=0.8,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        top = ax.get_ylim()[1]
        # The lower line's label goes left of it and the higher one's right,
        # so two lines close together don't write over each other.
        marks = (
            (
                panel["expected"],
                f"expected {panel['expected']:+,.0f}",
                ink,
                "--",
                1.2,
                "normal",
            ),
            (
                panel["actual"],
                f"actual {panel['actual']:+,.0f}",
                accent,
                "-",
                2.2,
                "bold",
            ),
        )
        lower = min(marks, key=lambda m: m[0])
        for x, label, color, style, width, weight in marks:
            ax.axvline(x, color=color, linewidth=width, linestyle=style)
            left = x == lower[0]
            ax.text(
                x + (-1 if left else 1),
                top * 0.96,
                label,
                color=color,
                fontsize=9,
                fontweight=weight,
                ha="right" if left else "left",
                va="top",
            )
        ax.set_title(panel["label"], color=ink, loc="left", fontsize=11)
        ax.set_xlabel("net \\$ from the week's bets", color=muted)
        ax.set_ylabel("density (per \\$)", color=muted)
        ax.tick_params(colors=muted)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c3")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
