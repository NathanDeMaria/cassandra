"""Success rate (EPA > 0) per NFL offense and defense, and the beta that fits it."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PLAYS = Path.home() / ".cassandra" / "success_rate" / "nfl_plays.parquet"


def kind(play_type: str | None, text: str | None) -> str:
    """pass / rush / special / penalty / other, off ESPN's play type."""
    t = (play_type or "").lower()
    x = (text or "").lower()
    if "punt" in t or "field goal" in t:
        return "special"
    # Not a bare "spike": Takeo and Brandon Spikes made a lot of tackles.
    if CLOCK_PLAY.search(x):
        return "other"
    if t == "penalty":
        return "penalty"
    if "pass" in t or "sack" in t or "interception" in t:
        return "pass"
    if "rush" in t:
        return "rush"
    if "fumble" in t or "safety" in t:
        return "pass" if (" pass" in x or "sacked" in x) else "rush"
    return "other"


CLOCK_PLAY = re.compile(r"\bkneels?\b|\btook a knee\b|\bspiked?\s+the\s+ball\b")


def load_plays() -> pd.DataFrame:
    plays = pd.read_parquet(PLAYS)
    plays["kind"] = [kind(t, x) for t, x in zip(plays.play_type, plays.text)]
    plays["success"] = plays.epa > 0
    return plays


def scrimmage(plays: pd.DataFrame) -> pd.DataFrame:
    """The plays success rate is over: dropbacks and designed runs."""
    return plays[plays.kind.isin(["pass", "rush"])]


def team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per offense per game: plays and successes.

    `home` is +1 for the home offense, -1 for the away one, 0 at a neutral site.
    """
    out = (
        scrimmage(plays)
        .groupby(
            ["season", "week", "date", "game_id", "offense", "defense",
             "offense_is_home", "neutral_site"]
        )
        .success.agg(n="size", k="sum")
        .reset_index()
        .sort_values(["date", "game_id"])
    )
    out["home"] = np.where(out.neutral_site, 0, np.where(out.offense_is_home, 1, -1))
    return out.drop(columns=["offense_is_home", "neutral_site"])


def team_seasons(games: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-season n and k for each team on one side of the ball."""
    out = games.groupby(["season", side])[["n", "k"]].sum().reset_index()
    return out.rename(columns={side: "team"})


def fit_concentration(n: np.ndarray, k: np.ndarray, mean: np.ndarray) -> float:
    """Beta-binomial concentration κ = α+β by marginal likelihood, mean held fixed.

    κ is the prior's weight in plays: a team's true rate sits within
    sqrt(μ(1-μ)/(κ+1)) of the league's.
    """

    def nll(log_kappa: float) -> float:
        kappa = np.exp(log_kappa)
        return -stats.betabinom.logpmf(k, n, mean * kappa, (1 - mean) * kappa).sum()

    return float(np.exp(optimize.minimize_scalar(nll, bounds=(1, 12), method="bounded").x))
