"""Offense meets defense: two success-rate betas collide on every snap.

Each team carries two betas -- its offense's success rate against a
league-average defense, and its defense's success rate allowed against a
league-average offense. When they meet, a snap succeeds with

    logit p = logit o + logit d - λ ± home

which is the odds-ratio (log5) rule: an average offense against an average
defense succeeds at the league rate e^λ/(1+e^λ), and each side moves the odds
by its own ratio to the league's.

The game's successes are beta-binomial around p rather than binomial: a game
has its own weather, script and matchup, so its ~63 snaps are not 63 coin
flips of the season-long rates. `kappa_game` is that game's beta weight in
plays.

The posterior over (o, d) after a game is not a product of betas, so it is
computed on a grid and each side is projected back to a beta by matching its
marginal mean and variance (assumed density filtering). Credit splits by
uncertainty on its own: a defense the model is sure of absorbs little of a
surprise, and the uncertain side absorbs most of it.

Teams are stored as logit deviations `x` from the league level `λ`, plus a
beta weight `kappa`. After every week the deviations are re-centred to mean
zero and `λ` takes up what they moved together, so a league-wide rise in
success rate (2006: 40%, 2021: 50%) is the league's, not split across every
offense and defense.

Between seasons each side regresses toward the league: the deviation keeps
`carry` of itself, and the variance becomes carry² of the old variance plus
(1 - carry²) of the talent spread -- an AR(1) on true quality whose
stationary spread is the talent beta, weight `kappa` plays.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import digamma, expit, gammaln, logit, logsumexp, polygamma

GRID = 48
SPAN = 5.0


@dataclass(frozen=True)
class Params:
    kappa_off: float = 290.0
    """Offensive talent spread, as a beta weight in plays."""
    kappa_def: float = 590.0
    """Defensive talent spread (success allowed), as a beta weight in plays."""
    carry_off: float = 0.6
    """Share of an offense's deviation that survives the offseason."""
    carry_def: float = 0.6
    kappa_game: float = 200.0
    """One game's rate ~ Beta(p·κ, (1-p)·κ) around the matchup rate p."""
    home: float = 0.03
    """Logit edge to the home offense, and against the home defense."""
    drift_off: float = 0.0
    """How far an offense's true rate can move in a week, as an sd in rate.

    Added to the beta's variance before every week, mean held, so recent
    games count for more than old ones -- the way a quarterback's injury
    gets into the offense's beta in weeks rather than a season. 0 is a
    season-long constant.
    """
    drift_def: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [np.log(self.kappa_off), np.log(self.kappa_def), logit(self.carry_off),
             logit(self.carry_def), np.log(self.kappa_game), self.home,
             np.log(self.drift_off + DRIFT_FLOOR), np.log(self.drift_def + DRIFT_FLOOR)]
        )

    @classmethod
    def from_vector(cls, v: Iterable[float]) -> "Params":
        ko, kd, co, cd, kg, h, do, dd = v
        return cls(np.exp(ko), np.exp(kd), expit(co), expit(cd), np.exp(kg), h,
                   max(0.0, np.exp(do) - DRIFT_FLOOR), max(0.0, np.exp(dd) - DRIFT_FLOOR))


#: Searched as log(drift + floor), so a drift of exactly 0 is in reach.
DRIFT_FLOOR = 1e-3


def beta_grid(a: np.ndarray, b: np.ndarray, size: int = GRID):
    """Nodes and normalised log weights for Beta(a, b), on a logit-scale grid.

    The grid spans ±SPAN sd of logit(x) around its mean, evenly in logit, so
    a tight beta and a loose one are both resolved. Weights are the beta
    density times the logit Jacobian, x(1-x).
    """
    mu = digamma(a) - digamma(b)
    sd = np.sqrt(polygamma(1, a) + polygamma(1, b))
    z = np.linspace(-SPAN, SPAN, size)
    t = mu[:, None] + sd[:, None] * z
    x = expit(t)
    logw = a[:, None] * np.log(x) + b[:, None] * np.log1p(-x)
    logw -= logsumexp(logw, axis=1, keepdims=True)
    return x, t, logw


def betabinom_logpmf(k, n, p, kappa):
    if np.isinf(kappa):
        return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                + k * np.log(p) + (n - k) * np.log1p(-p))
    a, b = p * kappa, (1 - p) * kappa
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + gammaln(k + a) + gammaln(n - k + b) - gammaln(n + kappa)
            + gammaln(kappa) - gammaln(a) - gammaln(b))


def moments_to_beta(mean: np.ndarray, var: np.ndarray):
    kappa = mean * (1 - mean) / var - 1
    return mean * kappa, (1 - mean) * kappa


@dataclass
class Collision:
    """What one batch of offense-vs-defense meetings did."""

    p: np.ndarray
    """Predicted success rate before the game (mean over both betas)."""
    loglik: np.ndarray
    """Log predictive probability of the observed successes."""
    off: tuple[np.ndarray, np.ndarray]
    """Offense betas (a, b) after."""
    deff: tuple[np.ndarray, np.ndarray]
    """Defense betas (a, b) after."""


def collide(
    off: tuple[np.ndarray, np.ndarray],
    deff: tuple[np.ndarray, np.ndarray],
    lam: float,
    edge: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    kappa_game: float,
    size: int = GRID,
) -> Collision:
    """C meetings at once: offense i of `off` against defense i of `deff`."""
    xo, to, wo = beta_grid(*off, size=size)
    xd, td, wd = beta_grid(*deff, size=size)
    eta = to[:, :, None] + td[:, None, :] - lam + edge[:, None, None]
    p = expit(eta)
    prior = wo[:, :, None] + wd[:, None, :]
    like = betabinom_logpmf(k[:, None, None], n[:, None, None], p, kappa_game)
    joint = prior + like
    evidence = logsumexp(joint, axis=(1, 2))
    post = np.exp(joint - evidence[:, None, None])

    po, pd_ = post.sum(axis=2), post.sum(axis=1)
    mo = (po * xo).sum(1)
    md = (pd_ * xd).sum(1)
    vo = (po * xo**2).sum(1) - mo**2
    vd = (pd_ * xd**2).sum(1) - md**2
    return Collision(
        p=(np.exp(prior) * p).sum(axis=(1, 2)),
        loglik=evidence,
        off=moments_to_beta(mo, vo),
        deff=moments_to_beta(md, vd),
    )


class League:
    """Every team's two betas, and the league level they are deviations from."""

    def __init__(self, teams: Iterable[str], params: Params, rate: float):
        self.teams = sorted(teams)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.params = params
        self.lam = float(logit(rate))
        n = len(self.teams)
        self.x_off, self.k_off = np.zeros(n), np.full(n, params.kappa_off)
        self.x_def, self.k_def = np.zeros(n), np.full(n, params.kappa_def)

    def betas(self, side: str, idx: np.ndarray):
        x, k = (self.x_off, self.k_off) if side == "off" else (self.x_def, self.k_def)
        m = expit(self.lam + x[idx])
        return m * k[idx], (1 - m) * k[idx]

    def store(self, side: str, idx: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        x, k = (self.x_off, self.k_off) if side == "off" else (self.x_def, self.k_def)
        x[idx] = logit(a / (a + b)) - self.lam
        k[idx] = a + b

    def recentre(self) -> None:
        s_off, s_def = self.x_off.mean(), self.x_def.mean()
        self.x_off -= s_off
        self.x_def -= s_def
        self.lam += s_off + s_def

    def new_season(self) -> None:
        for side, carry, kappa in (
            ("off", self.params.carry_off, self.params.kappa_off),
            ("def", self.params.carry_def, self.params.kappa_def),
        ):
            x, k = (self.x_off, self.k_off) if side == "off" else (self.x_def, self.k_def)
            m = expit(self.lam + x)
            v_post = m * (1 - m) / (k + 1)
            x *= carry
            m_new = expit(self.lam + x)
            v_talent = m_new * (1 - m_new) / (kappa + 1)
            v = carry**2 * v_post + (1 - carry**2) * v_talent
            k[:] = m_new * (1 - m_new) / v - 1

    def age(self) -> None:
        """A week passes: every beta widens by its side's drift, mean held."""
        for drift, x, k in (
            (self.params.drift_off, self.x_off, self.k_off),
            (self.params.drift_def, self.x_def, self.k_def),
        ):
            if not drift:
                continue
            m = expit(self.lam + x)
            v = m * (1 - m) / (k + 1) + drift**2
            k[:] = m * (1 - m) / v - 1

    def play_week(self, week: pd.DataFrame, size: int = GRID) -> pd.DataFrame:
        """Collide every meeting of one week; returns the week with predictions."""
        oi = week.offense.map(self.index).to_numpy()
        di = week.defense.map(self.index).to_numpy()
        off, deff = self.betas("off", oi), self.betas("def", di)
        result = collide(
            off, deff, self.lam, self.params.home * week.home.to_numpy(),
            week.n.to_numpy(), week.k.to_numpy(), self.params.kappa_game, size,
        )
        out = week.assign(
            p=result.p, loglik=result.loglik, lam=self.lam,
            off_a=off[0], off_b=off[1], def_a=deff[0], def_b=deff[1],
        )
        self.store("off", oi, *result.off)
        self.store("def", di, *result.deff)
        self.recentre()
        return out

    def snapshot(self) -> pd.DataFrame:
        rows = []
        for side, x, k in (("offense", self.x_off, self.k_off), ("defense", self.x_def, self.k_def)):
            m = expit(self.lam + x)
            rows.append(pd.DataFrame({"team": self.teams, "side": side, "mean": m,
                                      "a": m * k, "b": (1 - m) * k}))
        return pd.concat(rows, ignore_index=True)


def replay(games: pd.DataFrame, params: Params, size: int = GRID, snapshots: bool = False):
    """Walk every season in order; each game is predicted before it updates."""
    teams = set(games.offense) | set(games.defense)
    first = games.season.min()
    opening = games[games.season == first]
    league = League(teams, params, rate=opening.k.sum() / opening.n.sum())
    weeks, snaps = [], []
    for season, by_season in games.groupby("season", sort=True):
        if season != first:
            league.new_season()
        if snapshots:
            snaps.append(league.snapshot().assign(season=season, week=0, lam=league.lam))
        for week, rows in by_season.groupby("week", sort=True):
            league.age()
            weeks.append(league.play_week(rows, size).assign(season=season, week=week))
            if snapshots:
                snaps.append(league.snapshot().assign(season=season, week=week, lam=league.lam))
    history = pd.concat(weeks, ignore_index=True)
    return (history, pd.concat(snaps, ignore_index=True), league) if snapshots else history
