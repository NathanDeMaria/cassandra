"""Whether the model's picks beat the market, judged the way a bettor would.

`model_eval.score_predictions` grades a pick against whatever line the odds
database has on the game -- the close, for a played game -- and reports how
often it covered. That's the right number for choosing between models and
the wrong one for deciding whether to bet, for two reasons this module
exists to separate:

**Closing line value.** A bet is placed at the line that's up when the
model can act, and the model can act once both teams' previous games are
in. If the market then moves toward the model's side by kickoff, the bet
had value whether or not it cashed; if it moves away, the market knew
something the model didn't. Over a season CLV settles long before a
win-loss record does, because it doesn't have a coin flip on the end of it.
`spread_bets` grades each pick at the *entry* line (`line_windows`: the
first read after both teams' last game) against the *close* (the last read
before kickoff).

**Moneyline edge.** A win probability is a price, and the book has one too.
`moneyline_bets` compares the model's probability with the no-vig market
probability and runs the simplest strategies on the gap: bet the side the
model likes by more than a threshold, flat stake; the same split by whether
that side was the favorite; fractional Kelly on the model's own number. The
calibration comparison (`moneyline_calibration`: Brier, model vs market) is
printed next to the ROI because it explains it -- a model less sure than the
market "finds" edge on every underdog and loses on every one of them.

**Whether the model knows anything the market doesn't.** ROI and cover
rate answer "did betting it make money", which on a season of lines is
mostly luck. `market_information` asks the question they are noisy
answers to: regress the result's miss from the line on the model's
disagreement with it. The slope is the weight the model deserves next to
the line -- 0 if the line already had everything, 1 if the model is right
and the line wrong -- and its standard error says how sure. The same slope
on the line's *move* from entry to close is CLV per point of edge, which
settles faster still. `probability_information` is the moneyline's
version. Those are the numbers to compare two models on, and the ones to
watch week to week: a model worth betting has a weight whose error bar
clears zero before its bankroll does anything interesting.

`spread_strategies` then grades flat -110 bets by minimum edge with the
p-value against break-even beside the ROI, `by_week` says whether an edge
is drifting, and `team_disagreement` lists the teams the model and the
line disagree about most, with who the results sided with.

Everything here is a pure function of a predictions frame and an
`OddsDatabase`; `betting.py` at the repo root does the I/O.
"""

import math
from collections.abc import Sequence
from datetime import timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from .columns import GameDfColumns
from .odds import OddsDatabase

# How long after kickoff a game is over, for "after the previous game".
# Generous rather than exact: a line read during the previous game's fourth
# quarter is one the model's ratings hadn't seen the result of.
GAME_LENGTH = timedelta(hours=4)

# Model-minus-market probability gaps to run the flat-stake strategy at.
EDGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)

# The Kelly fraction to stake. Full Kelly on a model that's less calibrated
# than the market is how a bankroll goes to zero; a quarter is the usual
# hedge against the estimate being wrong, and the estimate is the question.
KELLY_FRACTION = 0.25

# Edge at the entry line, in points, to bucket the CLV report by. A pick the
# model barely prefers is a coin flip; the buckets say whether the bigger
# disagreements are the ones the market comes around to.
EDGE_POINT_BUCKETS = ((0, 2), (2, 4), (4, 7), (7, float("inf")))


def american_to_probability(price: pd.Series) -> pd.Series:
    """The probability a moneyline implies, vig included.

    -150 means risk 150 to win 100, so 150/250; +200 means risk 100 to win
    200, so 100/300. NaN stays NaN.
    """
    p = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(p < 0, -p / (100 - p), 100 / (100 + p))
    return pd.Series(implied, index=price.index)


def american_payout(price: pd.Series) -> pd.Series:
    """Profit on a unit stake if the bet wins. -150 pays 2/3; +200 pays 2."""
    p = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        payout = np.where(p < 0, 100 / -p, p / 100)
    return pd.Series(payout, index=price.index)


def no_vig_home_probability(home: pd.Series, away: pd.Series) -> pd.Series:
    """The market's home win probability with the book's margin taken out.

    The two implied probabilities sum to more than one by the hold;
    dividing each by that sum is the simplest way to remove it, and the
    one every comparison here uses. Fancier de-vigging (power, Shin)
    moves the answer by a fraction of a point and would be a second
    convention to keep straight.
    """
    p_home = american_to_probability(home)
    p_away = american_to_probability(away)
    return p_home / (p_home + p_away)


def previous_game_end(predictions: pd.DataFrame) -> pd.Series:
    """When each game's teams were both done with their previous one.

    Indexed by game_id. The later of the two teams' previous kickoffs, plus
    `GAME_LENGTH`; NaT when either team has no earlier game in the frame.
    "Previous" is by kickoff across every season in the frame, so a week 1
    game's window opens after last season's finale -- which is what "after
    the previous game" means for it, since the line was up all offseason.
    """
    sides = pd.concat(
        [
            predictions[["game_id", "date", "home_team"]].rename(
                columns={"home_team": "team"}
            ),
            predictions[["game_id", "date", "away_team"]].rename(
                columns={"away_team": "team"}
            ),
        ]
    ).sort_values("date", kind="stable")
    sides["previous"] = sides.groupby("team")["date"].shift(1)
    # `max` over a NaT and a timestamp is the timestamp, which would let a
    # team's first game through on the other team's history alone.
    ends = sides.groupby("game_id")["previous"].agg(
        lambda s: s.max() if s.notna().all() else pd.NaT
    )
    return ends + GAME_LENGTH


class LineColumns:
    """The columns `line_windows` adds: the same prices at two reads."""

    ENTRY_READ_AT = "entry_read_at"
    ENTRY_SPREAD = "entry_spread"
    ENTRY_HOME_ML = "entry_home_moneyline"
    ENTRY_AWAY_ML = "entry_away_moneyline"
    CLOSE_READ_AT = "close_read_at"
    CLOSE_SPREAD = "close_spread"
    CLOSE_HOME_ML = "close_home_moneyline"
    CLOSE_AWAY_ML = "close_away_moneyline"

    @classmethod
    def all(cls) -> list[str]:
        return [
            cls.ENTRY_READ_AT,
            cls.ENTRY_SPREAD,
            cls.ENTRY_HOME_ML,
            cls.ENTRY_AWAY_ML,
            cls.CLOSE_READ_AT,
            cls.CLOSE_SPREAD,
            cls.CLOSE_HOME_ML,
            cls.CLOSE_AWAY_ML,
        ]


def line_windows(predictions: pd.DataFrame, odds_db: OddsDatabase) -> pd.DataFrame:
    """The entry and closing prices on every game the frame has a forecast for.

    Entry is the first read after `previous_game_end` and before kickoff;
    close is the last read before kickoff. The entry columns are null on a
    game with no read in that window, and a game with no read before
    kickoff at all is dropped: there was nothing to bet into.

    The close is "last read before kickoff", which is only as close as the
    pulls got: an hour at best, since `today` runs hourly, and much longer
    for any game the hourly pulls missed -- every 2026 Saturday before
    09-19 came back cut to ESPN's default 25 events, so the later kickoffs
    "closed" at the morning `near` read. The read time is kept so a report
    can say how stale its closes are.
    """
    ends = previous_game_end(predictions)
    rows = []
    for game_id, date in zip(predictions["game_id"], predictions["date"]):
        game_id = str(game_id)
        kickoff = pd.Timestamp(date)
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        before = [s for s in odds_db.snapshots(game_id) if s.read_at < kickoff]
        if not before:
            continue
        window_open = ends.get(game_id, pd.NaT)
        after_previous = [
            s for s in before if pd.notna(window_open) and s.read_at > window_open
        ]
        entry = after_previous[0] if after_previous else None
        close = before[-1]
        rows.append(
            {
                "game_id": game_id,
                LineColumns.ENTRY_READ_AT: entry.read_at if entry else pd.NaT,
                LineColumns.ENTRY_SPREAD: entry.spread if entry else None,
                LineColumns.ENTRY_HOME_ML: entry.home_moneyline if entry else None,
                LineColumns.ENTRY_AWAY_ML: entry.away_moneyline if entry else None,
                LineColumns.CLOSE_READ_AT: close.read_at,
                LineColumns.CLOSE_SPREAD: close.spread,
                LineColumns.CLOSE_HOME_ML: close.home_moneyline,
                LineColumns.CLOSE_AWAY_ML: close.away_moneyline,
            }
        )
    lines = pd.DataFrame(rows, columns=["game_id"] + LineColumns.all())
    return predictions.merge(lines, on="game_id", how="inner")


def spread_bets(lines: pd.DataFrame) -> pd.DataFrame:
    """Grade the model's side at the entry line, and what the close did to it.

    Needs `predicted_margin` (the calibrated margin, home positive) on top
    of `line_windows`' columns; only games with both an entry and a closing
    spread are kept. Columns added:

    - `bet_home`: the model's side at the entry line -- home if its margin
      beats the line's, `predicted_margin > -entry_spread`.
    - `edge_points`: how far the model sits from the entry line, in points.
    - `clv`: points the close moved toward the bet. Positive means the line
      got worse for anyone betting the same side later. A line quoted from
      the home side gets *smaller* as home gets more favored, so for a home
      bet that's entry minus close, and the reverse for away.
    - `covered_entry`, `push_entry`, `covered_close`, `push_close`: the
      result against each line. A push is neither, unlike `score_predictions`
      where it counts against team1; a bettor gets the stake back.
    """
    bets = lines[
        lines[LineColumns.ENTRY_SPREAD].notna()
        & lines[LineColumns.CLOSE_SPREAD].notna()
    ].copy()
    entry = bets[LineColumns.ENTRY_SPREAD]
    close = bets[LineColumns.CLOSE_SPREAD]
    mov = bets["home_score"] - bets["away_score"]
    bets["bet_home"] = bets["predicted_margin"] > -entry
    bets["edge_points"] = (bets["predicted_margin"] + entry).abs()
    bets["clv"] = np.where(bets["bet_home"], entry - close, close - entry)
    for name, line in (("entry", entry), ("close", close)):
        result = line + mov
        bets[f"covered_{name}"] = np.where(bets["bet_home"], result > 0, result < 0)
        bets[f"push_{name}"] = result == 0
    return bets


class Record(NamedTuple):
    wins: int
    losses: int
    pushes: int

    def __str__(self) -> str:
        return f"{self.wins}-{self.losses}-{self.pushes}"

    @property
    def cover_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else float("nan")


def record(bets: pd.DataFrame, line: str) -> Record:
    """Wins-losses-pushes of `spread_bets`' picks against the `line` spread."""
    covered, push = bets[f"covered_{line}"], bets[f"push_{line}"]
    return Record(
        int((covered & ~push).sum()), int((~covered & ~push).sum()), int(push.sum())
    )


def clv_by_edge(bets: pd.DataFrame) -> pd.DataFrame:
    """CLV and the entry-line record, bucketed by how far the model was from the line."""
    rows = []
    for low, high in EDGE_POINT_BUCKETS:
        bucket = bets[(bets["edge_points"] >= low) & (bets["edge_points"] < high)]
        if bucket.empty:
            continue
        rec = record(bucket, "entry")
        rows.append(
            {
                "edge_points": f"{low:g}-{high:g}"
                if high < float("inf")
                else f"{low:g}+",
                "n": len(bucket),
                "clv_mean": bucket["clv"].mean(),
                "clv_positive": (bucket["clv"] > 0).mean(),
                "clv_negative": (bucket["clv"] < 0).mean(),
                "record_entry": str(rec),
                "cover_rate_entry": rec.cover_rate,
            }
        )
    return pd.DataFrame(rows)


#: What a winning spread bet pays per unit at the standard -110, and the
#: cover rate that breaks even against it: 110/210.
SPREAD_PAYOUT = 100 / 110
BREAK_EVEN = 110 / 210

#: Minimum edge, in points against the entry line, for `spread_strategies`.
SPREAD_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0)


class MeanWithError(NamedTuple):
    mean: float
    se: float
    n: int

    @property
    def t(self) -> float:
        return self.mean / self.se if self.se > 0 else float("nan")

    def __str__(self) -> str:
        return f"{self.mean:+.3f} ± {self.se:.3f} (t {self.t:+.1f}, n {self.n})"


def mean_with_error(values: pd.Series) -> MeanWithError:
    values = values.dropna()
    n = len(values)
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return MeanWithError(float(values.mean()) if n else float("nan"), se, n)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(intercept, slope, slope se, residual sd) of y on x, heteroskedasticity-robust (HC1)."""
    design = np.column_stack([np.ones_like(x), x])
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    n = len(y)
    meat = (design * residual[:, None] ** 2).T @ design
    covariance = bread @ meat @ bread * n / max(n - 2, 1)
    return (
        float(beta[0]),
        float(beta[1]),
        float(np.sqrt(max(covariance[1, 1], 0.0))),
        float(residual.std(ddof=2)) if n > 2 else float("nan"),
    )


class MarketInformation(NamedTuple):
    """Whether the model knows anything the line doesn't, at one read.

    `weight` is the slope of (result - line) on (model - line): the share of
    each point of disagreement that turns out to be right. 0 means the line
    already had everything the model had -- betting it is paying the vig to
    flip coins. 1 means the model is right and the line is wrong. The best
    blend of the two is `line + weight * (model - line)`, and `mae_blend`
    is its in-sample MAE (a single slope on a few hundred games, so the
    optimism is small). A weight is the thing to compare between two models
    on the same games: it is what each would be worth to someone who
    already has the line.

    `move` is the same slope for how the line moved from the entry read to
    the close -- how much of the model's disagreement at entry the market
    came around to by kickoff. It is CLV per point of edge, and it settles
    much faster than `weight`, since the move has no game on the end of it.
    """

    at: str
    n: int
    weight: float
    weight_se: float
    mae_line: float
    mae_model: float
    mae_blend: float
    move: float
    move_se: float


def market_information(lines: pd.DataFrame, at: str) -> MarketInformation:
    """`MarketInformation` for the `at` read ("entry" or "close")."""
    spread = f"{at}_spread"
    games = lines[lines[spread].notna() & lines["predicted_margin"].notna()]
    line = -games[spread].to_numpy(dtype=float)
    model = games["predicted_margin"].to_numpy(dtype=float)
    mov = (games["home_score"] - games["away_score"]).to_numpy(dtype=float)
    nan = float("nan")
    if len(games) < 3:
        return MarketInformation(at, len(games), nan, nan, nan, nan, nan, nan, nan)
    _, weight, weight_se, _ = _ols(model - line, mov - line)
    move, move_se = nan, nan
    moved = (
        games[LineColumns.ENTRY_SPREAD].notna()
        & games[LineColumns.CLOSE_SPREAD].notna()
    )
    if at == "entry" and moved.sum() > 2:
        entry = -games.loc[moved, LineColumns.ENTRY_SPREAD].to_numpy(dtype=float)
        close = -games.loc[moved, LineColumns.CLOSE_SPREAD].to_numpy(dtype=float)
        _, move, move_se, _ = _ols(model[moved.to_numpy()] - entry, close - entry)
    blend = line + weight * (model - line)
    return MarketInformation(
        at=at,
        n=len(games),
        weight=weight,
        weight_se=weight_se,
        mae_line=float(np.abs(mov - line).mean()),
        mae_model=float(np.abs(mov - model).mean()),
        mae_blend=float(np.abs(mov - blend).mean()),
        move=move,
        move_se=move_se,
    )


def binomial_tail(wins: int, n: int, p: float) -> float:
    """P(at least `wins` of `n`) for a coin that comes up with probability `p`.

    Exact, in log space so a season's worth of bets doesn't overflow; NaN
    for no bets. The one-sided p-value of a record against break-even.
    """
    if n <= 0:
        return float("nan")
    k = np.arange(wins, n + 1)
    log_terms = (
        math.lgamma(n + 1)
        - np.vectorize(math.lgamma)(k + 1)
        - np.vectorize(math.lgamma)(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return float(np.exp(log_terms).sum())


def spread_strategies(
    bets: pd.DataFrame, thresholds: Sequence[float] = SPREAD_THRESHOLDS
) -> pd.DataFrame:
    """Flat unit bets at -110 on the model's side of the entry line, by minimum edge.

    One row per threshold and per split -- every bet, then the model's side
    as the favorite or the underdog at entry, then home or away. `p_value`
    is one-sided against `BREAK_EVEN`: how often a coin that covers exactly
    as often as the vig requires would have done this well. Read it before
    `roi`; on a few hundred bets, a 55% cover rate is about one standard
    error from break-even.
    """
    favored = np.where(
        bets["bet_home"],
        bets[LineColumns.ENTRY_SPREAD] < 0,
        bets[LineColumns.ENTRY_SPREAD] > 0,
    )
    splits = {
        "all": np.ones(len(bets), dtype=bool),
        "favorite": favored,
        "underdog": ~favored & (bets[LineColumns.ENTRY_SPREAD] != 0).to_numpy(),
        "home": bets["bet_home"].to_numpy(dtype=bool),
        "away": ~bets["bet_home"].to_numpy(dtype=bool),
    }
    rows = []
    for threshold in thresholds:
        over = (bets["edge_points"] >= threshold).to_numpy()
        for split, mask in splits.items():
            chosen = bets[over & mask]
            if chosen.empty:
                continue
            rec = record(chosen, "entry")
            decided = rec.wins + rec.losses
            units = rec.wins * SPREAD_PAYOUT - rec.losses
            rate = rec.cover_rate
            rows.append(
                {
                    "min_edge": threshold,
                    "side": split,
                    "n": len(chosen),
                    "record": str(rec),
                    "cover_rate": rate,
                    "cover_se": np.sqrt(rate * (1 - rate) / decided)
                    if decided
                    else float("nan"),
                    "units": units,
                    "roi": units / len(chosen),
                    "p_value": binomial_tail(rec.wins, decided, BREAK_EVEN),
                    "clv": chosen["clv"].mean(),
                }
            )
    return pd.DataFrame(rows)


def by_week(bets: pd.DataFrame) -> pd.DataFrame:
    """CLV and the entry-line record per week, to see whether an edge is drifting."""
    rows = []
    for (year, week), chosen in bets.groupby(["year", "week_number"]):
        rec = record(chosen, "entry")
        rows.append(
            {
                "year": year,
                "week": week,
                "n": len(chosen),
                "clv": chosen["clv"].mean(),
                "record": str(rec),
                "cover_rate": rec.cover_rate,
                "units": rec.wins * SPREAD_PAYOUT - rec.losses,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["cumulative_units"] = table["units"].cumsum()
    return table


def team_disagreement(lines: pd.DataFrame, at: str = "close") -> pd.DataFrame:
    """Per team, how far the model sat from the `at` line and who the results sided with.

    Signed to the team: `model_vs_line` is how many more points the model
    gave the team than the line did, on average; `result_vs_line` is how
    many more it won by than the line said. A team where the first is large
    and the second is near zero is one the model keeps getting wrong in a
    way the market doesn't -- the list to read before betting on it, and
    the list to take to `team_seasons.py` and `evidence.py` for a cause.
    `model_closer` is the share of games the model's margin finished nearer
    the result than the line's.
    """
    spread = f"{at}_spread"
    games = lines[lines[spread].notna() & lines["predicted_margin"].notna()]
    line = -games[spread]
    mov = games["home_score"] - games["away_score"]
    sides = []
    for team, sign in (("home_team", 1.0), ("away_team", -1.0)):
        sides.append(
            pd.DataFrame(
                {
                    "team": games[team],
                    "model_vs_line": sign * (games["predicted_margin"] - line),
                    "result_vs_line": sign * (mov - line),
                    "model_closer": (games["predicted_margin"] - mov).abs()
                    < (line - mov).abs(),
                }
            )
        )
    grouped = pd.concat(sides).groupby("team")
    table = grouped.agg(
        n=("model_vs_line", "size"),
        model_vs_line=("model_vs_line", "mean"),
        result_vs_line=("result_vs_line", "mean"),
        model_closer=("model_closer", "mean"),
    ).reset_index()
    order = table["model_vs_line"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def _priced(lines: pd.DataFrame, at: str) -> pd.DataFrame:
    """The games with a moneyline on both sides at the `at` read."""
    return lines[
        lines[f"{at}_home_moneyline"].notna() & lines[f"{at}_away_moneyline"].notna()
    ]


def moneyline_calibration(lines: pd.DataFrame, at: str) -> dict[str, float]:
    """Brier for the model and for the no-vig market, on the games both priced.

    The number that says whose probabilities to trust. Compared on the same
    games, since the book prices the games it prices. `hold` is the book's
    margin on those games, which is what any edge has to clear.
    """
    priced = _priced(lines, at)
    home_won = (priced["home_score"] > priced["away_score"]).astype(float)
    home_ml, away_ml = priced[f"{at}_home_moneyline"], priced[f"{at}_away_moneyline"]
    market = no_vig_home_probability(home_ml, away_ml)
    model = priced[GameDfColumns.TEAM1_WIN_PROB]
    hold = american_to_probability(home_ml) + american_to_probability(away_ml) - 1
    return {
        "n": len(priced),
        "brier_model": float(((model - home_won) ** 2).mean()),
        "brier_market": float(((market - home_won) ** 2).mean()),
        "hold": float(hold.mean()),
    }


class ProbabilityInformation(NamedTuple):
    """`MarketInformation` for the moneyline: the model's weight against the no-vig price.

    A logistic regression of the result on the market's log-odds (as an
    offset, so it keeps weight 1) plus `weight` times the model's log-odds
    minus the market's, with an intercept for any home lean the de-vigging
    missed. `weight` 0 means the market's probability already has
    everything; a positive weight with `weight / weight_se` past 2 means
    the model's disagreement carries information the price doesn't. The
    brier columns are on the same games, the blend in-sample.
    """

    at: str
    n: int
    weight: float
    weight_se: float
    brier_market: float
    brier_model: float
    brier_blend: float


def probability_information(lines: pd.DataFrame, at: str) -> ProbabilityInformation:
    priced = _priced(lines, at)
    nan = float("nan")
    if len(priced) < 10:
        return ProbabilityInformation(at, len(priced), nan, nan, nan, nan, nan)
    won = (priced["home_score"] > priced["away_score"]).to_numpy(dtype=float)
    market = no_vig_home_probability(
        priced[f"{at}_home_moneyline"], priced[f"{at}_away_moneyline"]
    ).to_numpy(dtype=float)
    model = priced[GameDfColumns.TEAM1_WIN_PROB].to_numpy(dtype=float)

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    offset = logit(market)
    design = np.column_stack([np.ones_like(offset), logit(model) - offset])
    beta = np.zeros(2)
    information = np.eye(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(offset + design @ beta)))
        information = (design * (p * (1 - p))[:, None]).T @ design
        step = np.linalg.solve(information + 1e-9 * np.eye(2), design.T @ (won - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    blend = 1 / (1 + np.exp(-(offset + design @ beta)))
    covariance = np.linalg.pinv(information)
    return ProbabilityInformation(
        at=at,
        n=len(priced),
        weight=float(beta[1]),
        weight_se=float(np.sqrt(max(covariance[1, 1], 0.0))),
        brier_market=float(((market - won) ** 2).mean()),
        brier_model=float(((model - won) ** 2).mean()),
        brier_blend=float(((blend - won) ** 2).mean()),
    )


def _side_bets(priced: pd.DataFrame, at: str, bet_home: pd.Series) -> pd.DataFrame:
    """Per-game outcome of a unit on `bet_home`'s side at the `at` prices."""
    home_won = priced["home_score"] > priced["away_score"]
    home_ml = priced[f"{at}_home_moneyline"]
    away_ml = priced[f"{at}_away_moneyline"]
    won = np.where(bet_home, home_won, ~home_won)
    payout = np.where(bet_home, american_payout(home_ml), american_payout(away_ml))
    return pd.DataFrame(
        {
            "won": won,
            "profit": np.where(won, payout, -1.0),
            "favorite": np.where(bet_home, home_ml < 0, away_ml < 0),
            "payout": payout,
        },
        index=priced.index,
    )


def _strategy_row(name: str, bets: pd.DataFrame, stake: pd.Series) -> dict:
    """ROI on stake, overall and split by favorite/underdog, for one strategy."""
    placed = stake > 0
    profit = stake * bets["profit"]

    def roi(mask: pd.Series) -> float:
        staked = stake[mask].sum()
        return float(profit[mask].sum() / staked) if staked else float("nan")

    return {
        "strategy": name,
        "n": int(placed.sum()),
        "hit_rate": float(bets["won"][placed].mean()) if placed.any() else float("nan"),
        "units": float(profit.sum()),
        "roi": roi(placed),
        "n_favorites": int((bets["favorite"] & placed).sum()),
        "roi_favorites": roi(bets["favorite"] & placed),
        "roi_underdogs": roi(~bets["favorite"] & placed),
    }


def moneyline_bets(lines: pd.DataFrame, at: str = "close") -> pd.DataFrame:
    """Flat-stake ROI of betting the model's edge, at each threshold.

    One row per threshold in `EDGE_THRESHOLDS`: a unit on whichever side the
    model's probability beats the no-vig market's by more than that. Split
    by whether the bet was on the favorite, because that's the split that
    tells an underconfident model from an edge. Then quarter Kelly
    (`KELLY_FRACTION`) at any positive edge, sized on the model's
    probability against the book's actual price; and the market's own
    baselines -- every favorite, every underdog -- for scale.
    """
    priced = _priced(lines, at)
    market = no_vig_home_probability(
        priced[f"{at}_home_moneyline"], priced[f"{at}_away_moneyline"]
    )
    model = priced[GameDfColumns.TEAM1_WIN_PROB]
    edge_home = model - market
    bets = _side_bets(priced, at, edge_home > 0)
    unit = pd.Series(1.0, index=priced.index)

    rows = [
        _strategy_row(
            f"flat, edge > {threshold:.2f}",
            bets,
            unit.where(edge_home.abs() > threshold, 0.0),
        )
        for threshold in EDGE_THRESHOLDS
    ]
    # Kelly: f = (p*b - q) / b on the model's p and the book's b, floored at
    # zero (no bet against the model's own side), scaled by the fraction.
    p = model.where(edge_home > 0, 1 - model)
    b = bets["payout"]
    kelly = ((p * b - (1 - p)) / b).clip(lower=0) * KELLY_FRACTION
    rows.append(_strategy_row(f"kelly x{KELLY_FRACTION:g}, any edge", bets, kelly))
    home_favored = priced[f"{at}_home_moneyline"] < 0
    rows.append(
        _strategy_row("every favorite", _side_bets(priced, at, home_favored), unit)
    )
    rows.append(
        _strategy_row("every underdog", _side_bets(priced, at, ~home_favored), unit)
    )
    return pd.DataFrame(rows)
