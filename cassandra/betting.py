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

Everything here is a pure function of a predictions frame and an
`OddsDatabase`; `betting.py` at the repo root does the I/O.
"""

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
