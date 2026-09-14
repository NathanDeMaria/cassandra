"""How a search's knobs become a predictor's constructor arguments.

`GlickoPredictor` takes every matchup term -- home advantage, rest, travel,
a missing quarterback -- and its prediction scale in rating units, and its
two deviation increases as separate rating amounts. Searched that way the
box is badly conditioned, and a finite-difference Hessian of the brier
objective at both football leagues' 2026-09-13 fits says how badly: the
update learns at a fixed 400 (`glicko._Q`) while the prediction reads at
`prediction_scale`, so with sigmoid scoring the ratings settle at
`400 * margin / (sigmoid_scale * ln 10)` and the prediction only ever sees
the product `sigmoid_scale * prediction_scale` -- the two are +0.78
collinear on ncaafb, and every rating-unit term rides the same ridge. The
two deviation increases feed one deviation, so what is determined is their
budget over a season, not the split (+0.79 collinear on nfl).

The `points` frame searches the quantities that are determined instead.
One point of margin is worth `rating_units_per_point(sigmoid_scale)` rating
units, so the matchup terms are searched in points; the prediction scale is
searched as the margin that reads as a 73% favourite; the deviation
increases as a per-season budget and the share of it paid at the rollover.
In that frame the two leagues' quarterback penalties, 74 and 15 in rating
units, are both 3.6 points, and nfl's home advantage is 2.9 -- the numbers
the market quotes.

The predictor never sees a knob: `to_params` turns knobs into constructor
arguments, and a result records the arguments, so `load_predictor`, the
children that pin `glicko_full`'s fit, and `sync_pins.py` all keep reading
rating units.
"""

import math
from collections.abc import Mapping

from ..scoring import DEFAULT_SIGMOID_SCALE

RATING = "rating"
POINTS = "points"
FRAMES = (RATING, POINTS)

_LN10 = math.log(10)
#: The rating gap the *update* reads as 10-to-1, `glicko._Q`'s 400. Named
#: here rather than imported so the frame is a pure function of numbers.
_UPDATE_SCALE = 400.0

#: A matchup term in points and the constructor argument it sets.
_POINT_TERMS = {
    "hfa_pts": "home_advantage",
    "travel_pts": "travel_advantage",
    "rest_pts": "rest_advantage",
    "qb_pts": "qb_out_penalty",
}
#: Every constructor argument the points frame derives, and the knobs that
#: move it. `searched_params` reads this: an argument is searched whenever
#: any knob it depends on is.
_POINTS_DEPENDS = {
    **{param: (knob, "sigmoid_scale") for knob, param in _POINT_TERMS.items()},
    "prediction_scale": ("pred_margin_scale", "sigmoid_scale"),
    "weekly_rd_increase": ("rd_total", "rd_offseason_share"),
    "season_rd_increase": ("rd_total", "rd_offseason_share"),
}
_POINTS_KNOBS = frozenset(
    knob for deps in _POINTS_DEPENDS.values() for knob in deps
) - {"sigmoid_scale"}


def rating_units_per_point(sigmoid_scale: float) -> float:
    """What one point of margin is worth on the rating scale.

    With sigmoid scoring a game counts as `sigmoid(margin / sigmoid_scale)`,
    and the update moves ratings until the expected score at the 400 scale
    matches it, so a team that wins by `m` on average sits `400 m / (ss ln 10)`
    above the field.
    """
    return _UPDATE_SCALE / (sigmoid_scale * _LN10)


def knobs_of(frame: str) -> frozenset[str]:
    """The names `frame` gives meaning to; anything else passes through."""
    _check(frame)
    return _POINTS_KNOBS if frame == POINTS else frozenset()


def derived_params(frame: str) -> frozenset[str]:
    """The constructor arguments `frame` computes rather than passes through."""
    _check(frame)
    return frozenset(_POINTS_DEPENDS) if frame == POINTS else frozenset()


def searched_params(frame: str, searched: set[str] | frozenset[str]) -> frozenset[str]:
    """The constructor arguments that move when the knobs in `searched` do.

    What a pin copied from this model's fit has to be checked against. In the
    rating frame a searched knob is the argument. In the points frame a
    searched `sigmoid_scale` moves every matchup term with it, whether or not
    their point values were searched.
    """
    _check(frame)
    if frame == RATING:
        return frozenset(searched)
    moved = {
        param
        for param, deps in _POINTS_DEPENDS.items()
        if any(knob in searched for knob in deps)
    }
    return frozenset(searched - _POINTS_KNOBS) | moved


def to_params(
    frame: str, knobs: Mapping[str, float | str], weeks_per_season: float
) -> dict[str, float | str]:
    """The constructor arguments for one set of knobs.

    `weeks_per_season` is how many `pass_week` calls a season makes, which is
    what turns a per-season deviation budget back into a weekly increase.

    Lenient about what is missing, in the way the constructor is: a point
    term with no `sigmoid_scale` to price it is priced at the scale the
    predictor would default to, and a share with no budget to split is
    dropped. That is what lets `optimize.py`'s priors pass and `sync_pins.py`
    hand over a config's pins alone and get the arguments those pins fix. A
    *search* in this frame has to name `sigmoid_scale`, which
    `OptimizationConfig` checks.
    """
    _check(frame)
    if frame == RATING:
        return dict(knobs)

    out = dict(knobs)
    sigmoid_scale = float(out.get("sigmoid_scale", DEFAULT_SIGMOID_SCALE))
    per_point = rating_units_per_point(sigmoid_scale)

    for knob, param in _POINT_TERMS.items():
        if knob in out:
            _exclusive(out, knob, param)
            out[param] = float(out.pop(knob)) * per_point
    if "pred_margin_scale" in out:
        _exclusive(out, "pred_margin_scale", "prediction_scale")
        # p = sigmoid(margin / pred_margin_scale) once the ratings have
        # settled: prediction_scale = 400 * pms / sigmoid_scale.
        out["prediction_scale"] = (
            _UPDATE_SCALE * float(out.pop("pred_margin_scale")) / sigmoid_scale
        )
    if "rd_total" in out:
        if "rd_offseason_share" not in out:
            raise ValueError("rd_total needs rd_offseason_share to split it")
        for param in ("weekly_rd_increase", "season_rd_increase"):
            _exclusive(out, "rd_total", param)
        total = float(out.pop("rd_total")) ** 2
        share = float(out.pop("rd_offseason_share"))
        if not 0 <= share <= 1:
            raise ValueError(f"rd_offseason_share is a share: {share}")
        out["season_rd_increase"] = math.sqrt(share * total)
        out["weekly_rd_increase"] = math.sqrt((1 - share) * total / weeks_per_season)
    else:
        out.pop("rd_offseason_share", None)
    return out


def weeks_per_season(week_counts: list[int]) -> float:
    """The `weeks_per_season` a league replays with: the median season's."""
    if not week_counts:
        raise ValueError("no seasons to count weeks in")
    ordered = sorted(week_counts)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _exclusive(knobs: Mapping[str, object], knob: str, param: str) -> None:
    if param in knobs:
        raise ValueError(
            f"{knob} derives {param}; a config gives one or the other, not both"
        )


def _check(frame: str) -> None:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; one of {', '.join(FRAMES)}")
