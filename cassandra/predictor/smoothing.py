"""The Glicko smoother's sweep, over flat arrays instead of dicts.

`GlickoPredictor._smooth` re-walks the season from preseason every week --
see `passes` -- so its cost grows with the square of the weeks played, and
it is where a probe's time goes: 10.4s of a 20.8s ncaafb `glicko_full`
probe, 1.8 million `glicko_step` calls, a thousand times a search.

Nothing about that sweep needs Python. It is a few additions and a `pow`
per team per game, in an order that cannot be vectorized (each game reads
the rating the game before it wrote), which is exactly the shape a compiler
is good at and numpy is not. `sweep` is that loop with the teams addressed
by index rather than by name; `cassandra.predictor.glicko` keeps the names.

numba is imported lazily and the predictor keeps its dict implementation as
a fallback, for two reasons. The serving half of the package -- what a
webapp installs, see `pyproject.toml` -- must not import it, and the dict
version is the reference: `glicko_test.test_the_compiled_sweep_matches_the
_dict_sweep` holds the two together, and when they disagree it is this file
that is wrong.

They agree to the bit on anything small and to about the last bit on a
long replay. `10 ** x` is a libm call either way and the compiled one is
entitled to round it differently, which two million sequential updates
then carry forward. Measured, rather than assumed: replaying ncaafb's
`glicko_full` fit over all 25 seasons -- 78,000 games, `passes=2` -- the
median team's rating is identical, the furthest apart of the 798 teams
differ by 2.3e-13 rating points, and the brier scores land one ulp
apart (-0.15379666971382383 against -0.1537966697138238, 2.8e-17).

So a model fitted through one and replayed through the other is the same
model, and no number anybody reads is different; the replay is still not
bit-exact, which is why this is a deliberate dependency rather than an
import-if-available convenience. A comparison that has to be exact --
"did this refit change the model?" -- has to hold numba's presence
fixed, the way it already holds a package rev fixed.
"""

import math
from typing import Protocol

import numpy as np
import numpy.typing as npt

_Q = math.log(10) / 400

#: `glicko._MAX_EXPONENT` and `glicko._MIN_SPREAD`, repeated for the reason
#: `_step` is: a compiled function can only read what it can compile, and
#: importing `glicko` from here would be a cycle (it imports this).
#: `glicko_test.test_the_steps_agree` covers a gap past the clamp from both
#: sides, so the two cannot drift apart silently.
_MAX_EXPONENT = 100.0
_MIN_SPREAD = 10.0**-_MAX_EXPONENT

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Bools = npt.NDArray[np.bool_]


class Sweep(Protocol):
    """One pass of the smoother over a season's games."""

    def __call__(
        self,
        home: Ints,
        away: Ints,
        actual: Floats,
        adjustment: Floats,
        week_end: Ints,
        entry_rating: Floats,
        entry_rd: Floats,
        in_preseason: Bools,
        settled: Floats,
        settled_known: Bools,
        initial_rd: float,
        weekly_rd_increase: float,
    ) -> tuple[Floats, Bools]: ...


def sweep(
    home: Ints,
    away: Ints,
    actual: Floats,
    adjustment: Floats,
    week_end: Ints,
    entry_rating: Floats,
    entry_rd: Floats,
    in_preseason: Bools,
    settled: Floats,
    settled_known: Bools,
    initial_rd: float,
    weekly_rd_increase: float,
) -> tuple[Floats, Bools]:
    """Re-walk the season, holding each opponent at its settled mean.

    One row per game in `home`/`away`/`actual`/`adjustment`, in the order
    they were played; `week_end[w]` is where week `w` ends, so the deviation
    widening lands between weeks the way the forward pass's does. Teams are
    rows: `entry_rating`/`entry_rd` are what the replay starts a team at --
    its preseason rating, or its anchor if the season hasn't opened with one
    -- and `in_preseason` says which of those two it was, because a team the
    season did not open with is not in the replay, and so is not aged, until
    it plays.

    `settled` is the previous pass's mean per team and `settled_known` which
    teams have one; an opponent without one is held at where this pass has
    it, which is what `settled.get(opponent, ...)` did.

    Returns the replayed means and which teams the replay reached. Only the
    means are the smoother's output -- the deviations here are internal, and
    the caller keeps the forward pass's.
    """
    rating = entry_rating.copy()
    deviation = entry_rd.copy()
    present = in_preseason.copy()

    start = 0
    for end in week_end:
        for i in range(start, end):
            h = home[i]
            a = away[i]
            # Both sides read before either writes: a game measures the two
            # ratings as they stood when it kicked off.
            hr, hd = rating[h], deviation[h]
            ar, ad = rating[a], deviation[a]
            adj = adjustment[i]
            score = actual[i]
            rating[h], deviation[h] = _step(
                hr, hd, settled[a] if settled_known[a] else ar, ad, score, adj
            )
            rating[a], deviation[a] = _step(
                ar, ad, settled[h] if settled_known[h] else hr, hd, 1 - score, -adj
            )
            present[h] = True
            present[a] = True
        for t in range(len(rating)):
            if present[t]:
                deviation[t] = min(
                    initial_rd,
                    math.sqrt(deviation[t] ** 2 + weekly_rd_increase**2),
                )
        start = end

    return rating, present


def _step(
    my_rating: float,
    my_rd: float,
    opp_rating: float,
    opp_rd: float,
    score: float,
    home_adjustment: float,
) -> tuple[float, float]:
    """`glicko.glicko_step`, on floats rather than `_Rating`s.

    Written out again rather than imported because the compiled sweep can
    only call something it can compile, and `_Rating` is a NamedTuple built
    78,000 times a probe for no one to read. The arithmetic is expression
    for expression the same, which is what keeps the last bit the same, and
    `glicko_test.test_the_steps_agree` holds the two together.
    """
    g_opp = 1 / math.sqrt(1 + 3 * _Q**2 * opp_rd**2 / math.pi**2)
    exponent = g_opp * (opp_rating - (my_rating + home_adjustment)) / 400
    # The same clamp `glicko_step` takes, for the same reason: a gap the
    # arithmetic cannot hold is a certainty either way, and unclamped the
    # division by `p (1 - p)` below is a division by zero. See
    # `glicko._MAX_EXPONENT`.
    exponent = min(max(exponent, -_MAX_EXPONENT), _MAX_EXPONENT)
    expected_score = 1 / (1 + 10**exponent)
    spread = max(expected_score * (1 - expected_score), _MIN_SPREAD)
    d2 = 1 / (_Q**2 * g_opp**2 * spread)
    rd_inv_sq = 1 / my_rd**2
    rd_inv_plus_d2 = rd_inv_sq + 1 / d2
    rd_new = math.sqrt(1 / rd_inv_plus_d2)
    rating_new = my_rating + (_Q / rd_inv_plus_d2) * g_opp * (score - expected_score)
    return rating_new, rd_new


def compiled_sweep() -> Sweep | None:
    """`sweep`, compiled, or None where numba isn't installed.

    Resolved once and remembered, because the answer cannot change inside a
    process and the import is not free. The compile itself is numba's, on
    the first call, and cached on disk after that (`cache=True`) -- about a
    second per container against the hours it saves.

    None rather than raising: the fallback is the dict sweep the predictor
    has always used, which is slower and not a different model. A fitting
    image has numba (the `fit` group); a serving install reading a release
    does not, and must not import it -- see `import_boundary_test.py`.
    """
    global _COMPILED, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            from ._compiled_smoothing import sweep as compiled  # noqa: PLC0415
        except ImportError:
            _COMPILED = None
        else:
            _COMPILED = compiled
    return _COMPILED


_TRIED = False
_COMPILED: Sweep | None = None
