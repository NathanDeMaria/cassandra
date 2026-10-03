"""`smoothing.sweep` and the step it calls, compiled.

Importing this imports numba, so nothing imports it directly:
`smoothing.compiled_sweep()` is the one door, and it treats an ImportError
as "this install doesn't fit models" rather than as a failure.

The loop itself is not written again here. `_step` is swapped for its
compiled self *in the module the sweep came from*, because numba resolves a
called global at compile time: left alone, the compiled sweep would drop
back into the interpreter twice per game and run slower than the dict
version it exists to replace. Mutating `smoothing._step` is what makes one
copy of the arithmetic serve both -- and the pure-Python `sweep` is not
reached when this import succeeds, so nothing ends up with a half-compiled
sweep.
"""

from numba import njit

from . import smoothing

# A Dispatcher where a function was, which is the swap itself; nothing calls
# it from Python once the compiled sweep is the one in use.
smoothing._step = njit(cache=True, inline="always")(  # ty: ignore[invalid-assignment]
    smoothing._step
)

#: `fastmath` is deliberately off (njit's default). It would let the
#: compiler reassociate the rating update, and a model is fitted at one
#: last bit and published at another. See the module docstring in
#: `smoothing` for what the two paths do and don't agree on.
sweep = njit(cache=True)(smoothing.sweep)
