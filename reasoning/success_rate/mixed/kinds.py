"""Dropback or run, and a side's success rate at a fixed mix of the two.

A pooled success rate is wins over downs, so it moves with how often an
offense calls the kind of play that succeeds more often: dropbacks succeed
48% of the time on the NFL and runs 39%, and the offense calling more
dropbacks is usually the one behind. The mixed rate holds the mix fixed,

    mixed = w * pass_rate + (1 - w) * run_rate

each kind's rate shrunk toward the side's pooled rate plus that kind's
league gap by `prior` downs, so a triple-option offense's five dropbacks
don't carry `w` of a game on their own.

What counts as a dropback is read off the play type, as the feed files it:
a pass of any kind, a sack, an interception -- and a "Rush" whose text says
the quarterback scrambled, which is a dropback the defense flushed. A fumble
or a safety goes with the kind its text names. The sweep's own rule is
`cassandra.epa_build.is_pass`, written from this one; `kind_of` is
what the experiments here ran on (the two differ only on a fumble whose
text says "scrambled").
"""

import re
from typing import NamedTuple

PASS, RUN = "pass", "run"

_DROPBACK_TYPES = re.compile(r"pass|sack|interception")
_SCRAMBLE = re.compile(r"\bscrambles?\b|\bscrambled\b")
_DROPBACK_TEXT = re.compile(r"\bpass\b|\bsacked\b|\bscrambles?\b")


def kind_of(play_type: str | None, text: str | None) -> str:
    """PASS or RUN for a down `cassandra.epa_build.is_down` keeps."""
    lowered = (play_type or "").lower()
    words = (text or "").lower()
    if _DROPBACK_TYPES.search(lowered) or _SCRAMBLE.search(words):
        return PASS
    if ("fumble" in lowered or "safety" in lowered) and _DROPBACK_TEXT.search(words):
        return PASS
    return RUN


class Counts(NamedTuple):
    """One side's downs and successes, by kind."""

    pass_downs: float
    pass_wins: float
    run_downs: float
    run_wins: float

    @property
    def downs(self) -> float:
        return self.pass_downs + self.run_downs

    @property
    def pooled(self) -> float | None:
        return (self.pass_wins + self.run_wins) / self.downs if self.downs else None


def mixed_rate(counts: Counts, w: float, pass_gap: float, run_gap: float, prior: float) -> float | None:
    """The side's rate at `w` dropbacks, each kind shrunk by `prior` downs.

    `pass_gap` and `run_gap` are the league's kind rates less its pooled
    rate, so a kind with no downs reads as the side's pooled rate moved by
    the league's gap.
    """
    pooled = counts.pooled
    if pooled is None:
        return None
    p = (counts.pass_wins + prior * (pooled + pass_gap)) / (counts.pass_downs + prior)
    r = (counts.run_wins + prior * (pooled + run_gap)) / (counts.run_downs + prior)
    return w * p + (1 - w) * r
