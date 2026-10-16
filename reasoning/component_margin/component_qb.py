"""The component model with a missing quarterback priced by who is missing.

The flat `qb_out_penalty` charges every absence the same. Here a flagged
side is charged `qb_gap_scale` points more per 0.1 EPA per attempt by which
the expected starter was better than the man who started -- each measured
before kickoff, last season at any school plus this season to date, shrunk
toward a first-time starter by 150 attempts (scratch `qb_game_strength.py`
writes the index from a play sweep). A benching, where the new man has been
the better passer, is charged less, or credited.

In the edge, so in the update as well as the prediction, like every matchup
term: the ratings learn around it.
"""

import json
from pathlib import Path
from typing import Any

from cassandra.predictor.types import Matchup
from component import ComponentMarginPredictor

GAPS = Path.home() / ".cassandra/qb_strength/ncaafb_qb_gap.json"
_INDEX: dict[str, dict[str, float]] | None = None


def _gaps() -> dict[str, dict[str, float]]:
    global _INDEX
    if _INDEX is None:
        _INDEX = json.loads(GAPS.read_text())
    return _INDEX


class ComponentQbPredictor(ComponentMarginPredictor):
    def __init__(self, league: str, *, qb_gap_scale: float = 0.0, **kwargs: Any) -> None:
        super().__init__(league, **kwargs)
        self._qb_gap_scale = qb_gap_scale

    def matchup_adjustment(self, matchup: Matchup) -> float:
        points = super().matchup_adjustment(matchup)
        if not self._qb_gap_scale:
            return points
        gaps = _gaps().get(str(matchup.game_id))
        if not gaps:
            return points
        swing = gaps.get(matchup.away, 0.0) - gaps.get(matchup.home, 0.0)
        return points + self._qb_gap_scale * swing / self.points_per_rating
