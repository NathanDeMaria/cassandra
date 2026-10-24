"""Prototype: unsettle a team's ratings when its quarterback or its head coach changes.

`summer_change.py` found that on ncaafb a new week-one quarterback leaves an
offense keeping less of last season and drawing more new variance, and a
head coach leaving cuts what both sides keep. The published success model
reads a new quarterback only as a shift and a coach leaving for a job as a
shift; these knobs, all 0 by default (the published model to the bit), add
what the measurement says is missing:

- `new_qb_offense_sd`: the offense unit's deviation widened by this (past its
  cap, as `coach_change_offense_sd` does) when the new starter is known --
  after the team's first game with plays, where the shifts wait too;
- `new_qb_rd`: the record's deviation widened the same way, under `initial_rd`;
- `new_qb_offense_regression`: that share of the offense offset the summer
  carried over taken back to the anchor, at the same moment;
- `coach_change_offense_regression`, `coach_change_defense_regression`: each
  side's offset pulled that share of the way to the anchor at the rollover,
  for a team whose head coach left for any reason;
- `coach_change_rating_regression`: the record rating the same, toward the
  team's anchor.

Each lands on the preseason copy the smoother re-walks from as well, as
`_shift` does.

    python change_knobs.py NAME [knob=value ...]

Replays 2014-2026 from the harness's 2013 snapshot on the published success
model's knobs, and writes out/change_NAME.parquet and .json: FBS v FBS
brier on even and odd seasons -- the two folds -- and by slice.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "ncaafb"))
import replay  # noqa: E402  (pins the data)
from endgame.types import Game  # noqa: E402

from cassandra.offseason import OffseasonFacts  # noqa: E402
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from cassandra.predictor.types import Prediction  # noqa: E402
from cassandra.predictor.unit_glicko import (  # noqa: E402
    UnitMarginGlickoPredictor,
    _Sides,
)

OUT = Path.home() / ".cassandra/success_rate/mixed/out"
PUBLISHED = json.loads((replay.OUT / "batch_nc_success_cap_p.json").read_text())["knobs"]
KNOBS = ("new_qb_offense_sd", "new_qb_rd", "new_qb_offense_regression",
         "coach_change_offense_regression", "coach_change_defense_regression",
         "coach_change_rating_regression",
         # The production model's own widenings for a coach change, which the
         # ncaafb success model leaves at 0.
         "coach_change_rd", "coach_change_offense_sd", "coach_change_defense_sd")


class ChangeKnobs(UnitMarginGlickoPredictor):
    def __init__(self, *args, new_qb_offense_sd: float = 0.0, new_qb_rd: float = 0.0,
                 new_qb_offense_regression: float = 0.0, coach_change_offense_regression: float = 0.0,
                 coach_change_defense_regression: float = 0.0, coach_change_rating_regression: float = 0.0,
                 **kwargs) -> None:
        self._new_qb = (new_qb_offense_sd, new_qb_rd, new_qb_offense_regression)
        self._coach_regression = (coach_change_offense_regression, coach_change_defense_regression,
                                  coach_change_rating_regression)
        super().__init__(*args, **kwargs)

    def _roll_over(self) -> None:
        super()._roll_over()
        self.regress_coach_changes()

    def regress_coach_changes(self) -> None:
        offense, defense, record = self._coach_regression
        if not (offense or defense or record):
            return
        for team, fact in self._offseason.seasons(self._season):
            if fact.coach_departure is None:
                continue
            for sides in (self._sides, self._sides_preseason):
                if team in sides:
                    pair = sides[team]
                    sides[team] = _Sides(
                        pair.offense._replace(rating=(1 - offense) * pair.offense.rating),
                        pair.defense._replace(rating=(1 - defense) * pair.defense.rating),
                    )
            if record and team in self._ratings:
                anchor = self.anchor(team)
                for ratings in (self._ratings, self._preseason):
                    if team in ratings:
                        current = ratings[team]
                        ratings[team] = current._replace(
                            rating=anchor + (1 - record) * (current.rating - anchor))

    def _update_game(self, game: Game) -> Prediction:
        known = []
        if any(self._new_qb) and self._game_epa.get(game.game_id) is not None:
            known = [t for t in (game.home, game.away) if t in self._qb_pending]
        carried = {t: self._sides_preseason.get(t) for t in known}
        prediction = super()._update_game(game)
        for team in known:
            fact = self._offseason.get(team, self._season)
            if fact is not None and fact.new_quarterback:
                self._unsettle(team, carried[team])
        return prediction

    def _unsettle(self, team: str, carried: _Sides | None) -> None:
        sd, rd, regression = self._new_qb
        for sides in (self._sides, self._sides_preseason):
            if team not in sides:
                continue
            offense = sides[team].offense
            if regression and carried is not None:
                offense = offense._replace(rating=offense.rating - regression * carried.offense.rating)
            if sd:
                offense = offense._replace(rating_deviation=math.hypot(offense.rating_deviation, sd))
            sides[team] = sides[team]._replace(offense=offense)
        if rd:
            for ratings in (self._ratings, self._preseason):
                if team in ratings:
                    current = ratings[team]
                    ratings[team] = current._replace(
                        rating_deviation=min(self._initial_rd, math.hypot(current.rating_deviation, rd)))


def run(knobs: dict) -> pd.DataFrame:
    named, _, index = replay._inputs()
    warm, early = replay._warm()
    predictor = ChangeKnobs.from_state_dict({
        **warm.state_dict(), **PUBLISHED, **knobs,
        "opponent_prior_manager": OpponentPriorManager("ncaafb", path=replay.PRIORS),
        "game_epa": index,
    })
    predictor._season = warm._season
    predictor._qb_pending = set(warm._qb_pending)
    # 2014's rollover ran in the warm-up, on the published knobs.
    predictor.regress_coach_changes()
    late = replay._replay(predictor, [s for s in named if s.year >= replay.FIRST_READING])
    return pd.concat([early, late], ignore_index=True)


def tagged(df: pd.DataFrame) -> pd.DataFrame:
    """FBS v FBS from 2014, with each game's slices."""
    df = df.reset_index(drop=True)
    sides = pd.concat([
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.home_team, "date": df.date}),
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.away_team, "date": df.date}),
    ]).sort_values(["date", "row"], kind="stable")
    sides["n"] = sides.groupby(["year", "team"]).cumcount() + 1
    df["game_no"] = sides.groupby("row").n.min()
    facts = OffseasonFacts.for_league("ncaafb")

    def has(team, year, test):
        fact = facts.get(team, int(year))
        return fact is not None and test(fact)

    df["new_qb"] = [has(h, y, lambda f: f.new_quarterback) or has(a, y, lambda f: f.new_quarterback)
                    for h, a, y in zip(df.home_team, df.away_team, df.year)]
    df["coach"] = [has(h, y, lambda f: f.coach_departure is not None)
                   or has(a, y, lambda f: f.coach_departure is not None)
                   for h, a, y in zip(df.home_team, df.away_team, df.year)]
    df = df[df.fbs & (df.year >= 2014)].copy()
    df["loss"] = (df.team1_win_prob - df.team1_win) ** 2
    return df


def scores(df: pd.DataFrame) -> dict[str, float]:
    t = tagged(df)
    even = t.year % 2 == 0
    out = {"even": t.loss[even].mean(), "odd": t.loss[~even].mean(), "all": t.loss.mean()}
    for label, mask in (("games 1-4", t.game_no <= 4), ("games 5+", t.game_no > 4),
                        ("new QB", t.new_qb), ("coach left", t.coach)):
        out[f"{label} even"] = t.loss[mask & even].mean()
        out[f"{label} odd"] = t.loss[mask & ~even].mean()
    return {k: float(v) for k, v in out.items()}


if __name__ == "__main__":
    name = sys.argv[1]
    knobs = {k: float(v) for k, v in (pair.split("=") for pair in sys.argv[2:])}
    assert set(knobs) <= set(KNOBS), set(knobs) - set(KNOBS)
    began = time.time()
    df = run(knobs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"change_{name}.parquet")
    result = {"knobs": knobs, **scores(df)}
    (OUT / f"change_{name}.json").write_text(json.dumps(result, indent=2))
    print(name, knobs, f"even {result['even']:.6f} odd {result['odd']:.6f} ({time.time() - began:.0f}s)", flush=True)
