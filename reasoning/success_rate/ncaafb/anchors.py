"""How a reading answers the anchors, across conferences that barely play.

The units expect each offense's reading at

    center + (edge + anchor gap) * per_play + offense offset - defense offset

so a conference's level is its anchor, converted to EPA per play by the same
`points_per_epa` that converts the offsets back. In-conference games have
no anchor gap and teach the sides only where a team stands among its peers;
the few games across conferences are what test the conversion. If a reading
moves less across an anchor gap than the conversion says, every such game
pushes the stronger conference's sides down and the weaker one's up, and
conference play never takes it back.

    python anchors.py record NAME [LABEL knob=value ...]
                                      one refit's replay, every contest logged
    python anchors.py report NAME...  the test, on the recorded contests

`record` reads the same environment `replay.py` does (`NCAAFB_INDEX`,
`NCAAFB_EPA`, `NCAAFB_SUCCESS_SCALE`), so a refit is recorded on the reading
it was fit to.

Reported per refit, on FBS offense v FBS defense contests from different
conferences: the slope of the surprise (reading minus expected) on the
anchor term -- 0 when the conversion is right for that reading, -0.3 when
the reading moves 30% less than the anchors say -- and the mean surprise
for power offenses against Group of Five defenses and the other way round.
Then each season's end, the offsets' mean by tier in points, which is where
a bias that conference play can't undo would pile up.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import POWER  # noqa: E402
from replay import DATA, OUT, run_from_2014, save  # noqa: E402

from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor  # noqa: E402
from cassandra.residuals import team_tiers  # noqa: E402

CONTESTS = DATA / "ncaafb_contests"


class Recording(UnitMarginGlickoPredictor):
    """The production model, writing down each contest before it steps."""

    contests: list[tuple] = []
    seasons: list[tuple] = []

    def _update_sides(self, game, edge):
        epa = self._game_epa.get(game.game_id)
        if epa is not None and epa.home_plays and epa.away_plays:
            home, away = self._reading(epa, self._success_weight)
            per_play = self.points_per_rating / (2 * self._points_per_epa)
            anchor_home, anchor_away = self.anchor(game.home), self.anchor(game.away)
            center = self.epa_center
            for offense, defense, reading, sign, a_off, a_def in (
                (game.home, game.away, home, 1.0, anchor_home, anchor_away),
                (game.away, game.home, away, -1.0, anchor_away, anchor_home),
            ):
                o, d = self._offsets(offense).offense, self._offsets(defense).defense
                expected = center + sign * (edge + anchor_home - anchor_away) * per_play + o.rating - d.rating
                self.contests.append((
                    self._season, game.game_id, offense, defense, reading - expected,
                    (a_off - a_def) * per_play, sign * edge * per_play,
                ))
        super()._update_sides(game, edge)

    def pass_season(self, year=None):
        for team in self._sides:
            self.seasons.append((
                self._season, team,
                (self.unit_rating(team) - self.anchor(team)) * self.points_per_rating,
            ))
        super().pass_season(year)


def record(name: str, label: str | None = None, **overrides: float) -> None:
    """`name`'s refit, its knobs moved by `overrides`, saved as `label`."""
    knobs = {**json.loads((OUT / f"{name}.json").read_text())["knobs"], **overrides}
    df = run_from_2014(knobs, predictor_class=Recording)
    name = label or name
    if overrides:
        save(name, df, knobs)
    CONTESTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Recording.contests, columns=[
        "year", "game_id", "offense", "defense", "surprise", "anchor_term", "edge_term",
    ]).to_parquet(CONTESTS / f"{name}.parquet")
    pd.DataFrame(Recording.seasons, columns=["year", "team", "offset_points"]).to_parquet(
        CONTESTS / f"{name}.seasons.parquet"
    )


def _tiers(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[str | None]]:
    games = pd.read_parquet(OUT / "published.parquet")
    tier = team_tiers(games, "ncaafb")
    assert tier is not None

    def label(team: str, year: int) -> tuple[str, str | None]:
        found = tier(team, int(year))
        if found is None or found[0] != "FBS":
            return "below", None
        group = "power" if found[1] in POWER or team.startswith("Notre Dame") else "G5"
        return group, found[1]

    out: dict[str, list] = {}
    for column in columns:
        labels = [label(t, y) for t, y in zip(frame[column], frame.year)]
        out[f"{column}_group"] = [g for g, _ in labels]
        out[f"{column}_conf"] = [c for _, c in labels]
    return out


def report(names: list[str]) -> None:
    print("cross-conference FBS contests: surprise on the anchor term, and by tier "
          "(EPA per play, x100)")
    print(f"{'':>22} {'contests':>8} {'slope':>6} {'t':>6}   {'power off v G5 def':>18} "
          f"{'G5 off v power def':>18}   {'in-conference sd':>16}")
    for name in names:
        c = pd.read_parquet(CONTESTS / f"{name}.parquet")
        c = c.assign(**_tiers(c, ["offense", "defense"]))
        fbs = (c.offense_group != "below") & (c.defense_group != "below")
        cross = c[fbs & (c.offense_conf != c.defense_conf)]
        x, y = cross.anchor_term.to_numpy(), cross.surprise.to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        se = np.sqrt((resid**2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
        pg = cross[(cross.offense_group == "power") & (cross.defense_group == "G5")].surprise
        gp = cross[(cross.offense_group == "G5") & (cross.defense_group == "power")].surprise
        inside = c[fbs & (c.offense_conf == c.defense_conf)].surprise
        print(f"{name:>22} {len(cross):8d} {slope:+6.3f} {slope / se:+6.2f}   "
              f"{100 * pg.mean():+7.2f} ({100 * pg.std() / np.sqrt(len(pg)):.2f})   "
              f"{100 * gp.mean():+7.2f} ({100 * gp.std() / np.sqrt(len(gp)):.2f})   "
              f"{100 * inside.std():16.2f}")

    print("\nend of season: mean offset from the anchor in points, power minus Group of Five")
    rows = {}
    for name in names:
        s = pd.read_parquet(CONTESTS / f"{name}.seasons.parquet")
        s = s[s.year.notna()]
        s = s.assign(**_tiers(s, ["team"]))
        s = s[s.team_group != "below"]
        by = s.groupby(["year", "team_group"]).offset_points.mean().unstack()
        rows[name] = (by["power"] - by["G5"]).round(2)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "record":
        extra = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
        record(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, **extra)
    else:
        report(sys.argv[2:])
