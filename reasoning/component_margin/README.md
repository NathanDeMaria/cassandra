# One estimate in parts: the component margin prototype

2026-10-15 to 2026-10-16. The published ncaafb model (`glicko_margin_units_success`)
keeps two estimates of each team, a record stepped on margins and offense/defense units
stepped on plays. It averages them by precision at prediction and reads a win probability
off a logistic whose scale was searched apart from either filter. That is the mishmash
this prototype replaces with one estimate in four parts per team: level L, offense O,
defense D and rest R. The margin is predicted from their sum as a Normal in points, and
the win probability is read off it. `component.py` has the design.

**The upshot:** on brier it ties the published structure refit the same way, and it is
worse on margins. Fit through 2021 and held out on 2022-2026, it is no better on the win
probability than the published structure given the same refit. Read through the margin
kernel, it is worse in both windows. Its own Normal is far behind any kernel, because the
loss is the distribution's shape, not its spread. The parts are not yet a better model.
They are a working harness for asking why.

## Running it

    python run.py NAME [knob=value ...]          # one replay, to ncaafb_out/cm_NAME (~40 s)
    python search.py NAME [--rounds N]           # parallel pattern search, FBS brier 2014-2021
    ./variants.sh logs/NAME.best.json PREFIX "label knob=v ..." ...   # eight replays at once
    python summarize.py PREFIX ...               # brier by window against the published model
    python compare.py cm_NAME [against]          # paired: brier, kernel log loss/RPS, own Normal

Every replay resumes from the success-rate harness's 2013 snapshot of the published model
(`../success_rate/ncaafb/replay.py`), on its pinned predictor data. Nothing has a reading
before 2014, so L starts as the record. Run with `PYTHONPATH` at this checkout. Outputs go
to `~/.cassandra/success_rate/units/ncaafb_out/cm_*`.

## What the search found

A pattern search ran four rounds on 18 knobs (`search.py`, 8 replays at a time, ~10 min a
round), starting from the published reading knobs. It improved FBS brier on 2014-2021
from 0.173865 to 0.171381. The accepted moves:
- the Normal's spread: `game_sd` 14.5, `uncertainty_scale` 0.45;
- the points scale: `points_per_rating` 0.127, home 20.7 rating points;
- `success_weight` 0.59 → 0.29;
- `offense_carry` 0.7, `rest_carry` 0.3, `initial_rd` 260.

Offense and defense want a small prior, 0.05 per play: from 0.19 down to 0.05 took FBS
brier from 0.1857 to 0.1739.

Against the published model (brier 1e-4, FBS v FBS, paired t; negative favors the parts):

| | 2014-21 | 2022-26 held out | 2014-26 |
|---|---|---|---|
| vs published success model | -8.5 (t -1.3) | +12.2 (t +1.5) | -0.8 |
| vs the published structure refit through 2021 (`nc_fit_free_p`) | -2.7 (t -0.5) | +5.1 (t +0.8) | +0.2 |
| margin kernel log loss, vs the same refit (1e-3 nats) | +8.4 (t +3.5) | +15.5 (t +4.4) | +11.1 |
| own Normal vs the published kernel (1e-3 nats) | +166 | +186 | +173 |

The parts beat the published model from 2014 to 2020 and lose every season from 2021 on.
Any refit through 2021 loses held out, so part of that is the window: the published
structure refit the same way loses 4 to 7 too. Against that refit the brier difference is
noise. The margin isn't.

The Normal's coverage is honest: 70% of FBS games fall within one sd and 96% within two,
with z sd 0.97. The 0.17 nats it gives up to a kernel is the shape: no mass on the key
numbers and some on a zero margin. The mean is too wide as well. Actual margins regress
on mu at 0.84, as they do on the published model's own point margin (0.85-0.88).

## Locating the gap

Each row is one change from the search's best, replayed through 2026, in brier 1e-4
against the published model:

| change | 2014-21 | 2022-26 |
|---|---|---|
| the search's best | -8.4 | +12.2 |
| readings off (L is a record, nothing else moves) | +7.7 | +18.9 |
| published model with `unit_weight` 0, for comparison | +9.6 | +17.5 |
| success weight back to 0.59 | -2.8 | +20.4 |
| returning production on L instead of O/D | -6.9 | +13.7 |
| O 0.10 / O 0.15 per play | +45 / +132 | +48 / +120 |
| sides kept over the summer at the units model's carries | -2.9 | +15.2 |
| the same, defense widening 0.02 a season | -6.4 | +10.2 |
| R wide and carried (sd 5 / 8) | -3.9 / +1.6 | +9.8 / +12.9 |
| L stepped on the whole margin in every game, readings to O/D only | +10.7 | +16.1 |
| missing QB priced by the starter-less-replacement gap, 1 pt per 0.1 | -9.3 | +13.0 |

- **The record side is not it.** With readings off, the parts reproduce the published
  model at `unit_weight` 0 to within 2. The Normal and the record are equivalent.
- **The plays are.** The published units are worth -9.6 on 2014-2021 and -17.5 on
  2022-2026, more in the later window. The parts' readings are worth -16 and -6.7.
- **The scale is not the problem.** In the reading model, the read per-play gap tracks
  what the parts expected (slope 0.95 overall, 0.89 FBS; `diag_scale.py`), steadily
  across seasons.
- **What the units do that the parts don't.** A unit sitting at its prior adds nothing
  to the blend, since the blend counts only precision earned past the prior. So its noise
  never reaches a prediction. In the parts every bit of a side's movement does, which is
  why any prior over ~0.05 per play is ruinous here and harmless there. Making the parts
  remember sides the units' way, or step L on margins the record's way, doesn't recover
  the held-out value.

## Quarterbacks out, by who is out (`qb/`)

A play sweep (`qb/qb_sweep.py`, 18,602 ncaafb games 2013-2026) records each side's starter,
who took snaps, and every passer's attempts and EPA. `qb/qb_game_strength.py` replays the
production flag rule with identities. Each man's quality before kickoff is last season at
any school plus this season to date, shrunk 150 attempts toward a first-time starter
(-0.034 EPA/att). On the roster model's residuals, the flagged side's miss per 0.1 EPA/att
by which the man out was better than the man in:

| | 2014-21 | 2022-26 | 2014-26 |
|---|---|---|---|
| per 0.1 of starter less replacement | -2.27 ± 0.54 | -0.86 ± 0.61 | -1.65 ± 0.41 |

That's real on margins pooled, and smaller lately. On win probabilities it buys nothing
held out: post-hoc (`qb/qb_heldout.py`), from +0.2 to +4.6 per FBS game-season depending
on the form. In the parts' replay at 1 point per 0.1 (`component_qb.py`) it is -0.9
(t -0.9) on 2014-2021 and +0.8 (t +0.5) after.

## Where to go from here

- **The distribution's shape** is the biggest single loss and the most separable one. A
  kernel driven by the parts' mu and sd rather than by a win probability would keep the
  per-game spread the parts carry and add the key numbers. Score it against the published
  kernel held out.
- **Earned-only parts.** A side contributes `S * earned / (earned + prior)` of its mean:
  the units' shrink, kept inside one additive model. It is the cheapest test of whether
  the shrink is what the parts lack.
- **Fit both on every season.** The published knobs saw 2022-2026 and these didn't.
  Equal footing would take a search of the parts over 2014-2026, or an interleaved split
  (fit odd seasons, score even ones) for both.
