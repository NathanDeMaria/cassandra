# Where ncaafb `glicko_full` is wrong, September 2026

A sweep of the residuals of the 2026-09-16 `glicko_full` fit over every
ncaafb game since 2002 (75,113 of them), beyond the axes `diagnose.py`
already ran, plus a per-team home-field study and a look at what the two
compound fits' units actually do. The measurements below are the reason the
axes `previous_surprise`, `kickoff_hour` and `conference_game`, the
`margin_slope` column and `home_field_by` exist in `cassandra.residuals`;
the numbers are recorded here so the next sweep can say what changed.

Every number is a margin residual: actual margin minus the margin the win
probability implies through the MAE logistic fit, positive when the home side
beat the model. Margin MAE 12.76, residual sd 16.2.

## What has structure

**The residual is autocorrelated, and it is the season's shape rather than
the update's.** A team's residual on its previous surprise (own side, same
season) has slope +0.027 (t 9.9), identical in every division and at every
rating deviation. Cross-validated across odd/even seasons, a correction on
the last three games' surprise is worth +0.013 margin MAE and +0.00018
brier -- between the compound model's gain and `passes=2`'s. But it is not
symmetric:

    surprise this game      +1     +2     +3     +4 games later
    worse than -35        -1.17  -1.27  -1.14  -0.99
    better than +35       +1.15  +0.06  -0.52  -0.26

A team beaten by 35 more than expected stays a point worse for a month:
something about it changed. A team that won by 35 more than expected is
under-rated for one game and then isn't. The first is the `qb_out` index's
territory (the changes it misses); neither is the update rule's, and a
Gaussian- or t-observation Glicko with its clocks re-tuned leaves the
autocorrelation where it was (see below).

**Cross-division margins are over-dispersed.** Slope of the actual margin on
the predicted one: 0.96-1.03 inside every division, **0.85 (se 0.02) for FBS
hosting FCS**, 0.82 FCS hosting D-II. Modest FBS favorites over FCS win by
four more than predicted, 35-point favorites by 1.3 less; the bias cancels
and only the slope sees it. Shrinking the cross-tier gap by that slope gains
+0.16 MAE on those 2,511 games out of sample. This is the division *scale*
where `division_anchors` fixes the division *level*, and the FBS-vs-FCS
games are also the widest market gap of any lined slice (+2.1 vs +1.8 for
FBS-vs-FBS).

**The designated home side at a neutral site is +1.5** (n 1,650; +2.0 in the
regular season). Already an axis (`site`), recorded here because it is the
cleanest small knob in the list.

**Kickoff time.** FBS-vs-FBS: noon -0.1, 2-4pm +0.76, 8pm+ Central +0.9;
FCS 8pm+ +2.2. About 2 sigma inside FBS on its own, replicated in FCS, flat
in D-III (everything is at noon). Confounded with who gets televised.

**Early-season non-conference.** FBS-vs-FBS non-conference games in weeks
1-4 run +1.34 for the home side (home dogs +1.9); conference games in the
same weeks -0.9; non-conference after week 4 +0.14.

**Flat:** day of week (no midweek effect). No persistent per-team error:
year-to-year correlation of a team's mean residual -0.03, and a team's
history over all prior seasons predicts this season at slope +0.09 (FBS
-0.12). Stretching the predicted margin by an expected total is
inconsistent across folds.

## Home field by team, and by theme

Per team (`home_field_table`): the true spread of home advantages around the
3.06-point constant is about 1.07 points, and a team's estimate carries a
2.3-point standard error -- reliability 0.18, odd-vs-even-season correlation
0.09. Hawai'i (+7.3, t 3.9) is the one team that is real on its own. Alabama
and Georgia show *negative* excess, which is the favorite compression above,
not home field. Do not build a per-team parameter.

Pooled (`home_field_by`, own residual at home minus away, FBS-vs-FBS unless
noted):

- Division: FBS **+0.81 (t 4.5)**, FCS -0.1, D-II -0.44, D-III -0.42. One
  constant is 1.2 points off between tiers. The cheapest fix in this file.
- Travel: home residual -0.5 when the visitor came under 300 km, +1.3 over
  1,500 km; ~0.7 points per 1,000 km of the game's own trip, with venue
  remoteness adding little on top. It is why the geography reads the way
  it does -- west of -95 +1.4 to +2.4 at every stage of the season, east
  -0.4 to +0.6; Big 12 **+2.5 (t 5.0)**, MAC -0.7 -- and why Texas Tech,
  UTEP, Colorado, San Jose State, Nevada and Hawai'i top the team table
  while Vanderbilt, Duke, Northwestern, Rutgers and the MAC sit at the
  bottom. `travel.py` pinned `travel_advantage` at 0 on a 0.00015 brier
  gain; in margin terms it is about 1.8 points between a short trip and a
  long one.
- Program quality: top-quartile FBS programs **+1.2 (t 3.4)**, holding at
  +1.5 in games where they are not big favorites. Stadium capacity adds
  nothing once quality is controlled (they correlate at 0.81).
- 2020, no crowds: home bias **-1.6** (n 570, 2.4 sigma). Half the constant.
- Season stage: excess ~+1.1 in weeks 1-5, +0.65 in 6-10, +0.3 after 11.
- Weak: altitude +0.7 per 1,000 m after controls (1.7 sigma, not separable
  from "west"); domes +1.2 FBS-only (t 1.2); no late-season cold effect.

## The compound model's units

Both `glicko_compound` results on disk sit on the *old* parent fit (home
advantage 41, no passes): 0.15777 and 0.15782 against glicko_full's
0.15426. The configs already pin the new fixed block; they need the
re-search. Against their own parent the units are worth +0.8e-4 and
+0.5e-4 brier.

Where they help (parent residual on the units' disagreement with the record,
in points per rating point): early season 0.026 vs late 0.000; parent rd
250-300 0.026 vs rd under 150 0.003; non-conference 0.019 vs conference
0.009; FCS-vs-FCS 0.004 (t 4.0) vs FBS-vs-FBS 0.001 (t 0.9) in the narrow
fit. The blend applies about half the useful slope. Offense-minus-defense
*shape* carries nothing. Which unit is informative flips between the two
fits (wide: defense; narrow: offense), and the forward test settles it: on
the next game's residual, a team's own offense EPA carries +0.42 per sd
(t 4.8) and its defense side -0.09 (t -1.0).

## The observations' distributions, and a prototype

Residuals about the model's expectation, on the 25k games with plays:

    observation                         sd     kurtosis   t nu
    scoreboard margin                 15.9 pts   +0.17     36
    EPA margin (clip 3)               14.4 pts   +0.52     14
    control, logit                    1.13 lgt   -0.24     inf   (17.5 pts/logit)
    EPA per play, unweighted          0.196      +1.2      11
    EPA per play, garbage-time wtd    0.435      +4.1      5.5

The margin is Gaussian to three decimals of log-likelihood; logit(control)
is Gaussian; only the weighted per-play EPA -- what the compound rates its
units on -- has a real tail, and the fitted compound scales (32-37 logits
per EPA point) score 68-72% of contests outside [0.02, 0.98], i.e. as a
coin. The blend's EPA margin at scale ~11 is 21% saturated against the
scoreboard's 2.4%.

A Glicko whose observation is the margin itself (a Kalman step in rating
units, the opponent's rd folded into the noise, the smoother replayed with
the same step; a t likelihood as one-step IRLS) was prototyped with
glicko_full's other knobs and ~40 hand probes:

    glicko_full (1000-probe search)      brier 0.154181   MAE 12.756   slope on 21+ favorites 0.986
    Gaussian, sd 13, rd 200/60/5         brier 0.154292   MAE 12.744   0.995
    t nu=8, same                         brier 0.154647   MAE 12.768
    rd inflation after big surprises     worse at every setting

The Gaussian ties the searched model on 4% of the probes and removes the
favorite compression; every finite nu is worse, as the nu=36 residual said
it would be; the autocorrelation is unchanged. That model is the subject of
its own PR.
