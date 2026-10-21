---
name: kalshi-ledger
description: Update the Kalshi betting ledger week over week. Rebuilds the report of every bet from the account, writes a theme (why it won or lost) for each newly settled game, and reports the week and the running totals by result type. Use when asked to update, refresh or review the ledger, how the bets did this week, why bets won or lost, or bet-tracking metrics.
---

# Updating the Kalshi ledger

`make ledger` rebuilds everything from the Kalshi account each run. The account is
read-only here, as in `make positions`. It writes `~/.cassandra/betting/ledger/report.md`
and `ledger.csv`. The numbers and the box-score tags are automatic. The **themes** are
the judgment part, and that's this skill's job. Each settled game gets one entry in
`~/.cassandra/betting/ledger/themes.toml`. The report adds them up across weeks, so the
types have to be applied the same way every week.

## 1. Rebuild and list what needs a theme

```bash
env -u VIRTUAL_ENV make ledger ARGS=--digest
```

- **Kalshi key.** The client reads `KALSHI_KEY_ID`/`KALSHI_KEY_FILE` or `~/.kalshi/`.
  In the devcontainer the key is at `/workspaces/cassandra/.kalshi/`, so export
  `KALSHI_KEY_ID=$(cat /workspaces/cassandra/.kalshi/key_id)` and
  `KALSHI_KEY_FILE=/workspaces/cassandra/.kalshi/private_key.pem` first.
- **Which Python.** Run it from a worktree whose `.venv` is current (the `kalshi-tidy`
  one has its own). The shared `/workspaces/cassandra/.venv` is stale.
- **No AWS login.** Without one it prints `no s3 (...)` and carries on. What's lost:
  the cloud job's bedtime bets show no model price, and the themes aren't backed up to
  s3. If bedtime was used that week, ask Nathan to log in (see the `aws-login-from-phone`
  memory) rather than settle for `--offline`.

The summary lines say what's missing. Watch for:

- `no model price for: ...` — a bet with no orders log and no picks run before its
  first fill. It still counts everywhere except model EV.
- `no ESPN game for: ...` — no picks run listed the game, so it has no game id. It gets
  no box score, tags or theme digest.

`--digest` then prints one block per settled game without a theme (`=== KX...`), with:

- the price paid, model, close and result;
- the DraftKings line and the win-probability path;
- both teams' yards, rushing, turnovers, third downs and passers;
- the tags and the box-score read;
- the **suggested type**;
- every scoring play;
- the AP recap.

## 2. Write each theme

Append one table per game to `themes.toml`, under a `# --- Week of <Sat> ---` comment:

```toml
[KXNCAAFGAME-26OCT16MEMTULN]
type = "turnovers"
theme = "Turnover avalanche"
why = "Tulane forced 6 turnovers (1 in its first five games), one a pick-6 at the half, and won while outgained 488-325."
```

- `theme` is a few words, like a headline: "Wild upset", "Blown late lead",
  "Dominant run game by the more talented team".
- `why` is one or two sentences with the specific facts: who, what, the numbers that
  decided it. Write it from the bet's side; "us" is the team bet on.
- No double quotes inside the strings. TOML needs them escaped, so use single quotes or
  none.

**`type` is exactly one of these** (`cassandra.ledger.TYPES`). Pick the one that most
decided the result, from the bet's side:

| type | use when |
|---|---|
| `late_swing` | decided in the last ~5 minutes or OT: a late drive, a walk-off kick, a late lead lost, a goal-line call |
| `trenches` | the run game or the line of scrimmage decided it: one side ran over the other |
| `turnovers` | turnovers or return TDs decided it, often while the box score said otherwise |
| `qb` | a quarterback decided it: an injury, a benching, a debut, a career day or a meltdown |
| `outclassed` | one side was better all game: a rout, or a game never in doubt |
| `upset` | the underdog simply outplayed the favorite, no single cause |
| `special_teams` | kicks, punts or returns decided it |
| `officiating` | a call or a review swung it |

**Judgment rules**, so the weeks stay comparable:

- **Last minutes beat the rest.** If a one-score game turned in the final minutes, it's
  `late_swing`, even when something else set it up. Name the other cause in `why`.
- **Trust the box score when they disagree.** The suggested type is a heuristic. Take it
  only when the recap and numbers agree. For example, a blowout's second QB is a backup,
  not a `qb` game.
- **`upset` is for when no single cause stands out.** If the underdog won by running the
  ball, it's `trenches`.
- **Note big price moves.** If the market moved a lot toward the bet (CLV of 10¢ or
  more) and it still lost, say so in `why`: a good bet, a bad outcome.

`make ledger` checks every type and stops with the bad entry named if one isn't in the
list.

## 3. Rebuild and report

```bash
env -u VIRTUAL_ENV make ledger
```

When a theme exists, the `?` after a type in the report disappears. Check the last line
reads `0 settled games without a theme`.

Then tell Nathan, in chat:

1. **The week.** Record, P&L, ROI, the model's EV vs the close's EV, and CLV, from the
   week's heading line. Then each game's theme in one line, wins first.
2. **Running totals.** The overall card, plus anything that moved in the result-type,
   box-score-read and tag tables. Lead with what's notable, e.g. "late_swing is 7–3",
   or "every 'deserved' loss was an away favorite".
3. **What's open.** The count, the stake, the model EV, and how the marks moved.

Read the totals the way `betting-tools` does: CLV and the close's EV are the steadier
signal; W–L over a few dozen bets is mostly noise. Quote the sd line before reading
anything into a run of wins or losses.

## Where things live

- The **code**: `cassandra/ledger.py` (netting, prices, facts, tags, report) and
  `ledger.py` (fetching, caching, s3). Tests are in `cassandra/ledger_test.py`.
- The **data**, all in `~/.cassandra/betting/ledger/`:
  - `themes.toml`, the themes;
  - `ledger.csv`, one row per bet;
  - `report.md`, the report;
  - `kalshi/`, final events and prices, cached.
- The **ESPN summaries** are cached in `~/.cassandra/picks/summaries/`.
- **Bet-time prices** come from `~/.cassandra/betting/orders/*_orders.csv` (exact, by
  order id). Failing that, from the last `~/.cassandra/betting/picks/*.csv` run started
  before the first fill.
- The cloud job copies its bedtime runs' files to `s3://<batch>/cassandra/betting/`
  under `picks/` and `orders/`, and `make ledger` pulls them down.
- **Backups** go to `s3://<batch>/cassandra/betting/ledger/` after each run with a
  login. If `themes.toml` is missing locally, it's restored from there.
