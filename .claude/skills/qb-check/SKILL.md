---
name: qb-check
description: Check Kalshi picks the cloud job held on a QB check, and place the ones Nathan approves. Reads the holds from `make picks-status`, looks up each held team's quarterback news, sets what's actually going on beside the edge as priced and the edge if the starter sits, asks yes or no per game, and places the yeses with `make bet`. Use when a "QB check" hold email lands, when asked to check, research or place a held pick ("another landed", "how about Tulane?", "place 1 from the hold email"), or on /qb-check.
---

# Checking a QB hold

The cloud picks job holds a pick when the side bet on had its starter miss, leave
or come off the bench in its last game, or when the injury news names a
quarterback (`cassandra.qb_starts`, `placing.qb_news`). The model reads neither.
So a hold asks one thing: **will the backed team's usual starter sit?** The job
has already priced both answers. This skill finds out which is true, asks, and
places. Keep it cheap: no pricing run before the question.

## 1. What's held

```bash
make picks-status
```

It lists the last five emails. A held pick shows its game and kickoff, its
`edge` as priced, the edge `if its starter is out`, and the `QB:` sentence that
held it. Take the newest email's holds unless Nathan names a team or game. One
from an earlier email may already be placed or passed on; ask if the
conversation doesn't say.

A hold with no edge lines was emailed before they were recorded. Price it by
hand (below).

## 2. The news: one search per team

Run one `WebSearch` in standard mode per held team, naming both men from the
`QB:` sentence: `<team> quarterback <starter> <who started instead>`. Search a
second time only if the first doesn't say who's expected to start next. Use
extended mode only if both come back thin. When a summary is vague on injury
versus benching, `WebFetch` the article. Never trust a summary's score for a game
not yet played; one invented a final on 2026-11-08.

Sort each hold by what's going on, which says which edge is the real one:

| What's going on | Real edge |
|---|---|
| Starter hurt and ruled out, or out for the season | if out |
| Starter rested against a weak opponent, or day-to-day and reported back | as priced |
| Preseason starter back from injury, so the "missing" man was the fill-in | as priced |
| Benching: the coach picked the new man | as priced, a little generous |
| Unclear, no starter named | between them; lean no unless "if out" clears the bar |

The hold's starter is the man with the most starts this season, with ties going
to the earlier first start. So the middle rows are common: a fill-in with more
starts reads as the starter, and the man back in front of him as his backup. The
model then treats the returning man's wins as backup wins and nudges the team up,
so a pick right at the bar is thinner than it looks. Check the other side's
quarterback too. If its starter is out and the model doesn't know it, the pick
is better than priced.

## 3. Present, then ask

Show one compact table: game and kickoff, edge as priced, edge if out, and what's
going on in a sentence or two. Follow it with a `Sources:` list of the links you
used.

Then call `AskUserQuestion` with one question per game, up to four per call.
Offer "Yes, place it" and "No, skip it", with your recommendation first and
marked. Each yes description says what it does: it places only if the pick still
clears the rule at the current price.

## 4. Place the yeses

```bash
make bet ARGS="--only <game_id> --only <game_id> --yes" > run.txt 2>&1
sed -n '/^Kalshi:/,$p' run.txt | grep -vE '^\s+(passing|QB |\[\?\]|-- Recap)'
```

Use one run for all of them. It re-prices from scratch and places only what
still clears the league's rule in `autobet.json`, with immediate-or-cancel orders
at the limit, so it needs no dry run first. Report each fill (`N/N filled at P`),
any rungs left resting, and anything skipped with its reason. A price that
slipped under the bar since the hold is common: say so, and don't rerun unless
asked.

## Running it in the devcontainer

- **Kalshi key.** Export `KALSHI_KEY_ID=$(cat /workspaces/cassandra/.kalshi/key_id)`
  and `KALSHI_KEY_FILE=/workspaces/cassandra/.kalshi/private_key.pem`. Never print
  the key.
- **No `uv` on PATH.** The make targets are `uv run`. Run the same commands with
  a current venv's Python, with `PYTHONPATH` set to the checkout so its code is
  what runs: `env -u VIRTUAL_ENV <venv>/bin/python -m cassandra.picks_job status`,
  and `... picks.py --place --only <id> --yes` for `make bet`.
- **AWS.** Both commands read s3. On `TokenRetrievalError`, use the `aws-login`
  skill.
- **Long runs.** A `make bet` run takes a few minutes. Run it in the foreground
  with a long timeout rather than polling.

## Pricing by hand

`picks.py --only <id>`, without `--place`, prices the game without touching the
account. A pick with QB news or doubt shows an `if <team>'s starter is out` line
under its order. `--qb-out TEAM` prices the whole run with that starter out.

## Don'ts

- Don't place without Nathan's yes in this session. A hold email is not a yes.
- Don't write stakes, bars or Kelly fractions into anything. They live only in
  `autobet.json`.
