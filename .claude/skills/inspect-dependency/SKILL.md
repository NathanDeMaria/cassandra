---
name: inspect-dependency
description: Look at the source of a git dependency (call_it_what_you_want, endgame, endgame_aws) — either the exact revision installed in the venv, or a full local checkout for grepping history and preparing patches. Use when a question needs to know what a class/function in one of those packages actually does or exposes, rather than guessing from memory or docs.
---

# Inspecting a git dependency

There are two ways to read one, and they answer different questions. Reach for the wrong one and
you will read code that isn't running, or patch code that nobody will run.

## Which one you want

**The installed revision** — `inspect_module.py`, below. What cassandra is *actually executing*.
Use it for any question of the form "what does this do right now", "what does it return", "does
this argument exist".

**The checkout** — `~/.cassandra/repos/`, below. The whole repo, with history. Use it to `grep`
across files, read commit messages, or write a patch.

They disagree, and the disagreement is the point: a checkout sits on `main`, and cassandra pins
each dependency to a rev in `pyproject.toml`. `main` can be ahead by a fix that is not installed
here — and a bug you "confirm" against the checkout may already be fixed, or a bug you cannot
reproduce may have been introduced after the pin. Check the drift before trusting either:

```bash
git -C ~/.cassandra/repos/EndGame log --oneline <pinned-rev>..main -- py-endgame/
```

## The checkouts

Clones of the git dependencies, kept outside the repo so they never show up in `git status`:

| package(s) | checkout | note |
|---|---|---|
| `call_it_what_you_want` | `~/.cassandra/repos/call-it-what-you-want` | |
| `endgame`, `endgame_aws` | `~/.cassandra/repos/EndGame` | one repo, two packages: `py-endgame/` and `py-endgame-aws/` |

`endgame` and `endgame_aws` are subdirectories of the *same* repo and are pinned to **different**
revs in `pyproject.toml`. So "the EndGame checkout" is not one version of anything — check which
subdirectory and which rev you mean.

Clone one that's missing with `git clone <url from pyproject.toml> ~/.cassandra/repos/<name>`.

## The installed revision

Import and read the real source:

```bash
python3 .claude/skills/inspect-dependency/inspect_module.py call_it_what_you_want
python3 .claude/skills/inspect-dependency/inspect_module.py call_it_what_you_want.Teams
python3 .claude/skills/inspect-dependency/inspect_module.py endgame.types.NcaaFbGroup
```

Always invoke with the **relative** path shown above (cwd is already the repo root), not an
absolute one — the `.claude/settings.local.json` allow-rule for this script is a literal prefix
match on the command string, and an absolute path won't match it, forcing a manual approval every
time.

Given a module, it lists public names. Given a class or function, it prints the source (docstring
included — these packages are written in the same comment-the-why style as cassandra, so the
docstring is often the fastest way to learn a design constraint). Given something with no
retrievable source (e.g. a C extension or a `namedtuple` field), it falls back to `help()`.

Only imports names under `call_it_what_you_want`, `endgame`, `endgame_aws`, or `cassandra` — the
project's own declared dependencies — and refuses anything else. It runs whatever's on `pip`'s
path, so this is not a sandbox; the allowlist exists to keep it a lookup tool for this project's
own stack, not a general "import and run arbitrary code" script.

Use this before proposing a design that assumes a package does or doesn't already have some piece
of data (e.g. "does the team registry carry conference/division metadata") — check first, propose
second.

## Neither one tells you what the service returns

Both of these read *our* code. A question about what ESPN actually sends back — whether a
parameter is honoured, whether a range still works — is answered by one live request and nothing
else. `ODDS_PAGE_LIMIT = 1000` was a correct measurement on a 14-day NCAABB range and a wrong one
for a single-day FBS request, where ESPN silently falls back to 25 events; the comment explaining
the constant was accurate, current, and misleading. Source explains intent. Only the wire says
what happens.
