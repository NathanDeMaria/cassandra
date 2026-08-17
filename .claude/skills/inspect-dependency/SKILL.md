---
name: inspect-dependency
description: Look at the source of an installed dependency (call_it_what_you_want, endgame, endgame_aws) that has no local checkout in this repo. Use when a question needs to know what a class/function in one of those packages actually does or exposes, rather than guessing from memory or docs.
---

# Inspecting an installed dependency

`call_it_what_you_want` and `endgame` are git dependencies installed into the poetry env as
opaque site-packages — there's no copy of their repos here to `grep`. Guessing their API from
memory or from what cassandra happens to import is how stale assumptions creep in. Import and
read the real source instead:

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
