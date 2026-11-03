---
name: aws-login
description: Log Nathan in to AWS from his phone when the SSO token has expired, and check how long a login has left. Use when an AWS call fails with TokenRetrievalError, "Token has expired and refresh failed", ExpiredToken or "Unable to locate credentials"; when asked to log in; or before a long job that needs AWS the whole way.
---

# Logging in to AWS

The devcontainer has no browser, so a login is a link Nathan approves on his phone.
`aws_login.py` (`make login`) runs the device-code flow. It prints each link as
`approve: <link>` and starts a new code each time one expires unapproved. Codes last
about ten minutes. It needs only `python3` and the AWS CLI, so it works from any
worktree without a virtualenv.

## When a call fails on an expired token

1. Start it in the background, with a timeout longer than its three ten-minute codes:

   ```bash
   python3 aws_login.py      # Bash, run_in_background: true, timeout 2400000
   ```

2. Read the task's output file after a few seconds, and **put the `approve:` link in
   your chat reply right away** -- the next thing you write, before any other tool
   call. Nathan sees only your messages: a link in a log, a tool result or a tool
   description never reaches him, and a login nobody can see is a login nobody
   approves. A bare URL on its own line, not markdown link text, so it taps through
   on a phone. Only then go on with work that doesn't need AWS.
3. Don't sit in a blocking wait on the login (an `until` loop, `sleep`): he can't
   see the link while you wait, and can't interrupt you to ask for it. Send the link
   and end the turn, or keep working; the task notification says when it's done.
4. If he says the link is dead, or comes back after ten minutes, read the output file
   again. It holds a newer `approve:` line once the old code has expired. Send that
   one in chat the same way -- every new code, as soon as it appears.
5. The task notification says how it ended:
   - `logged in` (exit 0): carry on.
   - `3 codes expired unapproved` (exit 1): don't start another round while he's
     away. Say so and wait for him to ask ("log me in").

A second `python3 aws_login.py` while one is waiting doesn't start another login. It
prints the waiting one's links and waits with it. So a stale code is never the one
on offer.

## Before a long job

`python3 aws_login.py status` prints who's logged in and, when this tool did the
login, about when the session ends ("6.0 h left"). If the job will outlast that, run
`python3 aws_login.py --fresh` first. A new login starts a new session.

## Don'ts

- Don't `pkill -f "aws sso login"`. The pattern matches the shell running the
  command, so it kills that shell (exit 144). The tool stops its own login when it's
  stopped.
- Don't run `aws sso login` without `--no-browser`. It waits for a browser that never
  opens. The tool always passes it.
- Don't print, copy or move anything under `~/.aws/sso/cache/`. Those files hold the
  tokens.

## Why logins keep running out

IAM Identity Center's session duration is 8 hours by default. The CLI renews the
hour-long access token under it until the session ends, then
`Token has expired and refresh failed`. A morning login is gone by the night games.
The lasting fix is Nathan's to make, in the IAM Identity Center console:
Settings → Authentication → Session duration. If he lengthens it, set
`AWS_SESSION_HOURS` so `status` counts down from the right length.
