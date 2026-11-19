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

## Log in first, not last

The moment you see AWS is out -- a `no s3 (TokenRetrievalError)` line, a failed call,
or a task you know will write to s3 -- start the login, before the analysis or
whatever else the turn holds. A login found in the first minute and mentioned at the
end of a long answer is ten minutes of code gone, and the link lands under
everything else. If the task needs AWS later, `python3 aws_login.py status` at the
start of the turn says whether to log in now.

## When a call fails on an expired token

1. Start it in the background, with a timeout longer than its three ten-minute codes:

   ```bash
   python3 aws_login.py      # Bash, run_in_background: true, timeout 2400000
   ```

2. Read the task's output file after a few seconds. Then, before any other tool call:
   - **Push.** Send a `PushNotification` (load it with ToolSearch
     `select:PushNotification` if it isn't loaded), e.g. `AWS login waiting: approve
     the link at the bottom of the chat (code good until 1:42 PM CT)`. The link
     itself doesn't go in it: a push is capped near 200 characters and the link is
     ~500. Tapping the push opens the session, where the link is waiting. The tool
     skips the push when he's at the terminal; that's fine.
   - **End your reply with the link and end the turn.** One short line saying what
     the login is for, then the link as a bare URL on its own line, not markdown link
     text, so it taps through on a phone. Nothing after it. Nathan sees only your
     messages -- a link in a log, a tool result or a tool description never reaches
     him -- and anything you write after the link pushes it out of sight.
3. Don't wait on the login (an `until` loop, `sleep`) or carry on with other work in
   the same turn: either hides the link. The task notification wakes you when it ends.
4. If he says the link is dead, or comes back after ten minutes, read the output file
   again. It holds a newer `approve:` line once the old code has expired. Send that
   one the same way, push and all -- every new code, as soon as it appears.
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

The CLI renews the hour-long access token until IAM Identity Center's session
duration runs out, then `Token has expired and refresh failed`. That duration was the
8-hour default, so a morning login was gone by the night games. Nathan set it to
7 days on 2026-11-17, in the IAM Identity Center console (Settings → Authentication →
Session duration); there is no API for it. `aws_login.py` assumes 7 days for
`status`. If the console setting changes, change the default there too, or set
`AWS_SESSION_HOURS`.
