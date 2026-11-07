"""Log in to AWS from a phone: print a link to approve, and keep a live one
coming until it's approved.

The devcontainer has no browser, so the login is the device-code flow:
`aws sso login --use-device-code --no-browser` prints a link, you approve
it on your phone, and the CLI picks the approval up. (Without `--no-browser`
it waits on a browser that never opens.) By hand that's brittle in two ways
this handles:

- A link's code lives about ten minutes. Sent while you're away, it's dead
  by the time you tap it. This starts a new one each time the last one
  expires, up to `--tries`, and prints each as `approve: <link>`.
- Two logins waiting at once means the link you tap may be the stale one.
  A second call while one waits prints that one's links and waits on it.

The session a login starts lasts IAM Identity Center's session duration --
8 hours unless it's been changed -- and the CLI renews the hour-long token
under it until then. `status` says when this tool last logged in, and so
about when that runs out; `--fresh` starts a new session before a long run.

    python3 aws_login.py            # log in if logged out
    python3 aws_login.py --fresh    # log in again anyway: a new session
    python3 aws_login.py status     # logged in? until about when?

Only the standard library and the AWS CLI: it runs from any checkout,
without the project's virtualenv.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

STATE = Path.home() / ".cache" / "aws-login"
CENTRAL = ZoneInfo("America/Chicago")

#: IAM Identity Center's session duration, the default unless
#: `AWS_SESSION_HOURS` says yours is different.
SESSION = timedelta(hours=float(os.environ.get("AWS_SESSION_HOURS", "8")))

LOGIN = ["aws", "sso", "login", "--use-device-code", "--no-browser"]

#: Printed under every link. Whoever runs this is usually an agent whose
#: tool output Nathan never sees: a link left in a log is a login nobody
#: can approve, so the reminder travels with the link.
RELAY = "  ^ send this link to Nathan in the chat now, as a bare URL; he can't see this output"

#: What the CLI says when a code ran out before anyone approved it.
_EXPIRED = ("InvalidGrantException", "ExpiredTokenException", "expired")


def whoami() -> str | None:
    """The caller's ARN, or None when the credentials don't work."""
    try:
        found = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Arn", "--output", "text"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    return found.stdout.strip() if found.returncode == 0 else None


def expired(lines: Sequence[str]) -> bool:
    """Whether a failed login's output says its code ran out unapproved."""
    return any(marker in line for line in lines for marker in _EXPIRED)


def until(now: datetime | None = None) -> str:
    """When this tool last logged in and about when that session ends, or
    "" when it hasn't, or the session it started is over."""
    try:
        at = datetime.fromisoformat((STATE / "logged_in_at").read_text().strip())
    except FileNotFoundError, ValueError:
        return ""
    left = at + SESSION - (now or datetime.now(CENTRAL))
    if left <= timedelta(0):
        return ""
    end = (at + SESSION).astimezone(CENTRAL)
    return (
        f"; logged in {at.astimezone(CENTRAL):%a %H:%M} CT, so the session lasts "
        f"until about {end:%a %H:%M} CT ({left.total_seconds() / 3600:.1f} h left)"
    )


def _owner() -> int | None:
    """The pid of this tool's login that is still waiting, if one is."""
    try:
        pid = int((STATE / "owner").read_text())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError, ValueError:
        return None
    return pid if pid != os.getpid() and b"aws_login" in cmdline else None


def follow(pid: int) -> int:
    """Wait on the login another call started, printing its links as they come."""
    shown = None
    while _owner() == pid:
        try:
            link = (STATE / "link").read_text().strip()
        except FileNotFoundError:
            link = None
        if link and link != shown:
            print(f"approve: {link}\n{RELAY}", flush=True)
            shown = link
        time.sleep(3)
    arn = whoami()
    print(f"logged in{until()}" if arn else "the other login gave up", flush=True)
    return 0 if arn else 1


def login(tries: int) -> int:
    """Offer up to `tries` codes, each for as long as the CLI waits on it."""
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "owner").write_text(str(os.getpid()))
    proc = None
    try:
        for attempt in range(1, tries + 1):
            proc = subprocess.Popen(
                LOGIN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            said = []
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.strip()
                if "user_code=" in line:
                    (STATE / "link").write_text(line)
                    print(f"approve: {line}\n{RELAY}", flush=True)
                elif line:
                    said.append(line)
            proc.wait()
            if proc.returncode == 0 and whoami():
                (STATE / "logged_in_at").write_text(datetime.now(CENTRAL).isoformat())
                print(f"logged in{until()}", flush=True)
                return 0
            if not expired(said):
                print("login failed:\n  " + "\n  ".join(said[-5:]), flush=True)
                return 1
            if attempt < tries:
                print(
                    f"that code expired unapproved; a new one ({attempt + 1} of {tries}):",
                    flush=True,
                )
        print(f"{tries} codes expired unapproved; run it again when you're ready")
        return 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        for name in ("owner", "link"):
            (STATE / name).unlink(missing_ok=True)


def status() -> int:
    arn = whoami()
    if arn is None:
        print("logged out")
        return 1
    print(f"logged in as {arn.rsplit('/', 1)[-1]}{until()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", nargs="?", choices=["login", "status"], default="login"
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=3,
        help="codes to offer before giving up, about ten minutes each",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="log in even when logged in, to start a new session",
    )
    args = parser.parse_args()
    # A background run that's stopped still cleans up its waiting login.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    if args.command == "status":
        return status()
    owner = _owner()
    if owner is not None:
        return follow(owner)
    if not args.fresh and (arn := whoami()):
        print(f"already logged in as {arn.rsplit('/', 1)[-1]}{until()}")
        return 0
    return login(args.tries)


if __name__ == "__main__":
    sys.exit(main())
