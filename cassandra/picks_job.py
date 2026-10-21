"""What `jobs.py picks` keeps between runs: what it has emailed, bedtime, a lock.

The job runs `picks.py` on a schedule and after every ncaafb finals sweep.
Awake, it emails the picks it hasn't emailed before, each with the line
that places it. At bedtime -- a file in the bucket saying until when and
for how much -- it places them itself, as orders resting until the wake-up
time, and emails what it did.

Three objects in s3, each small JSON or TOML:

- `cassandra/betting/sent.json` (batch bucket): every pick emailed, by game
  id, and each email's numbered list, so "place 1 and 3 from the 22:40
  email" names games rather than positions in a list that has moved since.
- `cassandra/betting/bedtime.json` (batch bucket): `until`, `budget`, and
  `committed`, what the night's orders could cost at worst so far. Written
  by `make bedtime` and `make awake` on a laptop, charged by the job.
- `cassandra/picks.lock` (temp bucket): held for a run's length, so an
  hourly run and a finals-triggered one never price -- or place -- at once.
  Written with `If-None-Match: *`, so only one of two racing runs gets it.
  One older than `LOCK_STALE` is a run that died holding it, and is taken.

The overrides file (`cassandra.overrides`) lives at
`cassandra/betting/overrides.toml` for the job; `make push-overrides`
copies a laptop's up.

A bedtime run that places also copies its picks and orders files to
`cassandra/betting/picks/` and `orders/` (`keep`). They say what the
model gave each bet, and `make ledger` reads them; the container's copies
go when it does.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import botocore.session
import pandas as pd
from botocore.exceptions import ClientError

from cassandra.constants import CASSANDRA_HOME
from cassandra.picks import order_fee

SENT_KEY = "cassandra/betting/sent.json"
BEDTIME_KEY = "cassandra/betting/bedtime.json"
OVERRIDES_KEY = "cassandra/betting/overrides.toml"
#: Where a run writes every game it priced (`picks.py`) and the orders it
#: sent (`placing.place`), by the s3 prefix `keep` copies them to.
RUN_FILES = {
    "cassandra/betting/picks/": CASSANDRA_HOME / "betting" / "picks",
    "cassandra/betting/orders/": CASSANDRA_HOME / "betting" / "orders",
}
LOCK_KEY = "cassandra/picks.lock"
#: A run takes about five minutes; one holding the lock this long has died.
LOCK_STALE = timedelta(minutes=30)
#: An order rests at least this long, however close the wake-up time is.
MIN_REST = timedelta(minutes=30)
#: Emails kept in `sent.json`: a week of hourly runs and then some.
EMAILS_KEPT = 200


def client(service: str) -> Any:
    """A plain synchronous botocore client: what boto3 would give, without it."""
    return botocore.session.get_session().create_client(service)


def _code(error: ClientError) -> str:
    return error.response.get("Error", {}).get("Code", "")


def read_json(s3: Any, bucket: str, key: str) -> tuple[dict | None, str | None]:
    """An object's JSON and its ETag, or (None, None) when there isn't one."""
    try:
        found = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if _code(error) in ("NoSuchKey", "404"):
            return None, None
        raise
    return json.loads(found["Body"].read()), found["ETag"]


def write_json(s3: Any, bucket: str, key: str, value: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=1, default=str).encode(),
        ContentType="application/json",
    )


# ------------------------------------------------------------------------------
# The lock
# ------------------------------------------------------------------------------


def acquire(s3: Any, bucket: str, now: datetime, owner: str) -> bool:
    """Take the run lock, or say another run has it."""
    for _ in range(2):
        try:
            s3.put_object(
                Bucket=bucket,
                Key=LOCK_KEY,
                Body=json.dumps({"owner": owner, "at": now.isoformat()}).encode(),
                IfNoneMatch="*",
            )
            return True
        except ClientError as error:
            if _code(error) not in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise
        try:
            held = s3.head_object(Bucket=bucket, Key=LOCK_KEY)
        except ClientError as error:
            if _code(error) in ("NoSuchKey", "404"):
                continue
            raise
        if now - held["LastModified"] < LOCK_STALE:
            return False
        s3.delete_object(Bucket=bucket, Key=LOCK_KEY)
    return False


def release(s3: Any, bucket: str) -> None:
    s3.delete_object(Bucket=bucket, Key=LOCK_KEY)


def keep(s3: Any, bucket: str, since: datetime) -> list[str]:
    """Copy the picks and orders files written since `since` -- this run's --
    up to s3, for `make ledger`. The keys they went to."""
    kept = []
    for prefix, directory in RUN_FILES.items():
        for path in sorted(directory.glob("*.csv")):
            if datetime.fromtimestamp(path.stat().st_mtime, UTC) < since:
                continue
            s3.put_object(Bucket=bucket, Key=prefix + path.name, Body=path.read_bytes())
            kept.append(prefix + path.name)
    return kept


# ------------------------------------------------------------------------------
# Bedtime
# ------------------------------------------------------------------------------


@dataclass
class Bedtime:
    """Place the picks unattended until `until`, for at most `budget` dollars."""

    until: datetime
    budget: float
    committed: float = 0.0
    max_bet: float = 10.0
    kelly_fraction: float = 0.25

    @classmethod
    def from_json(cls, value: dict) -> "Bedtime":
        return cls(
            until=datetime.fromisoformat(value["until"]),
            budget=float(value["budget"]),
            committed=float(value.get("committed", 0.0)),
            max_bet=float(value.get("max_bet", 10.0)),
            kelly_fraction=float(value.get("kelly_fraction", 0.25)),
        )

    def to_json(self) -> dict:
        return {
            "until": self.until.isoformat(),
            "budget": self.budget,
            "committed": round(self.committed, 2),
            "max_bet": self.max_bet,
            "kelly_fraction": self.kelly_fraction,
        }

    def asleep(self, now: datetime) -> bool:
        return now < self.until

    @property
    def remaining(self) -> float:
        return max(self.budget - self.committed, 0.0)

    def rest_hours(self, now: datetime) -> float:
        """How long tonight's orders rest: until the wake-up time."""
        return max(self.until - now, MIN_REST).total_seconds() / 3600


def bedtime(s3: Any, bucket: str) -> Bedtime | None:
    found, _ = read_json(s3, bucket, BEDTIME_KEY)
    return None if found is None else Bedtime.from_json(found)


def charge(s3: Any, bucket: str, amount: float) -> Bedtime | None:
    """Add a run's worst-case cost to the night's, unless bedtime has changed.

    Conditional on the ETag read, so a `make awake` or a new `make bedtime`
    while this run was placing isn't overwritten by the copy it started
    with: the charge is re-applied to whatever is there now, and dropped if
    bedtime was cleared.
    """
    for _ in range(3):
        found, etag = read_json(s3, bucket, BEDTIME_KEY)
        if found is None:
            return None
        night = Bedtime.from_json(found)
        night.committed += amount
        try:
            s3.put_object(
                Bucket=bucket,
                Key=BEDTIME_KEY,
                Body=json.dumps(night.to_json(), indent=1).encode(),
                ContentType="application/json",
                IfMatch=etag,
            )
            return night
        except ClientError as error:
            if _code(error) not in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise
    raise RuntimeError("bedtime.json kept changing under the charge")


def committed(placed: pd.DataFrame | None) -> float:
    """What a run's orders cost at worst: fills at their price, rests at the limit."""
    if placed is None or placed.empty or "filled" not in placed:
        return 0.0
    total = 0.0
    for _, row in placed.iterrows():
        filled = 0.0 if pd.isna(row.get("filled")) else float(row["filled"])
        resting = 0.0 if pd.isna(row.get("resting")) else float(row["resting"])
        if filled and not pd.isna(row.get("fill_price")):
            fee = 0.0 if pd.isna(row.get("fee")) else float(row["fee"])
            total += filled * float(row["fill_price"]) + fee
        if resting:
            total += resting * float(row["limit"]) + order_fee(
                float(row["limit"]), int(resting)
            )
    return total


# ------------------------------------------------------------------------------
# What's been emailed
# ------------------------------------------------------------------------------


def unsent(picks: pd.DataFrame, sent: dict) -> pd.DataFrame:
    """The picks no email has carried: a new game, or the same game's other side."""
    games = sent.get("games", {})
    fresh = [
        game_id not in games or games[game_id].get("bet") != bet
        for game_id, bet in zip(picks["game_id"], picks["bet"])
    ]
    return picks[fresh]


def record(sent: dict, picks: pd.DataFrame, at: datetime, subject: str) -> dict:
    """`sent` with this email's picks in it, numbered as the email numbers them."""
    games = dict(sent.get("games", {}))
    listed = []
    for _, row in ordered(picks).iterrows():
        games[row["game_id"]] = {
            "bet": row["bet"],
            "game": row["game"],
            "edge": round(float(row["edge"]), 4),
            "at": at.isoformat(),
        }
        listed.append({"game_id": row["game_id"], "bet": row["bet"]})
    emails = [
        *sent.get("emails", []),
        {"at": at.isoformat(), "subject": subject, "picks": listed},
    ]
    return {"games": games, "emails": emails[-EMAILS_KEPT:]}


def ordered(picks: pd.DataFrame) -> pd.DataFrame:
    """The order `picks.py` numbers its orders in."""
    return picks.sort_values(["kickoff", "edge"], ascending=[True, False])


def only_args(game_ids: list[str]) -> str:
    return " ".join(f"--only {game_id}" for game_id in game_ids)


def subject(
    picks: pd.DataFrame, fresh: pd.DataFrame, placed: bool, at: datetime
) -> str:
    """One line a phone shows whole."""
    when = f"{at:%a %H:%M} CT"
    if placed:
        return f"Placed {len(picks)} Kalshi picks overnight ({when})"
    names = ", ".join(ordered(fresh)["bet"])
    head = f"{len(fresh)} new Kalshi pick{'s' if len(fresh) != 1 else ''} ({when}): "
    return (head + names)[:99]


def body(
    picks: pd.DataFrame,
    fresh: pd.DataFrame,
    report: str,
    placed: pd.DataFrame | None,
    night: Bedtime | None,
    slippage: float,
) -> str:
    """The email: the picks numbered as `picks.py` numbers them, how to place
    each, then the run's whole printout."""
    offered = f"offered_within_{round(slippage * 100)}c"
    new = set(fresh["game_id"])
    lines = []
    if placed is not None:
        lines.append(
            "Bedtime: placed these as resting orders"
            + (
                f" until {night.until:%a %H:%M}; ${night.committed:,.2f} of "
                f"${night.budget:,.2f} committed tonight."
                if night is not None
                else "."
            )
        )
        lines.append("Run `make awake` to stop, or `make positions` to see fills.\n")
    for number, (_, row) in enumerate(ordered(picks).iterrows(), start=1):
        mark = "NEW " if row["game_id"] in new else "    "
        lines.append(
            f"{number}. {mark}{row['bet']} -- {row['game']}, {row['kickoff']:%a %H:%M} CT"
        )
        lines.append(
            f"   {row['buy']}: {row['contracts']:.0f} @ <= {row['limit']:.2f}, "
            f"at most ${row['outlay']:.2f}"
        )
        since = row.get("hours_since_entry")
        lines.append(
            f"   model {row['model']:.3f} vs market {row['market']:.3f} "
            f"(edge {row['edge']:+.3f}); {row.get(offered, 0):.0f} offered within "
            f"{round(slippage * 100)}c"
            + ("" if since is None or pd.isna(since) else f"; {since:.0f}h since entry")
        )
        if row.get("no_plays"):
            lines.append(f"   no EPA yet for {row['no_plays']}'s last game")
        if placed is None:
            lines.append(f'   place: make bet ARGS="--only {row["game_id"]}"')
        lines.append("")
    if placed is None and len(picks) > 1:
        lines.append(
            f'All of them: make bet ARGS="{only_args(list(ordered(picks)["game_id"]))}"'
        )
        lines.append(
            'Or reply in a Claude session, e.g. "place 1 and 3 from the '
            'picks email" -- it reads the list from s3 and re-prices before buying.'
        )
        lines.append("")
    lines.append("-" * 72)
    lines.append(report)
    return "\n".join(lines)


def now_utc() -> datetime:
    return datetime.now(UTC)


def wake_time(text: str, now: datetime) -> datetime:
    """`08:00` is the next 8am Central; anything else is an ISO timestamp."""
    from cassandra.placing import CENTRAL

    if len(text) <= 5 and ":" in text:
        hour, minute = (int(part) for part in text.split(":"))
        local = now.astimezone(CENTRAL)
        wake = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if wake <= local:
            wake += timedelta(days=1)
        return wake.astimezone(UTC)
    found = datetime.fromisoformat(text)
    return (found if found.tzinfo else found.replace(tzinfo=CENTRAL)).astimezone(UTC)


def main(argv: list[str] | None = None) -> None:
    """`make bedtime` / `make awake` / `make push-overrides` / `make picks-status`."""
    import argparse

    from endgame_aws import Config

    from cassandra.placing import CENTRAL

    parser = argparse.ArgumentParser(prog="python -m cassandra.picks_job")
    commands = parser.add_subparsers(dest="command", required=True)
    bed = commands.add_parser("bedtime", help="place the picks unattended until a time")
    bed.add_argument(
        "--until", required=True, help="e.g. 08:00 (Central) or an ISO time"
    )
    bed.add_argument(
        "--budget", type=float, required=True, help="most the night commits"
    )
    bed.add_argument("--max-bet", type=float, default=10.0)
    bed.add_argument("--kelly-fraction", type=float, default=0.25)
    commands.add_parser("awake", help="stop placing; email the picks again")
    commands.add_parser("push-overrides", help="copy the overrides file up for the job")
    commands.add_parser("status", help="bedtime, and the last emails' picks")
    args = parser.parse_args(argv)

    s3 = client("s3")
    bucket = Config.init_from_file().bucket
    now = now_utc()
    if args.command == "bedtime":
        night = Bedtime(
            wake_time(args.until, now),
            args.budget,
            0.0,
            args.max_bet,
            args.kelly_fraction,
        )
        write_json(s3, bucket, BEDTIME_KEY, night.to_json())
        print(
            f"bedtime until {night.until.astimezone(CENTRAL):%a %H:%M} CT: up to "
            f"${night.budget:,.2f}, ${night.max_bet:g} a bet, {night.kelly_fraction:g} Kelly"
        )
    elif args.command == "awake":
        found = bedtime(s3, bucket)
        s3.delete_object(Bucket=bucket, Key=BEDTIME_KEY)
        print(
            "awake: the next run emails"
            + (f" (the night committed ${found.committed:,.2f})" if found else "")
        )
    elif args.command == "push-overrides":
        from cassandra.constants import CASSANDRA_HOME

        local = CASSANDRA_HOME / "betting" / "overrides.toml"
        s3.put_object(Bucket=bucket, Key=OVERRIDES_KEY, Body=local.read_bytes())
        print(f"{local} -> s3://{bucket}/{OVERRIDES_KEY}")
    else:
        found = bedtime(s3, bucket)
        if found is None or not found.asleep(now):
            print("awake: runs email new picks")
        else:
            print(
                f"bedtime until {found.until.astimezone(CENTRAL):%a %H:%M} CT, "
                f"${found.committed:,.2f} of ${found.budget:,.2f} committed"
            )
        sent, _ = read_json(s3, bucket, SENT_KEY)
        for email in (sent or {}).get("emails", [])[-5:]:
            at = datetime.fromisoformat(email["at"]).astimezone(CENTRAL)
            print(f"\n{at:%a %H:%M} CT  {email['subject']}")
            for number, pick in enumerate(email["picks"], start=1):
                print(f"  {number}. {pick['bet']}  --only {pick['game_id']}")


if __name__ == "__main__":
    main()
