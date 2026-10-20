import io
import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from cassandra import picks_job
from cassandra.picks_job import Bedtime

NOW = datetime(2026, 10, 21, 3, 30, tzinfo=UTC)  # Tue 22:30 CT


class FakeS3:
    """The four calls the job makes, with the two conditions it relies on."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, datetime, str]] = {}
        self.version = 0

    def _missing(self, op: str) -> ClientError:
        return ClientError({"Error": {"Code": "NoSuchKey"}}, op)

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **_):
        held = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and held is not None:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if IfMatch is not None and (held is None or held[2] != IfMatch):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.version += 1
        self.objects[(Bucket, Key)] = (Body, NOW, f'"{self.version}"')

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self._missing("GetObject")
        body, _, etag = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"LastModified": self.objects[(Bucket, Key)][1]}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def _picks(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "bet": bet,
                "game": f"{bet} @ Somebody",
                "edge": edge,
                "kickoff": pd.Timestamp("2026-10-27 18:00", tz="America/Chicago"),
                "buy": "BUY YES KXNCAAFGAME-X",
                "contracts": 10,
                "limit": 0.55,
                "outlay": 5.6,
                "model": 0.62,
                "market": 0.55,
                "offered_within_1c": 30,
                "hours_since_entry": 1.0,
                "no_plays": "",
            }
            for game_id, bet, edge in rows
        ]
    )


def test_one_lock_between_two_runs():
    s3 = FakeS3()
    assert picks_job.acquire(s3, "tmp", NOW, "a")
    assert not picks_job.acquire(s3, "tmp", NOW, "b")
    picks_job.release(s3, "tmp")
    assert picks_job.acquire(s3, "tmp", NOW, "b")


def test_a_dead_runs_lock_is_taken():
    s3 = FakeS3()
    assert picks_job.acquire(s3, "tmp", NOW, "a")
    later = NOW + picks_job.LOCK_STALE + timedelta(minutes=1)
    assert picks_job.acquire(s3, "tmp", later, "b")


def test_only_unemailed_picks_or_a_flipped_side_are_new():
    sent = picks_job.record({}, _picks(("1", "Toledo Rockets", 0.07)), NOW, "s")
    now = _picks(
        ("1", "Toledo Rockets", 0.08),
        ("2", "Ohio Bobcats", 0.06),
    )
    assert list(picks_job.unsent(now, sent)["game_id"]) == ["2"]
    flipped = _picks(("1", "Ball State Cardinals", 0.06))
    assert list(picks_job.unsent(flipped, sent)["game_id"]) == ["1"]


def test_record_keeps_each_emails_numbered_list():
    sent = picks_job.record(
        {}, _picks(("2", "Ohio Bobcats", 0.06), ("1", "Toledo Rockets", 0.09)), NOW, "s"
    )
    # Same kickoff, so numbered by edge, biggest first -- as picks.py numbers.
    assert [p["game_id"] for p in sent["emails"][-1]["picks"]] == ["1", "2"]
    assert json.loads(json.dumps(sent, default=str))


def test_committed_counts_fills_at_price_and_rests_at_the_limit():
    placed = pd.DataFrame(
        [
            {"filled": 4, "fill_price": 0.5, "fee": 0.07, "resting": 6, "limit": 0.55},
            {"filled": 0, "fill_price": None, "fee": None, "resting": 0, "limit": 0.4},
        ]
    )
    rest_fee = picks_job.order_fee(0.55, 6)
    assert picks_job.committed(placed) == pytest.approx(2.07 + 3.3 + rest_fee)
    assert picks_job.committed(None) == 0.0


def test_a_charge_lands_on_the_bedtime_thats_there_now():
    s3 = FakeS3()
    night = Bedtime(NOW + timedelta(hours=9), 40.0)
    picks_job.write_json(s3, "b", picks_job.BEDTIME_KEY, night.to_json())
    charged = picks_job.charge(s3, "b", 12.5)
    assert charged is not None and charged.remaining == pytest.approx(27.5)
    s3.delete_object("b", picks_job.BEDTIME_KEY)  # make awake, mid-run
    assert picks_job.charge(s3, "b", 5.0) is None


def test_orders_rest_until_the_wake_up_time_but_not_for_nothing():
    night = Bedtime(NOW + timedelta(hours=9), 40.0)
    assert night.asleep(NOW)
    assert night.rest_hours(NOW) == pytest.approx(9)
    assert night.rest_hours(NOW + timedelta(hours=8, minutes=50)) == pytest.approx(0.5)
    assert not night.asleep(NOW + timedelta(hours=9))


def test_a_wake_time_is_the_next_one_in_central():
    assert picks_job.wake_time("08:00", NOW) == datetime(2026, 10, 21, 13, tzinfo=UTC)
    assert picks_job.wake_time("23:00", NOW) == datetime(2026, 10, 21, 4, tzinfo=UTC)
    assert picks_job.wake_time("2026-10-21T09:15", NOW) == datetime(
        2026, 10, 21, 14, 15, tzinfo=UTC
    )


def test_the_email_numbers_picks_and_says_how_to_place_each():
    picks = _picks(("1", "Toledo Rockets", 0.09), ("2", "Ohio Bobcats", 0.06))
    sent = picks_job.record({}, picks.iloc[:1], NOW, "s")
    fresh = picks_job.unsent(picks, sent)
    subject = picks_job.subject(picks, fresh, placed=False, at=NOW)
    assert subject.startswith("1 new Kalshi pick (")
    assert subject.endswith("Ohio Bobcats")
    text = picks_job.body(picks, fresh, "REPORT", None, None, 0.01)
    assert "1.     Toledo Rockets" in text and "2. NEW Ohio Bobcats" in text
    assert 'make bet ARGS="--only 2"' in text
    assert 'make bet ARGS="--only 1 --only 2"' in text
    assert text.endswith("REPORT")
