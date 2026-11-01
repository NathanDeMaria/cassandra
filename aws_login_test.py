from datetime import datetime, timedelta

import aws_login

NOW = datetime(2026, 10, 24, 22, 0, tzinfo=aws_login.CENTRAL)


def test_a_code_that_ran_out_is_told_from_a_failed_login():
    assert aws_login.expired(
        [
            "aws: [ERROR]: An error occurred (InvalidGrantException) when calling "
            "the CreateToken operation:",
            "error_description: Invalid device code provided",
        ]
    )
    assert not aws_login.expired(["Unable to locate credentials"])


def test_until_counts_down_the_session_this_tool_started(tmp_path, monkeypatch):
    monkeypatch.setattr(aws_login, "STATE", tmp_path)
    assert aws_login.until(NOW) == ""
    (tmp_path / "logged_in_at").write_text((NOW - timedelta(hours=2)).isoformat())
    assert aws_login.until(NOW) == (
        "; logged in Sat 20:00 CT, so the session lasts until about Sun 04:00 CT "
        "(6.0 h left)"
    )
    # A session that's over says nothing about the credentials working now.
    assert aws_login.until(NOW + timedelta(hours=7)) == ""
