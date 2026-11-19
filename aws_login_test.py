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
    monkeypatch.setattr(aws_login, "SESSION", timedelta(hours=8))
    assert aws_login.until(NOW) == ""
    (tmp_path / "logged_in_at").write_text((NOW - timedelta(hours=2)).isoformat())
    assert aws_login.until(NOW) == (
        "; logged in Sat Oct 24 20:00 CT, so the session lasts until about "
        "Sun Oct 25 04:00 CT (6.0 h left)"
    )
    # A session that's over says nothing about the credentials working now.
    assert aws_login.until(NOW + timedelta(hours=7)) == ""


def test_the_link_to_approve_is_picked_out_of_either_cli_prompt():
    # Before 2.39: a bare page, then the same page with the code filled in.
    assert not aws_login.is_link("https://device.sso.us-east-2.amazonaws.com/")
    assert aws_login.is_link(
        "https://d-9a675ddd0f.awsapps.com/start/#/device?user_code=ABCD-EFGH"
    )
    # 2.39 on: one signin.aws link carrying the request, and no user_code.
    assert aws_login.is_link(
        "https://us-east-2.signin.aws/platform/authorize?request_uri=urn%3Aietf"
        "%3Aparams%3Aoauth%3Arequest_uri%3AAAAA&identitystore_id=d-9a675ddd0f"
    )
    assert not aws_login.is_link("Then enter the code:")
    assert not aws_login.is_link("ABCD-EFGH")
