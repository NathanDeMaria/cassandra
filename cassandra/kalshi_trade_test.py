import base64

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .kalshi_trade import (
    Fill,
    KalshiClient,
    KalshiError,
    event_of,
    fill,
    order_body,
    sign,
)


def test_rsa_signature_covers_timestamp_method_and_path_without_query():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signature = sign(
        key, "1700000000000", "GET", "/trade-api/v2/portfolio/orders?status=resting"
    )
    key.public_key().verify(
        base64.b64decode(signature),
        b"1700000000000GET/trade-api/v2/portfolio/orders",
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )


def test_ed25519_signature():
    key = Ed25519PrivateKey.generate()
    signature = sign(key, "1", "POST", "/trade-api/v2/portfolio/events/orders")
    key.public_key().verify(
        base64.b64decode(signature), b"1POST/trade-api/v2/portfolio/events/orders"
    )


def test_buying_yes_is_a_bid_at_its_price():
    body = order_body(
        "KXNCAAFGAME-26OCT17PITTBC-PITT", "yes", 7, 0.62, client_order_id="x"
    )
    assert body == {
        "ticker": "KXNCAAFGAME-26OCT17PITTBC-PITT",
        "client_order_id": "x",
        "side": "bid",
        "count": "7",
        "price": "0.62",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "cancel_order_on_pause": True,
    }


def test_buying_no_is_an_ask_on_yes_at_one_minus_the_price():
    body = order_body("T-BC", "no", 3, 0.38, expires_at=1_800_000_000)
    assert body["side"] == "ask"
    assert body["price"] == "0.62"
    assert body["time_in_force"] == "good_till_canceled"
    assert body["expiration_time"] == 1_800_000_000


@pytest.mark.parametrize(
    ("action", "count", "limit"), [("buy", 1, 0.5), ("yes", 0, 0.5), ("no", 1, 1.0)]
)
def test_bad_orders_are_refused(action, count, limit):
    with pytest.raises(ValueError):
        order_body("T", action, count, limit)


def test_fill_reads_a_no_at_its_own_price():
    response = {
        "order_id": "o",
        "fill_count": "4.00",
        "remaining_count": "0.00",
        "average_fill_price": "0.6100",
        "average_fee_paid": "0.0200",
    }
    done = fill(response, "no")
    assert done.filled == 4
    assert done.price == pytest.approx(0.39)
    assert done.fee == pytest.approx(0.08)
    assert (
        fill(
            {"order_id": "o", "fill_count": "0.00", "remaining_count": "0.00"}, "yes"
        ).price
        is None
    )


def test_a_dry_run_sends_nothing_but_reads(monkeypatch):
    client = KalshiClient("id", Ed25519PrivateKey.generate(), dry_run=True)

    def no_network(*args, **kwargs):
        raise AssertionError("a dry run reached the network")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    assert client.buy("T-A", "yes", 3, 0.4) == Fill("dry-run", 0.0, 0.0, None, None)
    with pytest.raises(KalshiError, match="refusing"):
        client._request("POST", "/portfolio/events/orders", {})
    with pytest.raises(KalshiError, match="refusing"):
        client._request("DELETE", "/portfolio/orders")
    with pytest.raises(AssertionError, match="network"):
        client._request("GET", "/portfolio/balance")


def test_event_of():
    assert event_of("KXNCAAFGAME-26OCT17BRWNPRIN-PRIN") == "KXNCAAFGAME-26OCT17BRWNPRIN"
