import base64

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .kalshi_trade import (
    Fill,
    KalshiClient,
    KalshiError,
    bought,
    event_of,
    fill,
    is_rung,
    order_body,
    order_price,
    rung_id,
    sign,
    transfer_body,
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


@pytest.mark.parametrize(
    "order, side",
    [
        ({"outcome_side": "yes", "book_side": "bid"}, "yes"),
        ({"outcome_side": "yes", "book_side": "ask"}, "no"),
        ({"outcome_side": "no", "book_side": "bid"}, "no"),
        ({"side": "yes", "action": "buy"}, "yes"),
        ({"side": "no", "action": "sell"}, "yes"),
    ],
)
def test_bought_reads_either_way_of_naming_a_side(order, side):
    assert bought(order) == side


def test_a_rung_is_told_by_its_client_order_id_and_priced_from_its_side():
    order = {
        "client_order_id": rung_id(),
        "outcome_side": "yes",
        "book_side": "ask",
        "yes_price_dollars": "0.6200",
    }
    assert is_rung(order) and not is_rung({"client_order_id": "abc"})
    assert order_price(order) == pytest.approx(0.38)


def test_a_transfer_stays_on_the_event_contracts_exchange():
    body = transfer_body(0, 3, 12_500)
    assert body.pop("client_transfer_id")
    assert body == {
        "source": "event_contract",
        "destination": "event_contract",
        "amount": 12_500,
        "source_exchange_shard": 0,
        "destination_exchange_shard": 3,
    }
    # A new id each time: Kalshi treats a reused one as the same transfer.
    assert (
        transfer_body(0, 3, 1)["client_transfer_id"]
        != transfer_body(0, 3, 1)["client_transfer_id"]
    )


def test_shard_balances_are_read_in_centicents(monkeypatch):
    client = KalshiClient("id", Ed25519PrivateKey.generate())
    breakdown = [
        {"balance": "0.5556", "exchange_index": 0},
        {"balance": "12.0000", "exchange_index": 3},
    ]
    monkeypatch.setattr(client, "_request", lambda *_: {"balance_breakdown": breakdown})
    assert client.shard_balances() == {0: 5556, 3: 120_000}
    monkeypatch.setattr(client, "_request", lambda *_: {"balance": 100})
    assert client.shard_balances() == {}


def test_a_dry_runs_transfer_is_printed_not_sent(monkeypatch, capsys):
    client = KalshiClient("id", Ed25519PrivateKey.generate(), dry_run=True)

    def boom(*_):
        raise AssertionError("sent")

    monkeypatch.setattr(client, "_request", boom)
    assert client.transfer(0, 3, 500) == "dry-run"
    assert "dry run, not sent" in capsys.readouterr().out
