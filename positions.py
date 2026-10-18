"""The Kalshi account: cash, and each open position with what it cost and is worth.

    python positions.py           # or `make positions`
    python positions.py --demo    # the demo exchange's account

Read-only: the client runs as a dry run, so it can't send anything but a
GET. A position's value is what selling it now would bring, at its side's
best bid -- a thin book's bid can sit well under its last trade, so this
is the cautious number, not a mark to market. Settled positions (games
over and paid) are in the cash, not here.
"""

import argparse

import pandas as pd

from cassandra.kalshi_trade import KalshiClient
from cassandra.picks import event_day


def _dollars(value: str | float | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", action="store_true", help="the demo exchange")
    args = parser.parse_args()
    client = KalshiClient.from_env(demo=args.demo, dry_run=True)
    cash, value = client.account()
    rows = []
    titles: dict[str, str] = {}
    for position in client.market_positions():
        count = _dollars(position.get("position_fp"))
        if not count:
            continue
        market = client.market(position["ticker"])
        event = market["event_ticker"]
        if event not in titles:
            titles[event] = client.event(event).get("title") or event
        yes = count > 0
        bid = _dollars(market.get("yes_bid_dollars" if yes else "no_bid_dollars"))
        cost = _dollars(position.get("market_exposure_dollars"))
        fees = _dollars(position.get("fees_paid_dollars"))
        rows.append(
            {
                "day": event_day(event),
                "game": titles[event],
                "holding": f"{abs(count):g} {'YES' if yes else 'NO'} "
                f"{market.get('yes_sub_title') or position['ticker']}",
                "cost": cost + fees,
                "bid": bid,
                "worth_now": abs(count) * bid,
                "pays": abs(count),
            }
        )
    print(f"cash ${cash:,.2f}; Kalshi values the positions at ${value:,.2f}")
    if not rows:
        print("no open positions")
        return
    table = pd.DataFrame(rows).sort_values(["day", "game"])
    table["if_sold"] = table["worth_now"] - table["cost"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print(
            table.to_string(
                index=False,
                formatters={
                    "cost": "${:,.2f}".format,
                    "bid": "{:.2f}".format,
                    "worth_now": "${:,.2f}".format,
                    "pays": "${:,.0f}".format,
                    "if_sold": "{:+,.2f}".format,
                },
            )
        )
    print(
        f"\n{len(table)} positions on {table['game'].nunique()} games: cost "
        f"${table['cost'].sum():,.2f} with fees, worth ${table['worth_now'].sum():,.2f} "
        f"at the bids, ${table['pays'].sum():,.0f} if every one wins"
    )


if __name__ == "__main__":
    main()
