# download_data.py
import argparse
import sys
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data from OKX.")
    parser.add_argument("--symbol", type=str, help="Trading pair, e.g. BTC/USDT:USDT")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe, default 1m")
    parser.add_argument("--since", type=str, help="ISO format start time, e.g. 2022-01-01T00:00:00Z")
    parser.add_argument("--until", type=str, help="ISO format end time")
    parser.add_argument("--limit", type=int, default=500, help="Max candles per request")
    parser.add_argument("--output", type=str, default="ohlcv.csv", help="Output CSV path")
    parser.add_argument("--demo", action="store_true", help="Force sandbox mode")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    return parser.parse_args()


def iso_to_millis(ts: str | None) -> int | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def main():
    args = parse_args()
    cfg = Config()
    symbol = args.symbol or cfg.symbol

    exchange = ccxt.okx({
        "apiKey": cfg.okx_api_key,
        "secret": cfg.okx_secret,
        "password": cfg.okx_password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
        "proxies": cfg.proxies(),
        "timeout": 20000,
    })
    sandbox = args.demo or cfg.use_demo
    exchange.set_sandbox_mode(bool(sandbox))
    exchange.load_markets()

    since = iso_to_millis(args.since)
    until = iso_to_millis(args.until)
    limit = max(1, min(args.limit, 500))

    all_rows = []
    next_since = since
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, args.timeframe, since=next_since, limit=limit)
        except Exception as exc:
            print(f"fetch_ohlcv failed: {exc}", file=sys.stderr)
            time.sleep(1)
            continue
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if not args.no_progress:
            last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(
                f"Fetched {len(batch)} rows this batch, total {len(all_rows)} rows up to {last_dt.isoformat()}",
                flush=True,
            )
        next_since = last_ts + 1
        if until and next_since >= until:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not all_rows:
        print("No data fetched.")
        return

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    if until:
        until_ts = pd.to_datetime(until, unit="ms", utc=True)
        df = df[df["timestamp"] <= until_ts]
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
