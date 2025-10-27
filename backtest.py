# backtest.py
import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from indicators import IndicatorEngine
from signals import SignalBuilder
from selector import FactorSelector
from evaluator import Evaluator


def timeframe_to_pandas(tf: str) -> str:
    tf = tf.strip().lower()
    unit = tf[-1]
    value = tf[:-1]
    mapping = {"m": "T", "h": "H", "d": "D"}
    if unit not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return f"{int(value)}{mapping[unit]}"


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    freq = timeframe_to_pandas(tf)
    agg = df.resample(freq, label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return agg.dropna()


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest SuperTrend strategy.")
    parser.add_argument("--data", required=True, help="CSV file with columns timestamp,open,high,low,close,volume")
    parser.add_argument("--symbol", help="Symbol override")
    parser.add_argument("--timeframe", default="5m", help="Target timeframe for backtest")
    parser.add_argument("--selection", choices=["rank", "cluster", "regime_kmeans"], default=None, help="Factor selection method")
    parser.add_argument("--factor", type=float, help="Force a fixed SuperTrend factor")
    parser.add_argument("--start", help="ISO start time filter")
    parser.add_argument("--end", help="ISO end time filter")
    return parser.parse_args()


def load_data(path: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    df = df.rename(columns=str.capitalize)
    return df


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "_")


def _prepare_result_dir(symbol: str, timeframe: str) -> Path:
    base = Path("backTestResult")
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base / f"{_safe_symbol(symbol)}_{timeframe}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _build_timeseries(df_atr: pd.DataFrame, st: dict, signals: np.ndarray, evaluator: Evaluator, cfg: Config) -> pd.DataFrame:
    df = df_atr.copy()
    df["signal"] = signals.astype(int)
    df["position"] = signals.astype(float)
    df["supertrend"] = st["output"]
    df["supertrend_upper"] = st["upper"]
    df["supertrend_lower"] = st["lower"]
    pnl = evaluator._apply_cost_on_pnl(
        df["position"].to_numpy(),
        df["Close"].to_numpy(),
        cfg.fee_rate,
        cfg.slippage_rate,
        cfg.turnover_penalty,
    )
    equity = np.cumprod(1.0 + pnl) * cfg.initial_capital
    df["equity"] = equity
    df.index.name = "timestamp"
    return df.reset_index()


def _extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "action"])
    sig = df["signal"].to_numpy()
    closes = df["Close"].to_numpy()
    timestamps = pd.to_datetime(df["timestamp"])
    trades = []
    for i in range(1, len(sig)):
        prev_sig = sig[i - 1]
        curr_sig = sig[i]
        if curr_sig == prev_sig:
            continue
        action = "buy" if curr_sig > prev_sig else "sell"
        trades.append(
            {
                "timestamp": timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i],
                "price": float(closes[i]),
                "action": action,
            }
        )
    return pd.DataFrame(trades)


def _plot_artifacts(timeseries: pd.DataFrame, trades: pd.DataFrame, out_path: Path):
    if timeseries.empty:
        return
    ts = pd.to_datetime(timeseries["timestamp"])
    fig, (ax_price, ax_equity) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    ax_price.plot(ts, timeseries["Close"], label="Close", color="#1f77b4", linewidth=1.2)
    ax_price.plot(ts, timeseries["supertrend"], label="SuperTrend", color="#ff7f0e", linewidth=1.0)
    ax_price.plot(ts, timeseries["supertrend_upper"], label="ST Upper", color="#2ca02c", linewidth=0.8, alpha=0.6)
    ax_price.plot(ts, timeseries["supertrend_lower"], label="ST Lower", color="#d62728", linewidth=0.8, alpha=0.6)

    if not trades.empty:
        buy_points = trades[trades["action"] == "buy"]
        sell_points = trades[trades["action"] == "sell"]
        if not buy_points.empty:
            ax_price.scatter(
                pd.to_datetime(buy_points["timestamp"]),
                buy_points["price"],
                marker="^",
                color="#2ca02c",
                label="Buy",
                s=60,
                zorder=5,
            )
        if not sell_points.empty:
            ax_price.scatter(
                pd.to_datetime(sell_points["timestamp"]),
                sell_points["price"],
                marker="v",
                color="#d62728",
                label="Sell",
                s=60,
                zorder=5,
            )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_equity.plot(ts, timeseries["equity"], label="Equity", color="#9467bd", linewidth=1.2)
    ax_equity.set_ylabel("Equity")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, linestyle="--", alpha=0.3)
    ax_equity.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config()
    symbol = args.symbol or cfg.symbol
    cfg = replace(cfg, symbol=symbol, timeframe=args.timeframe)
    if args.selection:
        cfg = replace(cfg, selection=args.selection)
    data = load_data(args.data, args.start, args.end)
    resampled = resample_ohlcv(data, args.timeframe)
    if len(resampled) < max(200, cfg.metric_lookback):
        raise ValueError("Not enough data after resampling")

    ind = IndicatorEngine(cfg)
    df_atr = ind.compute_atr(resampled)
    if args.factor is not None:
        factor = float(args.factor)
    else:
        selector = FactorSelector(cfg)
        factor = selector._select(df_atr)

    st = ind.compute_supertrend(df_atr, factor)
    sig_builder = SignalBuilder(cfg)
    signals = sig_builder.build(df_atr, st)

    evaluator = Evaluator(cfg)
    metrics = evaluator.evaluate_factor(df_atr, st)
    score = evaluator.score(metrics)

    print(f"Backtest result for {symbol} @ {args.timeframe}")
    print(f"Factor: {factor:.4f}")
    print(f"Score: {score:.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    result_dir = _prepare_result_dir(symbol, args.timeframe)
    timeseries = _build_timeseries(df_atr, st, signals, evaluator, cfg)
    trades = _extract_trades(timeseries)
    timeseries.to_csv(result_dir / "timeseries.csv", index=False)
    if not trades.empty:
        trades.to_csv(result_dir / "trades.csv", index=False)
    summary = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "factor": factor,
        "score": score,
        "metrics": metrics,
        "run_dir": str(result_dir),
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    _plot_artifacts(timeseries, trades, result_dir / "backtest_plot.png")
    print(f"Artifacts saved under: {result_dir}")


if __name__ == "__main__":
    main()
