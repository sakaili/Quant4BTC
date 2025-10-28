# backtest.py
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from evaluator import Evaluator
from indicators import IndicatorEngine
from selector import FactorSelector
from signals import SignalBuilder
from strategies.macd_triple_filter import build_macd_triple_filter_timeseries


def timeframe_to_pandas(tf: str) -> str:
    tf = tf.strip().lower()
    unit = tf[-1]
    value = tf[:-1]
    mapping = {"m": "min", "h": "H", "d": "D"}
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
    parser = argparse.ArgumentParser(description="Backtest supported strategies.")
    parser.add_argument("--data", required=True, help="CSV file with columns timestamp,open,high,low,close,volume")
    parser.add_argument("--symbol", help="Symbol override")
    parser.add_argument("--timeframe", default="5m", help="Target timeframe for backtest")
    parser.add_argument(
        "--strategy",
        choices=["supertrend", "macd_triple_filter"],
        default="supertrend",
        help="Strategy name to backtest",
    )
    parser.add_argument(
        "--selection",
        choices=["rank", "cluster", "regime_kmeans"],
        default=None,
        help="SuperTrend factor selection method",
    )
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


def _prepare_result_dir(symbol: str, timeframe: str, strategy: str) -> Path:
    base = Path("backTestResult")
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base / f"{_safe_symbol(symbol)}_{timeframe}_{strategy}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


class _ProgressPrinter:
    def __init__(self, total: int, label: str) -> None:
        self.total = max(1, int(total))
        self.label = label
        self._printed = False
        self._last_pct = -1

    def update(self, current: int) -> None:
        cur = max(0, min(self.total, int(current)))
        pct = int((cur * 100) / self.total)
        if pct == self._last_pct:
            return
        bar_len = 30
        filled = int((pct * bar_len) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r{self.label} [{bar}] {pct:3d}%")
        sys.stdout.flush()
        self._last_pct = pct
        self._printed = True

    def finish(self) -> None:
        self.update(self.total)
        if self._printed:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _simulate_supertrend(
    resampled: pd.DataFrame,
    cfg: Config,
    evaluator: Evaluator,
    indicator: IndicatorEngine,
    factor_override: float | None,
) -> tuple[pd.DataFrame, float | None, dict, float]:
    df_atr = indicator.compute_atr(resampled)
    if len(df_atr) < max(200, cfg.metric_lookback):
        raise ValueError("Not enough data after resampling")

    selector = None if factor_override is not None else FactorSelector(cfg)
    signal_builder = SignalBuilder(cfg)

    signals: list[int] = []
    positions: list[float] = []
    st_output: list[float] = []
    st_upper: list[float] = []
    st_lower: list[float] = []
    factors: list[float] = []
    progress = _ProgressPrinter(len(df_atr), "SuperTrend backtest")

    for i in range(len(df_atr)):
        df_slice = df_atr.iloc[: i + 1]
        factor = factor_override if factor_override is not None else selector.maybe_select(df_slice)
        st_slice = indicator.compute_supertrend(df_slice, factor)
        sig_slice = signal_builder.build(df_slice, st_slice)
        current_signal = int(sig_slice[-1])
        signals.append(current_signal)
        positions.append(float(current_signal))
        st_output.append(float(st_slice["output"][-1]))
        st_upper.append(float(st_slice["upper"][-1]))
        st_lower.append(float(st_slice["lower"][-1]))
        factors.append(float(factor))
        progress.update(i + 1)

    pos_arr = np.asarray(positions, dtype=float)
    close_arr = df_atr["Close"].to_numpy(dtype=float)
    pnl = evaluator._apply_cost_on_pnl(  # noqa: SLF001
        pos_arr,
        close_arr,
        cfg.fee_rate,
        cfg.slippage_rate,
        cfg.turnover_penalty,
    )
    equity = np.cumprod(1.0 + pnl) * cfg.initial_capital

    timeseries = df_atr.copy()
    timeseries["signal"] = np.asarray(signals, dtype=int)
    timeseries["position"] = pos_arr
    timeseries["supertrend"] = np.asarray(st_output, dtype=float)
    timeseries["supertrend_upper"] = np.asarray(st_upper, dtype=float)
    timeseries["supertrend_lower"] = np.asarray(st_lower, dtype=float)
    timeseries["factor"] = np.asarray(factors, dtype=float)
    timeseries["equity"] = equity
    timeseries.index.name = "timestamp"
    timeseries = timeseries.reset_index()
    progress.finish()

    metrics = evaluator.evaluate_positions(close_arr, pos_arr)
    score = evaluator.score(metrics)
    last_factor = factors[-1] if factors else None
    return timeseries, last_factor, metrics, score


def _build_macd_timeseries(df_macd: pd.DataFrame, evaluator: Evaluator, cfg: Config) -> pd.DataFrame:
    df = df_macd.copy()
    df["signal"] = df["signal"].astype(int)
    df["position"] = df["position"].astype(float)
    pnl = evaluator._apply_cost_on_pnl(  # noqa: SLF001
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


def _simulate_macd(
    resampled: pd.DataFrame,
    cfg: Config,
    evaluator: Evaluator,
    indicator: IndicatorEngine,
) -> tuple[pd.DataFrame, dict, float]:
    macd_ts = build_macd_triple_filter_timeseries(resampled, cfg, indicator)
    if macd_ts.empty:
        raise ValueError("Not enough data to compute MACD triple filter signals")

    macd_ts = _build_macd_timeseries(macd_ts, evaluator, cfg)
    positions = macd_ts["position"].to_numpy(dtype=float)
    closes = macd_ts["Close"].to_numpy(dtype=float)
    progress = _ProgressPrinter(len(macd_ts), "MACD backtest")
    for idx in range(len(macd_ts)):
        progress.update(idx + 1)
    progress.finish()

    metrics = evaluator.evaluate_positions(closes, positions)
    score = evaluator.score(metrics)
    return macd_ts, metrics, score


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


def _plot_artifacts(timeseries: pd.DataFrame, trades: pd.DataFrame, strategy: str, out_path: Path):
    if timeseries.empty:
        return
    ts = pd.to_datetime(timeseries["timestamp"])

    if strategy == "supertrend":
        fig, (ax_price, ax_equity) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        ax_price.plot(ts, timeseries["Close"], label="Close", color="#1f77b4", linewidth=1.2)
        ax_price.plot(ts, timeseries["supertrend"], label="SuperTrend", color="#ff7f0e", linewidth=1.0)
        ax_price.plot(ts, timeseries["supertrend_upper"], label="ST Upper", color="#2ca02c", linewidth=0.8, alpha=0.6)
        ax_price.plot(ts, timeseries["supertrend_lower"], label="ST Lower", color="#d62728", linewidth=0.8, alpha=0.6)
    else:
        fig, (ax_price, ax_macd, ax_equity) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))

        ax_price.plot(ts, timeseries["Close"], label="Close", color="#1f77b4", linewidth=1.2)
        if "RegimeEMA" in timeseries:
            ax_price.plot(ts, timeseries["RegimeEMA"], label="Regime EMA", color="#ff7f0e", linewidth=1.0)

        ax_macd.plot(ts, timeseries["DIF"], label="DIF", color="#2ca02c", linewidth=1.0)
        ax_macd.plot(ts, timeseries["DEA"], label="DEA", color="#d62728", linewidth=1.0)
        hist = timeseries["Histogram"]
        ax_macd.fill_between(ts, 0, hist, where=hist >= 0, facecolor="#2ca02c", alpha=0.2, interpolate=True)
        ax_macd.fill_between(ts, 0, hist, where=hist < 0, facecolor="#d62728", alpha=0.2, interpolate=True)
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left")
        ax_macd.grid(True, linestyle="--", alpha=0.3)

    if not trades.empty:
        buy_points = trades[trades["action"] == "buy"]
        sell_points = trades[trades["action"] == "sell"]
        target_ax = ax_price
        if not buy_points.empty:
            target_ax.scatter(
                pd.to_datetime(buy_points["timestamp"]),
                buy_points["price"],
                marker="^",
                color="#2ca02c",
                label="Buy",
                s=60,
                zorder=5,
            )
        if not sell_points.empty:
            target_ax.scatter(
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
    base_cfg = Config()
    strategy = (args.strategy or "supertrend").lower()

    symbol = args.symbol or base_cfg.symbol
    cfg = replace(base_cfg, symbol=symbol, timeframe=args.timeframe)
    if strategy == "supertrend" and args.selection:
        cfg = replace(cfg, selection=args.selection)
    data = load_data(args.data, args.start, args.end)
    resampled = resample_ohlcv(data, args.timeframe)

    ind = IndicatorEngine(cfg)
    evaluator = Evaluator(cfg)

    if strategy == "supertrend":
        factor_override = float(args.factor) if args.factor is not None else None
        timeseries, factor, metrics, score = _simulate_supertrend(resampled, cfg, evaluator, ind, factor_override)
    else:
        if args.selection or args.factor is not None:
            print("[WARN] Selection or factor arguments are ignored for macd_triple_filter.")
        timeseries, metrics, score = _simulate_macd(resampled, cfg, evaluator, ind)
        factor = None

    result_dir = _prepare_result_dir(symbol, args.timeframe, strategy)
    trades = _extract_trades(timeseries)
    timeseries.to_csv(result_dir / "timeseries.csv", index=False)
    if not trades.empty:
        trades.to_csv(result_dir / "trades.csv", index=False)

    summary = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "strategy": strategy,
        "score": score,
        "metrics": metrics,
        "run_dir": str(result_dir),
    }
    if strategy == "supertrend":
        summary["factor"] = factor

    with open(result_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _plot_artifacts(timeseries, trades, strategy, result_dir / "backtest_plot.png")

    print(f"Backtest result for {symbol} @ {args.timeframe} ({strategy})")
    if strategy == "supertrend":
        if factor is None:
            print("Factor: N/A")
        else:
            print(f"Factor: {factor:.4f}")
    print(f"Score: {score:.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Artifacts saved under: {result_dir}")


if __name__ == "__main__":
    main()
class _ProgressPrinter:
    def __init__(self, total: int, label: str) -> None:
        self.total = max(1, int(total))
        self.label = label
        self._printed = False
        self._last_pct = -1

    def update(self, current: int) -> None:
        cur = max(0, min(self.total, int(current)))
        pct = int((cur * 100) / self.total)
        if pct == self._last_pct:
            return
        bar_len = 30
        filled = int((pct * bar_len) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r{self.label} [{bar}] {pct:3d}%")
        sys.stdout.flush()
        self._last_pct = pct
        self._printed = True

    def finish(self) -> None:
        self.update(self.total)
        if self._printed:
            sys.stdout.write("\n")
            sys.stdout.flush()
