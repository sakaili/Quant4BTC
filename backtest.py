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


def _event_driven_backtest(close: np.ndarray, signal: np.ndarray, cfg: Config) -> dict[str, np.ndarray | float]:
    close_arr = np.asarray(close, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)
    if close_arr.shape != signal_arr.shape:
        raise ValueError("Price and signal length mismatch")
    signal_arr = signal_arr.astype(int)
    qty = float(getattr(cfg, "backtest_trade_size", 0.01))
    qty = max(0.0, qty)
    fee_rate = max(0.0, float(getattr(cfg, "fee_rate", 0.0)))
    slippage = max(0.0, float(getattr(cfg, "slippage_rate", 0.0)))
    turnover_penalty = max(0.0, float(getattr(cfg, "turnover_penalty", 0.0)))
    initial_capital = float(getattr(cfg, "initial_capital", 0.0))

    cash = initial_capital
    position_units = 0.0
    equity_list: list[float] = []
    cash_list: list[float] = []
    position_units_list: list[float] = []
    pnl_list: list[float] = []
    return_list: list[float] = []
    delta_list: list[float] = []
    cost_list: list[float] = []
    fee_list: list[float] = []
    penalty_list: list[float] = []
    notional_list: list[float] = []
    trade_pnl_list: list[float] = []
    exec_price_list: list[float] = []

    prev_equity = initial_capital
    eps = 1e-12

    for price, target_signal in zip(close_arr, signal_arr):
        price_f = float(price)
        target_units = float(target_signal) * qty
        delta = target_units - position_units
        if abs(delta) <= eps:
            delta = 0.0

        equity_before_trade = cash + position_units * price_f
        fee = 0.0
        penalty_cost = 0.0
        exec_price = float("nan")
        trade_pnl = 0.0
        if delta != 0.0:
            exec_price = price_f * (1.0 + slippage) if delta > 0 else price_f * (1.0 - slippage)
            cash -= delta * exec_price
            fee = abs(delta) * exec_price * fee_rate
            cash -= fee
            if turnover_penalty > 0.0 and qty > 0.0:
                penalty_cost = turnover_penalty * (abs(delta) / (qty + eps))
                cash -= penalty_cost
            position_units = target_units
            trade_pnl = (cash + position_units * price_f) - equity_before_trade
        else:
            position_units = target_units

        trade_cost = fee + penalty_cost
        delta_list.append(delta)
        cost_list.append(trade_cost)
        fee_list.append(fee)
        penalty_list.append(penalty_cost)
        exec_price_list.append(exec_price)

        notional = position_units * price_f
        notional_list.append(notional)
        equity = cash + notional
        pnl = equity - prev_equity
        ret = pnl / (prev_equity + eps)

        equity_list.append(equity)
        pnl_list.append(pnl)
        return_list.append(ret)
        cash_list.append(cash)
        position_units_list.append(position_units)
        trade_pnl_list.append(trade_pnl)

        prev_equity = equity

    delta_arr = np.asarray(delta_list, dtype=float)
    turns = float(np.sum(np.abs(delta_arr)) / qty) if qty > 0.0 else 0.0

    return {
        "equity": np.asarray(equity_list, dtype=float),
        "cash": np.asarray(cash_list, dtype=float),
        "pnl": np.asarray(pnl_list, dtype=float),
        "returns": np.asarray(return_list, dtype=float),
        "position_units": np.asarray(position_units_list, dtype=float),
        "notional": np.asarray(notional_list, dtype=float),
        "trade_delta": delta_arr,
        "trade_cost": np.asarray(cost_list, dtype=float),
        "trade_fee": np.asarray(fee_list, dtype=float),
        "trade_penalty": np.asarray(penalty_list, dtype=float),
        "trade_exec_price": np.asarray(exec_price_list, dtype=float),
        "trade_pnl": np.asarray(trade_pnl_list, dtype=float),
        "turns": turns,
    }


def _evaluate_event_metrics(state: dict[str, np.ndarray | float], cfg: Config) -> dict[str, float]:
    ret = np.asarray(state.get("returns", []), dtype=float)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    equity = np.asarray(state.get("equity", []), dtype=float)
    initial_capital = float(getattr(cfg, "initial_capital", 0.0))

    eq_norm: np.ndarray
    if equity.size > 0 and initial_capital > 0:
        eq_norm = equity / initial_capital
    elif ret.size > 0:
        eq_norm = np.cumprod(1.0 + ret)
    else:
        eq_norm = np.array([], dtype=float)

    if eq_norm.size == 0:
        mdd = 0.0
        last_eq = 1.0
    else:
        peak = np.maximum.accumulate(eq_norm)
        drawdown = (peak - eq_norm) / (peak + 1e-12)
        mdd = float(drawdown.max())
        last_eq = float(eq_norm[-1])

    ann = 252.0
    sharpe = float((np.mean(ret) * ann) / (np.std(ret) * np.sqrt(ann) + 1e-12)) if ret.size > 0 else 0.0
    if eq_norm.size == 0:
        cagr = 0.0
    elif last_eq <= 0:
        cagr = -1.0
    else:
        cagr = float(last_eq ** (ann / max(len(eq_norm), 1)) - 1.0)

    sortino = Evaluator._sortino(ret, ann=np.sqrt(ann)) if ret.size > 0 else 0.0
    calmar = (cagr / (mdd + 1e-12)) if mdd > 0 else cagr
    turns = float(state.get("turns", 0.0) or 0.0)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "last_eq": last_eq,
        "turns": turns,
        "sortino": sortino,
        "calmar": calmar,
    }


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
    backtest_state = _event_driven_backtest(close_arr, pos_arr.astype(int), cfg)

    timeseries = df_atr.copy()
    timeseries["signal"] = np.asarray(signals, dtype=int)
    timeseries["position"] = pos_arr
    timeseries["supertrend"] = np.asarray(st_output, dtype=float)
    timeseries["supertrend_upper"] = np.asarray(st_upper, dtype=float)
    timeseries["supertrend_lower"] = np.asarray(st_lower, dtype=float)
    timeseries["factor"] = np.asarray(factors, dtype=float)
    timeseries["equity"] = backtest_state["equity"]
    timeseries["cash"] = backtest_state["cash"]
    timeseries["pnl"] = backtest_state["pnl"]
    timeseries["return"] = backtest_state["returns"]
    timeseries["position_units"] = backtest_state["position_units"]
    timeseries["notional"] = backtest_state["notional"]
    timeseries["trade_delta"] = backtest_state["trade_delta"]
    timeseries["trade_cost"] = backtest_state["trade_cost"]
    timeseries["trade_fee"] = backtest_state["trade_fee"]
    timeseries["trade_penalty"] = backtest_state["trade_penalty"]
    timeseries["trade_exec_price"] = backtest_state["trade_exec_price"]
    timeseries["trade_pnl"] = backtest_state["trade_pnl"]
    timeseries.index.name = "timestamp"
    timeseries = timeseries.reset_index()
    progress.finish()

    metrics = _evaluate_event_metrics(backtest_state, cfg)
    score = evaluator.score(metrics)
    last_factor = factors[-1] if factors else None
    return timeseries, last_factor, metrics, score


def _build_macd_timeseries(df_macd: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray | float]]:
    df = df_macd.copy()
    df["signal"] = df["signal"].astype(int)
    signal_arr = df["signal"].to_numpy(dtype=int)
    close_arr = df["Close"].to_numpy(dtype=float)
    backtest_state = _event_driven_backtest(close_arr, signal_arr, cfg)
    df["position"] = signal_arr.astype(float)
    df["position_units"] = backtest_state["position_units"]
    df["notional"] = backtest_state["notional"]
    df["cash"] = backtest_state["cash"]
    df["equity"] = backtest_state["equity"]
    df["pnl"] = backtest_state["pnl"]
    df["return"] = backtest_state["returns"]
    df["trade_delta"] = backtest_state["trade_delta"]
    df["trade_cost"] = backtest_state["trade_cost"]
    df["trade_fee"] = backtest_state["trade_fee"]
    df["trade_penalty"] = backtest_state["trade_penalty"]
    df["trade_exec_price"] = backtest_state["trade_exec_price"]
    df["trade_pnl"] = backtest_state["trade_pnl"]
    df.index.name = "timestamp"
    return df.reset_index(), backtest_state


def _simulate_macd(
    resampled: pd.DataFrame,
    cfg: Config,
    evaluator: Evaluator,
    indicator: IndicatorEngine,
) -> tuple[pd.DataFrame, dict, float]:
    macd_ts = build_macd_triple_filter_timeseries(resampled, cfg, indicator)
    if macd_ts.empty:
        raise ValueError("Not enough data to compute MACD triple filter signals")

    macd_ts, backtest_state = _build_macd_timeseries(macd_ts, cfg)
    progress = _ProgressPrinter(len(macd_ts), "MACD backtest")
    for idx in range(len(macd_ts)):
        progress.update(idx + 1)
    progress.finish()

    metrics = _evaluate_event_metrics(backtest_state, cfg)
    score = evaluator.score(metrics)
    return macd_ts, metrics, score


def _extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "action", "quantity", "pnl"])
    if "trade_delta" not in df or "trade_pnl" not in df:
        raise ValueError("timeseries missing trade attribution columns")
    trade_delta = df["trade_delta"].to_numpy(dtype=float)
    mask = np.abs(trade_delta) > 1e-12
    if not mask.any():
        return pd.DataFrame(columns=["timestamp", "price", "action", "quantity", "pnl"])
    timestamps = pd.to_datetime(df.loc[mask, "timestamp"])
    exec_price = df.loc[mask, "trade_exec_price"].to_numpy(dtype=float)
    close_price = df.loc[mask, "Close"].to_numpy(dtype=float)
    price = np.where(np.isfinite(exec_price), exec_price, close_price)
    pnl = df.loc[mask, "trade_pnl"].to_numpy(dtype=float)
    quantity = np.abs(trade_delta[mask])
    actions = np.where(trade_delta[mask] > 0, "buy", "sell")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": price,
            "action": actions,
            "quantity": quantity,
            "pnl": pnl,
        }
    )


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
