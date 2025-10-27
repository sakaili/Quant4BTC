# backtest.py
import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    main()
