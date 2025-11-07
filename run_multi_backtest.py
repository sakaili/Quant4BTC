import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path("data/btc_usdc_5m_2025.csv")
TIMEFRAME = "5m"
STRATEGY = "supertrend"
BACKTEST_SCRIPT = Path("backtest.py")


@dataclass
class ParamCombo:
    name: str
    env: Dict[str, str]


COMBOS: List[ParamCombo] = [
    ParamCombo(
        name="ultra_fast_regime",
        env={
            "METRIC_LOOKBACK": "16",
            "REGIME_LOOKBACK": "32",
            "ZSCORE_WINDOW": "10",
            "MIN_REGIME_SAMPLES": "6",
            "FACTOR_HOLD_BARS": "8",
            "SELECTION": "regime_kmeans",
        },
    ),
    ParamCombo(
        name="fast_regime_short",
        env={
            "METRIC_LOOKBACK": "24",
            "REGIME_LOOKBACK": "48",
            "ZSCORE_WINDOW": "16",
            "MIN_REGIME_SAMPLES": "10",
            "FACTOR_HOLD_BARS": "10",
            "SELECTION": "regime_kmeans",
        },
    ),
    ParamCombo(
        name="reactive_cluster",
        env={
            "METRIC_LOOKBACK": "28",
            "REGIME_LOOKBACK": "56",
            "ZSCORE_WINDOW": "20",
            "MIN_REGIME_SAMPLES": "12",
            "FACTOR_HOLD_BARS": "12",
            "SELECTION": "cluster",
        },
    ),
    ParamCombo(
        name="balanced_regime",
        env={
            "METRIC_LOOKBACK": "40",
            "REGIME_LOOKBACK": "80",
            "ZSCORE_WINDOW": "36",
            "MIN_REGIME_SAMPLES": "24",
            "FACTOR_HOLD_BARS": "14",
            "SELECTION": "regime_kmeans",
        },
    ),
    ParamCombo(
        name="cluster_metric",
        env={
            "METRIC_LOOKBACK": "50",
            "REGIME_LOOKBACK": "100",
            "ZSCORE_WINDOW": "45",
            "MIN_REGIME_SAMPLES": "30",
            "FACTOR_HOLD_BARS": "16",
            "SELECTION": "cluster",
        },
    ),
    ParamCombo(
        name="medium_regime",
        env={
            "METRIC_LOOKBACK": "70",
            "REGIME_LOOKBACK": "140",
            "ZSCORE_WINDOW": "100",
            "MIN_REGIME_SAMPLES": "70",
            "FACTOR_HOLD_BARS": "24",
            "SELECTION": "regime_kmeans",
        },
    ),
    ParamCombo(
        name="baseline_rank",
        env={
            "METRIC_LOOKBACK": "35",
            "REGIME_LOOKBACK": "70",
            "ZSCORE_WINDOW": "35",
            "MIN_REGIME_SAMPLES": "18",
            "FACTOR_HOLD_BARS": "12",
            "SELECTION": "rank",
        },
    ),
]


def parse_artifact_dir(output: str) -> str:
    match = re.search(r"Artifacts saved under:\s*(.+)", output)
    return match.group(1).strip() if match else "UNKNOWN"


def run_combo(combo: ParamCombo, idx: int, total: int, start_time: float) -> dict:
    env = os.environ.copy()
    env.update(combo.env)
    cmd = [
        sys.executable,
        str(BACKTEST_SCRIPT),
        "--data",
        str(DATA_PATH),
        "--timeframe",
        TIMEFRAME,
        "--strategy",
        STRATEGY,
    ]
    print(f"\n[{idx}/{total}] Running '{combo.name}' ...")
    run_start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - run_start
    remaining = max(total - idx, 0)
    avg_time = (time.time() - start_time) / idx
    eta = avg_time * remaining
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Backtest failed for {combo.name} (exit code {proc.returncode})")
    artifact_dir = parse_artifact_dir(proc.stdout)
    print(f"    done in {elapsed:0.1f}s, ETA for remaining ≈ {eta:0.1f}s")
    return {
        "name": combo.name,
        "artifact_dir": artifact_dir,
        "stdout": proc.stdout,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file {DATA_PATH} not found.")
    if not BACKTEST_SCRIPT.exists():
        raise FileNotFoundError("backtest.py not found in current directory.")
    total = len(COMBOS)
    print(f"Starting {total} backtests using data {DATA_PATH} ...")
    start = time.time()
    results = []
    for idx, combo in enumerate(COMBOS, 1):
        results.append(run_combo(combo, idx, total, start))
    print("\nAll runs completed. Summary:")
    for res in results:
        print(f"- {res['name']}: {res['artifact_dir']}")


if __name__ == "__main__":
    main()
