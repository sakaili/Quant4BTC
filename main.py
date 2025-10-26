# main.py
import signal
import time
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from config import Config, setup_logging
from exchange_client import ExchangeClient
from runner import StrategyRunner
from data import DataFetcher


def shutdown_handler(signum, frame):
    """捕捉关闭信号，保持退出过程干净。"""
    print("收到关闭信号")


def main():
    """读取配置，初始化组件后启动主循环。"""
    base_cfg = Config()
    base_logger = setup_logging(base_cfg.log_level)

    def safe_symbol(symbol: str) -> str:
        return symbol.replace('/', '_').replace(':', '_').replace(' ', '_')

    def resolve_csv_path(base_path: str, symbol_tag: str) -> str:
        path = Path(base_path)
        if path.is_dir():
            return str(path / f"{symbol_tag}.csv")
        if path.suffix:
            return str(path.with_name(f"{path.stem}_{symbol_tag}{path.suffix}"))
        return str(path.parent / f"{path.name}_{symbol_tag}.csv")

    def run_for_symbol(symbol: str):
        symbol_tag = safe_symbol(symbol)
        symbol_logger = base_logger.getChild(symbol_tag)
        csv_path = resolve_csv_path(base_cfg.csv_log_file, symbol_tag)
        symbol_cfg = replace(base_cfg, symbol=symbol, csv_log_file=csv_path)
        exch = ExchangeClient(symbol_cfg, symbol_logger)
        runner = StrategyRunner(symbol_cfg, exch, symbol_logger)

        def align_to_next_candle():
            tf_sec = DataFetcher.timeframe_seconds(symbol_cfg.timeframe)
            now = datetime.now()
            epoch = int(now.timestamp())
            remain = tf_sec - (epoch % tf_sec)
            sleep_secs = max(0, remain)
            symbol_logger.info(f"[{symbol}] 等待 {sleep_secs:.2f}s 对齐下一根 {symbol_cfg.timeframe} K 线")
            time.sleep(sleep_secs)

        runner.align_and_loop(align_to_next_candle)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    symbols = base_cfg.symbol_list
    if len(symbols) == 1:
        run_for_symbol(symbols[0])
        return

    threads = []
    for symbol in symbols:
        t = threading.Thread(target=run_for_symbol, args=(symbol,), name=f"runner-{safe_symbol(symbol)}", daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

