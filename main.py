# main.py
import signal
import time
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from config import Config, setup_logging
from csv_logger import CsvLogger
from data import DataFetcher
from exchange_client import ExchangeClient
from indicators import IndicatorEngine
from order_executor import OrderExecutor
from position_reader import PositionReader
from selector import FactorSelector
from signals import SignalBuilder
from strategies.supertrend import SuperTrendStrategy
from strategies.ultimate_scalping import UltimateScalpingStrategy


STRATEGY_REGISTRY = {
    "supertrend": SuperTrendStrategy,
    "ultimate_scalping": UltimateScalpingStrategy,
}


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

    strategy_cls = STRATEGY_REGISTRY.get(base_cfg.strategy_name.lower())
    if strategy_cls is None:
        raise ValueError(f"未知策略 '{base_cfg.strategy_name}'，请检查 STRATEGY_NAME 环境变量")

    def run_for_symbol(symbol: str):
        symbol_tag = safe_symbol(symbol)
        symbol_logger = base_logger.getChild(symbol_tag)
        csv_path = resolve_csv_path(base_cfg.csv_log_file, symbol_tag)
        symbol_cfg = replace(base_cfg, symbol=symbol, csv_log_file=csv_path)
        fixed_size = symbol_cfg.fixed_order_size
        if symbol.upper().startswith("ETH/") or symbol.upper().startswith("ETH"):
            fixed_size = 0.3
        symbol_cfg = replace(symbol_cfg, fixed_order_size=fixed_size)
        exch = ExchangeClient(symbol_cfg, symbol_logger)
        data_fetcher = DataFetcher(symbol_cfg, exch, symbol_logger)
        indicator_engine = IndicatorEngine(symbol_cfg)
        signal_builder = SignalBuilder(symbol_cfg)
        factor_selector = FactorSelector(symbol_cfg)
        order_executor = OrderExecutor(symbol_cfg, exch, symbol_logger)
        csv_logger = CsvLogger(csv_path, symbol_logger)
        position_reader = PositionReader(symbol_cfg, exch)
        strategy = strategy_cls(
            cfg=symbol_cfg,
            logger=symbol_logger,
            data_fetcher=data_fetcher,
            indicator_engine=indicator_engine,
            signal_builder=signal_builder,
            factor_selector=factor_selector,
            order_executor=order_executor,
            position_reader=position_reader,
            csv_logger=csv_logger,
        )

        def align_to_next_candle():
            tf_sec = DataFetcher.timeframe_seconds(symbol_cfg.timeframe)
            now = datetime.now()
            epoch = int(now.timestamp())
            remain = tf_sec - (epoch % tf_sec)
            sleep_secs = max(0, remain)
            symbol_logger.info(f"[{symbol}] 等待 {sleep_secs:.2f}s 对齐下一根 {symbol_cfg.timeframe} K 线")
            time.sleep(sleep_secs)

        strategy.align_and_loop(align_to_next_candle)

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

