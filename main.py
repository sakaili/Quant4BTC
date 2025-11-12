# main.py
import signal
import time
from dataclasses import replace
from datetime import datetime

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

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    strategy_cls = STRATEGY_REGISTRY.get(base_cfg.strategy_name.lower())
    if strategy_cls is None:
        raise ValueError(f"未知策略 '{base_cfg.strategy_name}'，请检查 STRATEGY_NAME 环境变量")

    symbols = base_cfg.symbol_list
    base_logger.info(f"初始化多品种交易: {symbols}")

    # 为每个品种创建独立的策略实例
    strategies = []
    for symbol in symbols:
        symbol_cfg = replace(base_cfg, symbol=symbol)
        symbol_logger = base_logger.getChild(symbol.replace('/', '_').replace(':', '_'))

        exch = ExchangeClient(symbol_cfg, symbol_logger)
        data_fetcher = DataFetcher(symbol_cfg, exch, symbol_logger)
        indicator_engine = IndicatorEngine(symbol_cfg)
        signal_builder = SignalBuilder(symbol_cfg)
        factor_selector = FactorSelector(symbol_cfg)
        order_executor = OrderExecutor(symbol_cfg, exch, symbol_logger)
        csv_logger = CsvLogger(base_cfg.csv_log_file, symbol_logger)  # 共享CSV文件
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
        strategies.append((symbol, strategy))
        symbol_logger.info(f"[{symbol}] 策略实例初始化完成")

    # 对齐到下一根K线
    tf_sec = DataFetcher.timeframe_seconds(base_cfg.timeframe)
    now = datetime.now()
    epoch = int(now.timestamp())
    remain = tf_sec - (epoch % tf_sec)
    sleep_secs = max(0, remain)
    base_logger.info(f"等待 {sleep_secs:.2f}s 对齐下一根 {base_cfg.timeframe} K 线")
    time.sleep(sleep_secs)

    # 主循环: 顺序执行各品种
    base_logger.info("开始多品种循环交易...")
    while True:
        cycle_start = time.time()

        # 在周期开始时读取一次净值，所有品种共享此快照
        try:
            # 使用第一个策略的executor读取净值（所有品种共享同一个账户）
            shared_equity = strategies[0][1].exec.account_equity()
            base_logger.info(f"📊 周期净值快照: {shared_equity:.2f} USDC (所有品种共享)")
        except Exception as e:
            base_logger.error(f"读取账户净值失败: {e}", exc_info=True)
            shared_equity = None  # 失败时传递None，策略会自行读取

        for symbol, strategy in strategies:
            try:
                base_logger.info(f"========== 执行 {symbol} ==========")
                strategy.run_once(equity=shared_equity)
            except Exception as e:
                base_logger.error(f"[{symbol}] 执行失败: {e}", exc_info=True)

        # 等待下一个周期
        elapsed = time.time() - cycle_start
        wait_time = max(0, tf_sec - elapsed)
        if wait_time > 0:
            base_logger.info(f"本周期耗时 {elapsed:.2f}s, 等待 {wait_time:.2f}s 进入下一周期")
            time.sleep(wait_time)


if __name__ == "__main__":
    main()
