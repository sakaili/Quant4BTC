# main.py
import signal
import time
from datetime import datetime
from config import Config, setup_logging
from exchange_client import ExchangeClient
from runner import StrategyRunner
from data import DataFetcher


def shutdown_handler(signum, frame):
    """捕捉关闭信号，保持退出过程干净。"""
    print("收到关闭信号")


def main():
    """读取配置，初始化组件后启动主循环。"""
    cfg = Config()
    logger = setup_logging(cfg.log_level)
    exch = ExchangeClient(cfg, logger)
    runner = StrategyRunner(cfg, exch, logger)

    def align_to_next_candle():
        """睡眠至下一根 K 线开始，确保节奏与数据对齐。"""
        tf_sec = DataFetcher.timeframe_seconds(cfg.timeframe)
        now = datetime.now()
        epoch = int(now.timestamp())
        remain = tf_sec - (epoch % tf_sec)
        sleep_secs = max(0, remain)
        logger.info(f"等待 {sleep_secs:.2f}s 对齐下一根 {cfg.timeframe} K 线")
        time.sleep(sleep_secs)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    runner.align_and_loop(align_to_next_candle)


if __name__ == "__main__":
    main()
