# data.py
import pandas as pd
from datetime import datetime
from config import Config


class DataFetcher:
    """负责从交易所取数并整理成 pandas DataFrame。"""

    def __init__(self, cfg: Config, exch, logger):
        """保存依赖实例，便于后续拉取行情数据。"""
        self.cfg = cfg
        self.exch = exch
        self.logger = logger

    def fetch_ohlcv_df(self) -> pd.DataFrame:
        """抓取原始 K 线并转换成按时间索引的 DataFrame。"""
        try:
            raw = self.exch.fetch_ohlcv(self.cfg.fetch_limit)
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def timeframe_seconds(tf: str) -> int:
        """把 ccxt 的 timeframe 字符串转换为秒。"""
        tf = str(tf).lower().strip()
        if tf.endswith('m'):
            return int(tf[:-1]) * 60
        if tf.endswith('h'):
            return int(tf[:-1]) * 3600
        if tf.endswith('d'):
            return int(tf[:-1]) * 86400
        return 60

    def drop_unclosed_tail(self, df: pd.DataFrame) -> pd.DataFrame:
        """在实时运行中抛弃尚未收盘的最后一根 K 线。"""
        tf_sec = self.timeframe_seconds(self.cfg.timeframe)
        try:
            if len(df) >= 2:
                last_ts = df.index[-1].to_pydatetime()
                if (datetime.now(datetime.UTC) - last_ts).total_seconds() < (tf_sec - 1):
                    return df.iloc[:-1]
        except Exception:
            pass
        return df
