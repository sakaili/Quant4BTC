# indicators.py
import numpy as np
import pandas as pd
from typing import Optional
from .config import Config


class IndicatorEngine:
    """集中实现 ATR 与 SuperTrend 相关的指标计算。"""

    def __init__(self, cfg: Config):
        """保留配置引用，允许在计算过程中读取参数。"""
        self.cfg = cfg

    def compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算平滑 ATR 并补充 hl2, Close 等列，返回去掉缺失值后的 DataFrame。"""
        df = df.copy()
        if 'hl2' not in df.columns:
            df['hl2'] = (df['High'] + df['Low']) / 2.0
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            (df['High'] - df['Low']).abs(),
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        n = max(2, int(self.cfg.atr_length))
        df['atr'] = tr.ewm(alpha=1 / n, adjust=False).mean()
        return df.dropna(subset=['atr', 'hl2', 'Close'])

    def compute_ema(self, series: pd.Series, length: int) -> pd.Series:
        """����ָ���ȡ� EMA ϵ�С�"""
        span = max(1, int(length))
        return series.ewm(span=span, adjust=False).mean()

    def compute_macd(
        self,
        df: pd.DataFrame,
        fast_length: Optional[int] = None,
        slow_length: Optional[int] = None,
        signal_length: Optional[int] = None,
    ) -> pd.DataFrame:
        """���� MACD �� DIF/DEA/Histogram ��"""
        if df.empty:
            return df.copy()

        close = df['Close']
        fast = max(1, int(fast_length if fast_length is not None else self.cfg.macd_fast_length))
        slow = max(1, int(slow_length if slow_length is not None else self.cfg.macd_slow_length))
        signal = max(1, int(signal_length if signal_length is not None else self.cfg.macd_signal_length))

        ema_fast = self.compute_ema(close, fast)
        ema_slow = self.compute_ema(close, slow)
        dif = ema_fast - ema_slow
        dea = self.compute_ema(dif, signal)
        hist = dif - dea

        macd_df = df.copy()
        macd_df['DIF'] = dif
        macd_df['DEA'] = dea
        macd_df['Histogram'] = hist
        return macd_df

    def compute_supertrend(self, df_atr: pd.DataFrame, factor: float) -> dict:
        """基于 ATR 平台生成 SuperTrend 上下轨、趋势状态等结果。"""
        c = df_atr["Close"].to_numpy()
        hl2 = df_atr["hl2"].to_numpy()
        atr = df_atr["atr"].to_numpy()
        n = len(c)
        trend = np.zeros(n, dtype=int)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        output = np.zeros(n, dtype=float)
        up_basic0 = hl2[0] + factor * atr[0]
        dn_basic0 = hl2[0] - factor * atr[0]
        trend[0] = 1 if c[0] > hl2[0] else 0
        upper[0] = up_basic0
        lower[0] = dn_basic0
        output[0] = lower[0] if trend[0] == 1 else upper[0]
        for i in range(1, n):
            up_basic = hl2[i] + factor * atr[i]
            dn_basic = hl2[i] - factor * atr[i]
            if c[i] > upper[i - 1]:
                trend[i] = 1
            elif c[i] < lower[i - 1]:
                trend[i] = 0
            else:
                trend[i] = trend[i - 1]
            if trend[i] == 1:
                upper[i] = min(up_basic, upper[i - 1])
                lower[i] = dn_basic
            else:
                upper[i] = up_basic
                lower[i] = max(dn_basic, lower[i - 1])
            output[i] = lower[i] if trend[i] == 1 else upper[i]
        return {"trend": trend, "upper": upper, "lower": lower, "output": output, "factor": factor}
