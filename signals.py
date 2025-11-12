# signals.py
import numpy as np
import pandas as pd
from config import Config


class SignalBuilder:
    """根据 SuperTrend 输出构造最终交易信号。"""

    def __init__(self, cfg: Config):
        """保存配置以便读取确认次数、带宽等参数。"""
        self.cfg = cfg

    def build(self, df: pd.DataFrame, st: dict) -> np.ndarray:
        """根据价格与趋势线构建 long/short/falt 信号序列。"""
        c = df['Close'].to_numpy()
        out = np.asarray(st['output'], dtype=float)
        n = len(c)
        sig = np.zeros(n, dtype=int)

        def is_long(i):
            return c[i] > (out[i - 1] * (1.0 + self.cfg.band_eps))

        def is_short(i):
            return c[i] < (out[i - 1] * (1.0 - self.cfg.band_eps))

        if self.cfg.mode == 'long_flat':
            sig[0] = 1 if c[0] > out[0] else 0
            up = dn = 0
            for i in range(1, n):
                if is_long(i):
                    up, dn = up + 1, 0
                elif is_short(i):
                    dn, up = dn + 1, 0
                else:
                    up = max(0, up - 1)
                    dn = max(0, dn - 1)
                if sig[i - 1] == 0 and up >= self.cfg.signal_confirm:
                    sig[i] = 1
                    up = 0
                elif sig[i - 1] == 1 and dn >= self.cfg.signal_confirm:
                    sig[i] = 0
                    dn = 0
                else:
                    sig[i] = sig[i - 1]
        else:
            sig[0] = 1 if c[0] >= out[0] else -1
            up = dn = 0
            for i in range(1, n):
                if is_long(i):
                    up, dn = up + 1, 0
                elif is_short(i):
                    dn, up = dn + 1, 0
                else:
                    up = max(0, up - 1)
                    dn = max(0, dn - 1)
                if sig[i - 1] <= 0 and up >= self.cfg.signal_confirm:
                    sig[i] = 1
                    up = 0
                elif sig[i - 1] >= 0 and dn >= self.cfg.signal_confirm:
                    sig[i] = -1
                    dn = 0
                else:
                    sig[i] = sig[i - 1]
        return sig * 1
