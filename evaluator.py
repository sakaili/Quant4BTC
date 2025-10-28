# evaluator.py
import numpy as np
import pandas as pd
from config import Config
from signals import SignalBuilder


class Evaluator:
    """评估历史行情下策略持仓表现的通用工具。"""

    def __init__(self, cfg: Config):
        """初始化评估器并复用现有的信号生成器。"""
        self.cfg = cfg
        self._sb = SignalBuilder(cfg)

    @staticmethod
    def _apply_cost_on_pnl(pos: np.ndarray, close: np.ndarray, fee_rate: float, slippage: float, turnover_penalty: float) -> np.ndarray:
        """按照成交成本、滑点和换手惩罚扣减收益。"""
        ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()
        gross = pos * ret
        dpos = np.abs(np.diff(np.concatenate([[pos[0]], pos])))
        unit_cost = (fee_rate + slippage) * 2.0
        cost = unit_cost * dpos
        cost = np.pad(cost, (0, len(gross) - len(cost)))
        if turnover_penalty > 0:
            cost += turnover_penalty * dpos
        return gross - cost

    @staticmethod
    def _sortino(x: np.ndarray, ann: float = 252.0) -> float:
        """按年化尺度计算 Sortino 比率。"""
        downside = x.copy()
        downside[downside > 0] = 0.0
        ds = np.sqrt((downside ** 2).mean()) + 1e-12
        mu = x.mean()
        return float((mu * ann) / ds)

    def evaluate_factor(self, df, st):
        """保持兼容的 SuperTrend 评估接口。"""
        sig = self._sb.build(df, st)
        pos = sig.astype(float)
        close = df['Close'].to_numpy()
        return self.evaluate_positions(close, pos)

    def evaluate_positions(self, close: np.ndarray, pos: np.ndarray):
        """对任意持仓序列计算关键绩效指标。"""
        strat_ret = self._apply_cost_on_pnl(pos, close, self.cfg.fee_rate, self.cfg.slippage_rate, self.cfg.turnover_penalty)
        eq = np.cumprod(1.0 + strat_ret)
        if len(eq) == 0:
            mdd, last_eq = 0.0, 1.0
        else:
            peak = np.maximum.accumulate(eq)
            drawdown = (peak - eq) / (peak + 1e-12)
            mdd = float(drawdown.max())
            last_eq = float(eq[-1])
        ann = 252.0
        sharpe = float((np.mean(strat_ret) * ann) / (np.std(strat_ret) * np.sqrt(ann) + 1e-12))
        cagr = float(last_eq ** (ann / max(len(eq), 1)) - 1.0) if len(eq) > 0 else 0.0
        turns = float(np.sum(np.abs(np.diff(pos))))
        sortino = self._sortino(strat_ret, ann=np.sqrt(ann))
        calmar = (cagr / (mdd + 1e-12)) if mdd > 0 else cagr
        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'mdd': mdd,
            'last_eq': last_eq,
            'turns': turns,
            'sortino': sortino,
            'calmar': calmar,
        }

    @staticmethod
    def score(m: dict) -> float:
        """将多项指标压缩为单一综合得分。"""
        return (1.0 * m['sharpe']) + (0.7 * m['sortino']) + (0.5 * m['calmar']) - (0.7 * m['mdd']) - (0.001 * m['turns'])
