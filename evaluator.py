# evaluator.py
import numpy as np
import pandas as pd
from config import Config
from signals import SignalBuilder


class Evaluator:
    """负责根据历史数据评估给定 SuperTrend 因子的有效性。"""

    def __init__(self, cfg: Config):
        """准备信号构造器与配置引用。"""
        self.cfg = cfg
        self._sb = SignalBuilder(cfg)

    @staticmethod
    def _apply_cost_on_pnl(pos: np.ndarray, close: np.ndarray, fee_rate: float, slippage: float, turnover_penalty: float) -> np.ndarray:
        """在理论收益上施加手续费、滑点与换手惩罚。"""
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
        """计算 Sortino 比例以强调下行波动。"""
        downside = x.copy()
        downside[downside > 0] = 0.0
        ds = np.sqrt((downside ** 2).mean()) + 1e-12
        mu = x.mean()
        return float((mu * ann) / ds)

    def evaluate_factor(self, df, st):
        """回测指定因子，输出品味分指标（年化、回撤、换手等）。"""
        sig = self._sb.build(df, st)
        pos = sig.astype(float)
        close = df['Close'].to_numpy()
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
        return {'cagr': cagr, 'sharpe': sharpe, 'mdd': mdd, 'last_eq': last_eq, 'turns': turns, 'sortino': sortino, 'calmar': calmar}

    @staticmethod
    def score(m: dict) -> float:
        """把多个指标压缩成单一分数，便于挑选最佳因子。"""
        return (1.0 * m['sharpe']) + (0.7 * m['sortino']) + (0.5 * m['calmar']) - (0.7 * m['mdd']) - (0.001 * m['turns'])
