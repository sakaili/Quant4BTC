# selector.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import Config
from indicators import IndicatorEngine
from evaluator import Evaluator


class FactorSelector:
    """根据历史表现挑选当前应使用的 SuperTrend 因子。"""

    def __init__(self, cfg: Config):
        """初始化指标、评估器，并保存最近一次选择结果。"""
        self.cfg = cfg
        self.ind = IndicatorEngine(cfg)
        self.eval = Evaluator(cfg)
        self._last_factor = None
        self._last_bar_idx = None

    def maybe_select(self, df_atr: pd.DataFrame) -> float:
        """按需触发重新选参，减少频繁切换造成的噪音。"""
        cur_idx = len(df_atr)
        if not self._allow_recalc(cur_idx):
            return float(self._last_factor)
        try:
            f = self._select(df_atr)
        except Exception:
            f = self._last_factor if self._last_factor is not None else 2.0
        if (self._last_factor is None) or (abs(f - self._last_factor) > self.cfg.factor_sticky):
            self._last_factor = f
        self._last_bar_idx = cur_idx
        return float(self._last_factor)

    def _allow_recalc(self, cur_idx: int) -> bool:
        """检查是否满足重新评估因子的最小间隔。"""
        return (self._last_factor is None) or (self._last_bar_idx is None) or ((cur_idx - self._last_bar_idx) >= self.cfg.factor_hold_bars)

    def _select(self, df_atr: pd.DataFrame) -> float:
        """按照配置选择具体的选参策略。"""
        if self.cfg.selection == 'regime_kmeans':
            return self._select_regime_kmeans(df_atr)
        elif self.cfg.selection == 'cluster':
            return self._select_cluster_metric_space(df_atr)
        else:
            return self._select_rank(df_atr)

    def _select_rank(self, df_atr: pd.DataFrame) -> float:
        """遍历所有候选因子，按评价分数直接取最优。"""
        factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
        metrics = []
        sample = df_atr.tail(self.cfg.metric_lookback)
        for f in factors:
            st = self.ind.compute_supertrend(sample, f)
            m = self.eval.evaluate_factor(sample, st)
            m['factor'] = f
            metrics.append(m)
        best = max(metrics, key=self.eval.score)
        return float(best['factor'])

    @staticmethod
    def _kmeans_labels(X: np.ndarray, k: int = 3, iters: int = 200):
        """使用 sklearn 的 KMeans 聚类，返回簇标签和质心。"""
        X = np.asarray(X, dtype=float)
        n, _ = X.shape
        if n < k:
            return np.zeros(n, dtype=int), np.array([X.mean(axis=0)])
        model = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=iters,
            random_state=42,
        )
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_
        return labels, centroids

    def _select_cluster_metric_space(self, df_atr: pd.DataFrame) -> float:
        """基于 KMeans 聚类的绩效空间选参。"""
        factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
        sample = df_atr.tail(self.cfg.metric_lookback)
        metrics = []
        for f in factors:
            st = self.ind.compute_supertrend(sample, f)
            m = self.eval.evaluate_factor(sample, st)
            m['factor'] = f
            metrics.append(m)
        X = np.array([[m['sharpe'], -m['mdd'], -m['turns']] for m in metrics], dtype=float)
        labels, cents = self._kmeans_labels(X, k=min(self.cfg.n_clusters, len(metrics)))
        cent_scores = cents[:, 0] + 0.5 * cents[:, 1] + 0.001 * cents[:, 2]
        best_cluster = int(np.argmax(cent_scores))
        cand = [m for m, lb in zip(metrics, labels) if lb == best_cluster]
        best = max(cand, key=self.eval.score)
        return float(best['factor'])

    def _build_regime_features(self, df_atr: pd.DataFrame) -> pd.DataFrame:
        """构建用于 regime 聚类的特征序列。"""
        feat = pd.DataFrame(index=df_atr.index.copy())
        close = df_atr['Close']
        high = df_atr['High']
        low = df_atr['Low']
        vola = (df_atr['atr'] / (close + 1e-12)).clip(0, 1.0)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        trend_strength = (ema12 - ema50).abs() / (close + 1e-12)
        intraday_range = (high - low).abs() / (close + 1e-12)
        span = max(2, int(self.cfg.feature_ema))
        feat['vola'] = vola.ewm(span=span, adjust=False).mean()
        feat['trend_strength'] = trend_strength.ewm(span=span, adjust=False).mean()
        feat['intraday_range'] = intraday_range.ewm(span=span, adjust=False).mean()
        return feat.dropna()

    def _select_regime_kmeans(self, df_atr: pd.DataFrame) -> float:
        """结合 regime 聚类与局部绩效评估的选参策略。"""
        N = len(df_atr)
        if N < max(self.cfg.regime_lookback, self.cfg.metric_lookback) + 30:
            return self._select_rank(df_atr)
        df_feat_full = self._build_regime_features(df_atr)
        df_feat = df_feat_full.tail(self.cfg.regime_lookback).copy()
        if len(df_feat) < max(self.cfg.zscore_window, self.cfg.min_regime_samples):
            return self._select_rank(df_atr)
        roll = df_feat.rolling(window=self.cfg.zscore_window, min_periods=self.cfg.zscore_window)
        mean = roll.mean()
        std = roll.std().replace(0, np.nan)
        z = (df_feat - mean) / (std + 1e-12)
        z = z.dropna()
        if len(z) < self.cfg.min_regime_samples:
            return self._select_rank(df_atr)
        k_use = max(2, min(self.cfg.n_clusters, len(z) // max(10, int(self.cfg.min_regime_samples / 2))))
        labels, _ = self._kmeans_labels(z[['vola', 'trend_strength', 'intraday_range']].to_numpy(), k=k_use)
        df_z = z.copy()
        df_z['regime'] = labels
        sizes = df_z['regime'].value_counts(normalize=True)
        valid_regs = sizes[sizes >= self.cfg.min_cluster_frac].index.tolist()
        current_label = int(df_z['regime'].iloc[-1])
        if current_label not in valid_regs:
            return self._select_rank(df_atr)
        idx_reg = df_z.index[df_z['regime'] == current_label]
        idx_use = idx_reg.intersection(df_z.index[-self.cfg.metric_lookback:])
        if len(idx_use) < self.cfg.min_regime_samples:
            return self._select_rank(df_atr)
        df_eval = df_atr.loc[idx_use]
        factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
        metrics = []
        for f in factors:
            st = self.ind.compute_supertrend(df_eval, f)
            m = self.eval.evaluate_factor(df_eval, st)
            m['factor'] = f
            metrics.append(m)
        best = max(metrics, key=self.eval.score)
        return float(best['factor'])
