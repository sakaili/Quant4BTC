# selector.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import Config
from indicators import IndicatorEngine
from evaluator import Evaluator
logger = logging.getLogger(__name__)
def _to_native(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return float(value) if isinstance(value, float) else value
class FactorSelector:
    """根据历史表现挑选当前应使用的 SuperTrend 因子。"""
    def __init__(self, cfg: Config):
        """初始化指标、评估器，并保存最近一次选择结果。"""
        self.cfg = cfg
        self.ind = IndicatorEngine(cfg)
        self.eval = Evaluator(cfg)
        self._last_factor = None
        self._last_bar_idx = None
        self._debug_counter = 0
        self._selection_meta: dict[str, Any] = {
            "method": "init",
            "factor": None,
            "reason": "not_evaluated",
            "details": {},
            "fallback": False,
            "reuse": False,
        }
    def maybe_select(self, df_atr: pd.DataFrame) -> float:
        """Select factor with minimal recalculation noise."""
        cur_idx = len(df_atr)
        force_recalc = bool(getattr(self.cfg, "force_factor_recalc", False))
        if (not force_recalc) and (not self._allow_recalc(cur_idx)):
            self._mark_reuse()
            return float(self._last_factor)
        try:
            f = self._select(df_atr)
        except Exception as exc:
            logger.warning("Factor selection failed, reuse last factor: %s", exc)
            f = self._last_factor if self._last_factor is not None else 2.0
            self._selection_meta = {
                "method": "exception_fallback",
                "factor": float(f),
                "reason": str(exc),
                "details": {},
                "fallback": True,
                "reuse": False,
            }
        if (self._last_factor is None) or (abs(f - self._last_factor) > self.cfg.factor_sticky):
            self._last_factor = f
        self._selection_meta["factor"] = float(self._last_factor)
        self._last_bar_idx = cur_idx
        return float(self._last_factor)
    def _allow_recalc(self, cur_idx: int) -> bool:
        """检查是否满足重新评估因子的最小间隔。"""
        return (self._last_factor is None) or (self._last_bar_idx is None) or ((cur_idx - self._last_bar_idx) >= self.cfg.factor_hold_bars)
    def last_selection_info(self) -> dict[str, Any]:
        """Return metadata about the latest selection decision."""
        return dict(self._selection_meta)
    def _mark_reuse(self) -> None:
        meta = dict(self._selection_meta)
        meta["reuse"] = True
        if self._last_factor is not None:
            meta["factor"] = float(self._last_factor)
        self._selection_meta = meta
    def _set_selection_meta(
        self,
        method: str,
        factor: float,
        *,
        reason: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        fallback: bool = False,
    ) -> None:
        self._selection_meta = {
            "method": method,
            "factor": float(factor),
            "reason": reason,
            "details": details or {},
            "fallback": bool(fallback),
            "reuse": False,
        }
    def _maybe_dump_kmeans_debug(
        self,
        stage: str,
        raw,
        scaled,
        labels: Optional[np.ndarray],
        centroids: Optional[np.ndarray],
        extra: Optional[dict] = None,
    ) -> None:
        if not getattr(self.cfg, "dump_kmeans_debug", False):
            return
        extra = extra or {}
        try:
            base = Path(getattr(self.cfg, "kmeans_debug_dir", "kmeans_debug")).expanduser()
            base.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            safe_stage = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in stage)
            self._debug_counter += 1
            filename = f"{safe_stage}_{self._debug_counter:04d}.json"
            payload = {
                "stage": stage,
                "timestamp": timestamp,
                "symbol": getattr(self.cfg, "symbol", ""),
                "selection": getattr(self.cfg, "selection", ""),
                "raw_features": None if raw is None else np.asarray(raw).tolist(),
                "scaled_features": None if scaled is None else np.asarray(scaled).tolist(),
                "labels": None if labels is None else np.asarray(labels, dtype=int).tolist(),
                "centroids": None if centroids is None else np.asarray(centroids).tolist(),
            }
            payload.update(extra)
            path = base / filename
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("无法写入KMeans调试文件: %s", exc)
    def _select(self, df_atr: pd.DataFrame) -> float:
        """��������ѡ������ѡ�β��ԡ�"""
        if self.cfg.selection == 'regime_kmeans':
            return self._select_regime_kmeans(df_atr)
        elif self.cfg.selection == 'cluster':
            return self._select_cluster_metric_space(df_atr)
        else:
            return self._select_rank(df_atr)
    def _select_rank(self, df_atr: pd.DataFrame, *, reason: Optional[str] = None, fallback: bool = False) -> float:
        """�������к�ѡ���ӣ������۷���ֱ��ȡ���š�"""
        factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
        metrics = []
        sample = df_atr.tail(self.cfg.metric_lookback)
        for f in factors:
            st = self.ind.compute_supertrend(sample, f)
            m = self.eval.evaluate_factor(sample, st)
            m['factor'] = f
            metrics.append(m)
        best = max(metrics, key=self.eval.score)
        self._set_selection_meta('rank', best['factor'], reason=reason, details=None, fallback=fallback)
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
        col_names = ['sharpe', 'neg_mdd', 'neg_turns']
        X = np.array([[m['sharpe'], -m['mdd'], -m['turns']] for m in metrics], dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        X_scaled = (X - mean) / std
        labels, cents_scaled = self._kmeans_labels(X_scaled, k=min(self.cfg.n_clusters, len(metrics)))
        metrics_payload = [{k: _to_native(v) for k, v in m.items()} for m in metrics]
        if labels.size == 0:
            best = max(metrics, key=self.eval.score)
            self._set_selection_meta(
                "cluster_rank",
                best["factor"],
                reason="empty_labels",
                details={"columns": col_names},
                fallback=True,
            )
            self._maybe_dump_kmeans_debug(
                stage="cluster_metrics",
                raw=X,
                scaled=X_scaled,
                labels=labels,
                centroids=None,
                extra={
                    "columns": col_names,
                    "reason": "empty_labels",
                    "metrics": metrics_payload,
                },
            )
            return float(best['factor'])
        unique_clusters = np.unique(labels)
        best_cluster = int(unique_clusters[0])
        best_score = float("-inf")
        cluster_scores = {}
        for cluster in unique_clusters:
            idx = labels == cluster
            cluster_vals = X[idx]
            cluster_score = cluster_vals[:, 0].mean() + 0.5 * cluster_vals[:, 1].mean() + 0.001 * cluster_vals[:, 2].mean()
            cluster_scores[int(cluster)] = float(cluster_score)
            if cluster_score > best_score:
                best_score = cluster_score
                best_cluster = int(cluster)
        cand = [m for m, lb in zip(metrics, labels) if lb == best_cluster]
        best = max(cand, key=self.eval.score)
        cents = (cents_scaled * std) + mean if cents_scaled is not None else None
        self._set_selection_meta(
            "cluster_kmeans",
            best["factor"],
            reason=None,
            details={
                "best_cluster": best_cluster,
                "cluster_scores": cluster_scores,
            },
            fallback=False,
        )
        self._maybe_dump_kmeans_debug(
            stage="cluster_metrics",
            raw=X,
            scaled=X_scaled,
            labels=labels,
            centroids=cents,
            extra={
                "columns": col_names,
                "metrics": metrics_payload,
                "factors": [float(m['factor']) for m in metrics],
                "best_cluster": best_cluster,
                "best_factor": float(best['factor']),
                "cluster_scores": cluster_scores,
            },
        )
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
            return self._select_rank(df_atr, reason="regime_insufficient_history", fallback=True)
        df_feat_full = self._build_regime_features(df_atr)
        df_feat = df_feat_full.tail(self.cfg.regime_lookback).copy()
        if len(df_feat) < max(self.cfg.zscore_window, self.cfg.min_regime_samples):
            return self._select_rank(df_atr, reason="regime_feature_window_short", fallback=True)
        roll = df_feat.rolling(window=self.cfg.zscore_window, min_periods=self.cfg.zscore_window)
        mean = roll.mean()
        std = roll.std().replace(0, np.nan)
        z = (df_feat - mean) / (std + 1e-12)
        z = z.dropna()
        if len(z) < self.cfg.min_regime_samples:
            return self._select_rank(df_atr, reason="regime_zscore_samples_short", fallback=True)
        k_use = max(2, min(self.cfg.n_clusters, len(z) // max(10, int(self.cfg.min_regime_samples / 2))))
        feature_cols = ['vola', 'trend_strength', 'intraday_range']
        labels, cents = self._kmeans_labels(z[feature_cols].to_numpy(), k=k_use)
        df_z = z.copy()
        df_z['regime'] = labels
        sizes = df_z['regime'].value_counts(normalize=True)
        valid_regs = sizes[sizes >= self.cfg.min_cluster_frac].index.tolist()
        current_label = int(df_z['regime'].iloc[-1])
        if current_label not in valid_regs:
            return self._select_rank(df_atr, reason="regime_invalid_label", fallback=True)
        idx_reg = df_z.index[df_z['regime'] == current_label]
        idx_use = idx_reg.intersection(df_z.index[-self.cfg.metric_lookback:])
        if len(idx_use) < self.cfg.min_regime_samples:
            return self._select_rank(df_atr, reason="regime_eval_samples_short", fallback=True)
        df_eval = df_atr.loc[idx_use]
        factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
        metrics = []
        for f in factors:
            st = self.ind.compute_supertrend(df_eval, f)
            m = self.eval.evaluate_factor(df_eval, st)
            m['factor'] = f
            metrics.append(m)
        best = max(metrics, key=self.eval.score)
        metrics_payload = [{k: _to_native(v) for k, v in m.items()} for m in metrics]
        raw_feat = df_feat.loc[df_z.index, feature_cols].to_numpy(dtype=float)
        scaled_feat = df_z[feature_cols].to_numpy(dtype=float)
        self._set_selection_meta(
            "regime_kmeans",
            best["factor"],
            reason=None,
            details={
                "current_label": current_label,
                "valid_clusters": [int(v) for v in valid_regs],
                "cluster_fractions": {int(k): float(v) for k, v in sizes.items()},
            },
            fallback=False,
        )
        self._maybe_dump_kmeans_debug(
            stage="regime_features",
            raw=raw_feat,
            scaled=scaled_feat,
            labels=df_z['regime'].to_numpy(),
            centroids=cents,
            extra={
                "columns": feature_cols,
                "index": [str(idx) for idx in df_z.index],
                "current_label": current_label,
                "valid_clusters": [int(v) for v in valid_regs],
                "selected_indices": [str(idx) for idx in idx_use],
                "best_factor": float(best['factor']),
                "cluster_fractions": {int(k): float(v) for k, v in sizes.items()},
                "metrics": metrics_payload,
            },
        )
        return float(best['factor'])
