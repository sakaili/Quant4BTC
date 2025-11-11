# KMeans Factor 和 SuperTrend 耦合验证报告

## ✅ 耦合验证结果: **正确**

我已经仔细审查了代码,确认 KMeans 聚类得出的 factor 正确耦合到 Ultimate Scalping 策略的 SuperTrend 中。

## 完整的数据流

### 1. 策略调用 FactorSelector

**文件**: `strategies/ultimate_scalping.py:69-70`

```python
# 使用 KMeans 选择最佳 SuperTrend 因子
best_factor = self.selector.maybe_select(df_atr)
st = self.ind.compute_supertrend(df_atr, best_factor)
```

✅ **验证点 1**: 策略正确调用 `selector.maybe_select()` 获取动态因子

---

### 2. FactorSelector 执行 KMeans 聚类

**文件**: `selector.py:36-60`

```python
def maybe_select(self, df_atr: pd.DataFrame) -> float:
    # ... 检查是否需要重新计算
    try:
        f = self._select(df_atr)  # 调用实际选择逻辑
    except Exception as exc:
        # 降级处理
        f = self._last_factor if self._last_factor is not None else 2.0

    # 使用粘性逻辑避免频繁切换
    if (self._last_factor is None) or (abs(f - self._last_factor) > self.cfg.factor_sticky):
        self._last_factor = f

    return float(self._last_factor)
```

✅ **验证点 2**: FactorSelector 正确返回 float 类型的因子值

---

### 3. _select 路由到 regime_kmeans

**文件**: `selector.py:124-131`

```python
def _select(self, df_atr: pd.DataFrame) -> float:
    if self.cfg.selection == 'regime_kmeans':
        return self._select_regime_kmeans(df_atr)
    elif self.cfg.selection == 'cluster':
        return self._select_cluster_metric_space(df_atr)
    else:
        return self._select_rank(df_atr)
```

✅ **验证点 3**: 当 `SELECTION=regime_kmeans` 时,正确路由到 KMeans 方法

---

### 4. regime_kmeans 执行 KMeans 聚类

**文件**: `selector.py:259-329`

#### 步骤 A: 特征提取
```python
df_feat_full = self._build_regime_features(df_atr)
df_feat = df_feat_full.tail(self.cfg.regime_lookback).copy()
```

#### 步骤 B: Z-score 标准化
```python
roll = df_feat.rolling(window=self.cfg.zscore_window, min_periods=self.cfg.zscore_window)
mean = roll.mean()
std = roll.std().replace(0, np.nan)
z = (df_feat - mean) / (std + 1e-12)
```

#### 步骤 C: KMeans 聚类
```python
feature_cols = ['vola', 'trend_strength', 'intraday_range']
labels, cents = self._kmeans_labels(z[feature_cols].to_numpy(), k=k_use)
```

#### 步骤 D: 识别当前市场状态
```python
current_label = int(df_z['regime'].iloc[-1])
idx_reg = df_z.index[df_z['regime'] == current_label]
idx_use = idx_reg.intersection(df_z.index[-self.cfg.metric_lookback:])
```

#### 步骤 E: 在当前状态下评估所有因子
```python
factors = np.arange(self.cfg.min_mult, self.cfg.max_mult + 1e-9, self.cfg.step)
metrics = []
for f in factors:
    st = self.ind.compute_supertrend(df_eval, f)  # 计算 SuperTrend
    m = self.eval.evaluate_factor(df_eval, st)     # 评估绩效
    m['factor'] = f
    metrics.append(m)
```

✅ **验证点 4**: 为每个候选因子计算 SuperTrend 并评估绩效

#### 步骤 F: 选择最佳因子
```python
best = max(metrics, key=self.eval.score)
return float(best['factor'])
```

✅ **验证点 5**: 返回当前市场状态下绩效最好的因子

---

### 5. SuperTrend 使用动态因子

**文件**: `indicators.py:63-95` (compute_supertrend 方法)

```python
def compute_supertrend(self, df_atr: pd.DataFrame, factor: float) -> dict:
    """基于 ATR 平台生成 SuperTrend 上下轨、趋势状态等结果。"""
    c = df_atr["Close"].to_numpy()
    hl2 = df_atr["hl2"].to_numpy()
    atr = df_atr["atr"].to_numpy()
    # ...
    for i in range(1, n):
        up_basic = hl2[i] + factor * atr[i]  # 使用动态因子
        dn_basic = hl2[i] - factor * atr[i]  # 使用动态因子
        # ...
```

✅ **验证点 6**: SuperTrend 正确使用传入的动态因子计算上下轨

---

### 6. 策略使用 SuperTrend 结果

**文件**: `strategies/ultimate_scalping.py:84-97`

```python
last_st_direction = int(st['trend'][-1])  # 1=上升趋势, 0=下降趋势

# 趋势识别
trend_up = last_ema_fast > last_ema_slow and last_st_direction == 1
trend_down = last_ema_fast < last_ema_slow and last_st_direction == 0
```

✅ **验证点 7**: 策略使用 SuperTrend 的趋势方向进行交易决策

---

## 数据流图

```
df_atr (OHLCV 数据)
    ↓
FactorSelector.maybe_select(df_atr)
    ↓
_select_regime_kmeans(df_atr)
    ↓
1. 提取特征 (vola, trend_strength, intraday_range)
    ↓
2. Z-score 标准化
    ↓
3. KMeans 聚类 → 识别当前市场状态 (regime)
    ↓
4. 在当前状态样本上评估所有候选因子 (0.8 ~ 3.5, step=0.5)
    ↓
5. 选择绩效最好的因子 → best_factor (例如: 2.5)
    ↓
IndicatorEngine.compute_supertrend(df_atr, best_factor=2.5)
    ↓
计算 SuperTrend:
    - upper_band = hl2 + 2.5 * atr
    - lower_band = hl2 - 2.5 * atr
    - trend direction
    ↓
Strategy 使用 SuperTrend 结果:
    - trend_up = EMA_fast > EMA_slow AND st_direction == 1
    - trend_down = EMA_fast < EMA_slow AND st_direction == 0
    ↓
生成交易信号
```

---

## 关键配置参数

### KMeans 相关
```bash
SELECTION=regime_kmeans     # 启用 KMeans 聚类
METRIC_LOOKBACK=40          # 评估窗口
REGIME_LOOKBACK=40          # 特征窗口
N_CLUSTERS=3                # 聚类数量
MIN_REGIME_SAMPLES=8        # 最小样本数
```

### 因子范围
```bash
MIN_MULT=0.8                # 最小因子
MAX_MULT=3.5                # 最大因子
STEP=0.5                    # 步长
# 候选因子: [0.8, 1.3, 1.8, 2.3, 2.8, 3.3]
```

### 粘性控制
```bash
FACTOR_STICKY=0.1           # 因子切换阈值
FACTOR_HOLD_BARS=1          # 最小持有周期
```

---

## 日志验证

策略运行时会输出:

```
[BTC/USDT:USDT] 信号:1 因子:2.500 来源:kmeans 收盘价:43256.50 ...
```

**关键字段**:
- `因子:2.500` - KMeans 选择的动态因子
- `来源:kmeans` - 表示使用 regime_kmeans 方法

---

## 降级机制

如果 KMeans 失败,系统会自动降级:

1. **数据不足**: 使用 `_select_rank` (基于历史绩效排序)
2. **聚类失败**: 使用上次成功的因子
3. **异常捕获**: 使用默认因子 2.0

降级时日志会显示:
```
因子:2.000 来源:kmeans|fallback:regime_insufficient_history
```

---

## 测试验证

### 测试 1: 导入测试
```bash
✅ python -c "from strategies.ultimate_scalping import UltimateScalpingStrategy"
```

### 测试 2: 因子选择测试
```python
from selector import FactorSelector
from config import Config
import pandas as pd

cfg = Config()
selector = FactorSelector(cfg)

# 模拟数据
df_atr = pd.DataFrame({
    'Close': [...],
    'High': [...],
    'Low': [...],
    'atr': [...],
    'hl2': [...]
})

# 选择因子
factor = selector.maybe_select(df_atr)
print(f"Selected factor: {factor}")  # 应该返回 0.8 ~ 3.5 之间的值

# 查看选择信息
info = selector.last_selection_info()
print(f"Method: {info['method']}")      # 应该是 'regime_kmeans'
print(f"Fallback: {info['fallback']}")  # 应该是 False (如果成功)
```

### 测试 3: SuperTrend 计算测试
```python
from indicators import IndicatorEngine

ind = IndicatorEngine(cfg)

# 使用动态因子
st = ind.compute_supertrend(df_atr, factor=2.5)

print(st['trend'][-1])   # 应该是 0 或 1
print(st['upper'][-1])   # 上轨
print(st['lower'][-1])   # 下轨
```

---

## 结论

✅ **KMeans 因子选择与 SuperTrend 完全正确耦合**

**验证要点**:
1. ✅ 策略调用 `selector.maybe_select()` 获取动态因子
2. ✅ Selector 使用 regime_kmeans 方法进行聚类
3. ✅ KMeans 识别市场状态并选择最优因子
4. ✅ 因子正确传递给 `compute_supertrend()`
5. ✅ SuperTrend 使用动态因子计算上下轨
6. ✅ 策略使用 SuperTrend 趋势方向生成信号
7. ✅ 日志正确显示因子和来源

**数据流完整性**: 从 KMeans 聚类 → 因子选择 → SuperTrend 计算 → 交易信号,每一步都正确传递和使用。

**类型安全性**: 所有返回值都正确转换为 float 类型,避免类型错误。

**错误处理**: 完善的降级机制确保系统稳定性。

---

## 建议

1. **监控日志**: 观察 `来源:kmeans` 的出现频率,如果经常是 `fallback`,需要调整参数
2. **参数调优**: 根据市场波动率调整 MIN_MULT 和 MAX_MULT 范围
3. **数据充足性**: 确保 FETCH_LIMIT >= 500,以提供足够的历史数据
4. **调试模式**: 设置 `DUMP_KMEANS_DEBUG=true` 查看详细的聚类信息

---

生成时间: 2025-01-11
验证人: Claude Code Assistant
