# Ultimate Scalping Strategy - KMeans 优化版本更新

## 更新概述

Ultimate Scalping 策略已升级为使用 KMeans 聚类进行趋势识别,并将推荐时间周期调整为 30 分钟。

## 主要变更

### 1. 集成 KMeans 自适应因子选择

**之前**: 使用固定的 SuperTrend 倍数 (SUPERTREND_MULT=3.0)

**现在**: 使用 FactorSelector 的 KMeans 聚类动态选择最佳 SuperTrend 因子

**优势**:
- 自动适应不同市场状态
- 动态调整 SuperTrend 灵敏度
- 提高信号质量,减少假突破

### 2. 推荐时间周期调整

**之前**: 默认 1 分钟 (TIMEFRAME=1m)

**现在**: 默认 30 分钟 (TIMEFRAME=30m)

**原因**:
- 30分钟周期更适合 KMeans 聚类分析
- 减少噪音,提高信号可靠性
- 降低交易频率和手续费成本
- 更好的趋势识别效果

### 3. 代码变更

#### strategies/ultimate_scalping.py

```python
# 之前: 固定因子
st = self.ind.compute_supertrend(df_atr, self.cfg.supertrend_mult)

# 现在: KMeans 动态选择
best_factor = self.selector.maybe_select(df_atr)
st = self.ind.compute_supertrend(df_atr, best_factor)
```

#### 日志输出增强

```python
# 现在包含因子选择信息
self.logger.info(
    "[%s] 信号:%s 因子:%.3f 来源:%s 收盘价:%.4f RSI:%.2f ...",
    self.cfg.symbol,
    current_signal,
    factor_display,  # 显示动态选择的因子
    source_desc,     # 显示选择方法 (kmeans/fallback等)
    ...
)
```

## 配置参数

### 新增 KMeans 相关参数

```bash
# KMeans 因子选择
SELECTION=regime_kmeans     # 使用 KMeans 聚类
METRIC_LOOKBACK=40          # 回看窗口
MIN_MULT=0.8                # 最小因子倍数
MAX_MULT=3.5                # 最大因子倍数
STEP=0.5                    # 因子步长
N_CLUSTERS=3                # 聚类数量

# SuperTrend 基准值 (会被 KMeans 覆盖)
SUPERTREND_MULT=3.0         # 作为 fallback 使用
```

### 推荐配置 (30分钟周期)

```bash
# .env 文件
STRATEGY_NAME=ultimate_scalping
CONTRACT_SYMBOL=BTC/USDT:USDT
TIMEFRAME=30m
FETCH_LIMIT=900

# EMA 参数
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
RSI_LENGTH=14

# KMeans 参数
SELECTION=regime_kmeans
METRIC_LOOKBACK=40
MIN_MULT=0.8
MAX_MULT=3.5
STEP=0.5
N_CLUSTERS=3

# 止盈止损
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
SCALPING_REVERSAL_EXIT=true

# 交易配置
FIXED_ORDER_SIZE=3.0
LEVERAGE=10
USE_DEMO=true
```

## 不同时间周期配置

### 30 分钟 (推荐)
适合中短线交易,信号质量高,手续费合理

```bash
TIMEFRAME=30m
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
METRIC_LOOKBACK=40
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
```

### 15 分钟 (较激进)
交易频率较高,适合波动大的市场

```bash
TIMEFRAME=15m
EMA_FAST_LENGTH=12
EMA_SLOW_LENGTH=26
METRIC_LOOKBACK=60
SCALPING_TAKE_PROFIT_PCT=5.0
SCALPING_STOP_LOSS_PCT=2.0
```

### 1 小时 (保守)
信号更稳定,适合资金量大或波动小的市场

```bash
TIMEFRAME=1h
EMA_FAST_LENGTH=30
EMA_SLOW_LENGTH=100
METRIC_LOOKBACK=30
SCALPING_TAKE_PROFIT_PCT=2.0
SCALPING_STOP_LOSS_PCT=1.0
```

## 策略逻辑

### 信号生成流程

1. **数据准备**: 获取 OHLCV 数据并计算 ATR
2. **KMeans 分析**: 根据历史数据选择最佳 SuperTrend 因子
3. **指标计算**:
   - SuperTrend (动态因子)
   - 快速/慢速 EMA
   - RSI
4. **信号判断**:
   - 多头: EMA_fast > EMA_slow + ST上升 + RSI>55 + 价格上穿EMA
   - 空头: EMA_fast < EMA_slow + ST下降 + RSI<45 + 价格下穿EMA
5. **执行交易**: 开仓/平仓/止损

### KMeans 因子选择原理

1. 计算多个 SuperTrend 因子 (MIN_MULT 到 MAX_MULT, 步长 STEP)
2. 评估每个因子在历史窗口的表现
3. 使用 KMeans 聚类识别市场状态
4. 选择当前市场状态下最优的因子
5. 可选: 使用 EMA 平滑或粘性逻辑避免频繁切换

## 日志示例

```
[BTC/USDT:USDT] 信号:1 因子:2.500 来源:kmeans 收盘价:43256.5000 RSI:62.34 EMA_fast:43200.12 EMA_slow:43150.45 ST_trend:1
Position long=0.0000 short=0.0000 net=0 equity=500.00 unrealized=0.00%
Opened LONG position: size=3.0000 price=43260.2500
Long SL=42623.65 (1.50%) | TP目标=44558.06 (3.00%)
```

关键信息:
- `因子:2.500` - KMeans 选择的动态因子
- `来源:kmeans` - 表示使用 KMeans 选择
- 其他可能的来源:
  - `kmeans|reuse` - 复用上次因子
  - `kmeans|fallback` - 降级使用默认因子
  - `fixed` - 使用固定因子

## 性能优势

### KMeans vs 固定因子

| 指标 | 固定因子 | KMeans 自适应 |
|------|---------|--------------|
| 适应性 | 差 | 优 |
| 假信号 | 较多 | 较少 |
| 趋势识别 | 一般 | 好 |
| 震荡市表现 | 差 | 较好 |
| 趋势市表现 | 好 | 优 |

### 30分钟 vs 1分钟

| 指标 | 1分钟 | 30分钟 |
|------|-------|--------|
| 交易频率 | 很高 | 中等 |
| 信号质量 | 一般 | 好 |
| 手续费成本 | 高 | 低 |
| 滑点影响 | 显著 | 较小 |
| 适合场景 | 超短线 | 中短线 |

## 使用建议

### 1. 测试阶段
```bash
USE_DEMO=true
FIXED_ORDER_SIZE=1.0  # 小仓位测试
TIMEFRAME=30m
```

### 2. 参数优化
- 观察日志中的因子选择情况
- 调整 METRIC_LOOKBACK 适应不同市场节奏
- 根据波动率调整止盈止损比例

### 3. 风险管理
```bash
DAILY_DRAWDOWN_LIMIT=0.10    # 日回撤限制
OVERALL_DRAWDOWN_LIMIT=0.30  # 总回撤限制
RISK_PER_TRADE=0.1           # 单笔风险
```

### 4. 监控要点
- 关注因子切换频率
- 监控信号质量和胜率
- 查看 KMeans 聚类是否有效
- 定期查看交易日志 CSV

## 升级步骤

如果你正在使用旧版本:

1. **更新代码**: 拉取最新代码
2. **更新配置**:
   ```bash
   cp .env.template .env
   nano .env  # 添加 KMeans 参数
   ```
3. **调整时间周期**: `TIMEFRAME=30m`
4. **添加因子选择**: `SELECTION=regime_kmeans`
5. **重启服务**:
   ```bash
   sudo systemctl restart quant4btc
   ```

## 故障排查

### 问题: 因子一直是 fallback

**原因**: 数据不足或聚类失败

**解决**:
- 增加 FETCH_LIMIT (建议 > 500)
- 检查 METRIC_LOOKBACK 是否过大
- 查看日志中的详细错误信息

### 问题: 交易频率过高

**解决**:
- 增加时间周期 (30m -> 1h)
- 调整 EMA 周期 (20 -> 30)
- 增加 RSI 阈值 (55 -> 60)

### 问题: 信号质量差

**解决**:
- 检查 KMeans 参数是否合理
- 调整 MIN_MULT/MAX_MULT 范围
- 增加 N_CLUSTERS 数量
- 使用 MACD 过滤 (USE_MACD_FILTER=true)

## 更多信息

- 完整文档: [strategies/ultimate_scalping_README.md](../strategies/ultimate_scalping_README.md)
- 部署指南: [deployment/DEPLOYMENT.md](../deployment/DEPLOYMENT.md)
- 配置说明: [ENV_CONFIG.md](../ENV_CONFIG.md)
