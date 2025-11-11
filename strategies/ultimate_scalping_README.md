# Ultimate Scalping Strategy

## 策略简介

Ultimate Scalping Strategy 是一个综合了 EMA 趋势、RSI 动量和 SuperTrend 指标的多空交易策略。该策略从 TradingView 的策略移植而来,并集成了 KMeans 聚类自适应因子选择,专为中短线交易设计。

## 核心逻辑

### 技术指标

1. **快速 EMA (默认20周期)**: 捕捉短期价格趋势
2. **慢速 EMA (默认50周期)**: 确认中期趋势方向
3. **RSI (默认14周期)**: 评估市场动量和超买超卖状态
4. **SuperTrend (KMeans 自适应)**: 动态调整因子,辅助趋势确认和止损设置
5. **KMeans 聚类**: 自动识别市场状态并选择最佳 SuperTrend 参数

### 做多信号

主信号:
- 快速EMA > 慢速EMA (上升趋势)
- SuperTrend方向为上升 (direction = 1)
- RSI > 55 (动量偏多)
- 价格向上穿越快速EMA

重入信号 (回调加仓):
- 处于上升趋势中
- 价格仍在快速EMA之上
- RSI 在 50-70 区间 (动量健康但未超买)

### 做空信号

主信号:
- 快速EMA < 慢速EMA (下降趋势)
- SuperTrend方向为下降 (direction = 0)
- RSI < 45 (动量偏空)
- 价格向下穿越快速EMA

重入信号 (反弹加仓):
- 处于下降趋势中
- 价格仍在快速EMA之下
- RSI 在 30-50 区间 (动量疲弱但未超卖)

### 平仓机制

1. **反向信号平仓**: 当配置 `SCALPING_REVERSAL_EXIT=true` 时,出现反向信号自动平仓
2. **止损单**: 设置固定百分比止损 (默认1.5%)
3. **止盈**: 通过反向信号实现止盈 (默认目标3%),当前不支持限价止盈单

## 配置参数

### 环境变量配置

```bash
# 选择策略
STRATEGY_NAME=ultimate_scalping

# 市场配置
CONTRACT_SYMBOL=BTC/USDT:USDT
TIMEFRAME=30m                # 推荐30分钟周期 (可选: 15m, 1h)
FETCH_LIMIT=900

# EMA 参数
EMA_FAST_LENGTH=20          # 快速EMA周期
EMA_SLOW_LENGTH=50          # 慢速EMA周期

# RSI 参数
RSI_LENGTH=14               # RSI周期

# SuperTrend 参数 (基准值,会被 KMeans 动态调整)
SUPERTREND_MULT=3.0         # SuperTrend基准倍数
ATR_LENGTH=5                # ATR周期

# KMeans 因子选择参数
SELECTION=regime_kmeans     # 使用 KMeans 聚类选择因子
METRIC_LOOKBACK=40          # 回看窗口
MIN_MULT=0.8                # 最小因子倍数
MAX_MULT=3.5                # 最大因子倍数
STEP=0.5                    # 因子步长
N_CLUSTERS=3                # 聚类数量

# 止盈止损
SCALPING_TAKE_PROFIT_PCT=3.0   # 止盈百分比
SCALPING_STOP_LOSS_PCT=1.5     # 止损百分比
SCALPING_REVERSAL_EXIT=true    # 反向信号平仓

# 基础配置
FIXED_ORDER_SIZE=3.0        # 固定仓位大小
USE_DEMO=true               # 是否使用测试网
```

### 配置示例

#### 30分钟周期 (推荐)
```bash
TIMEFRAME=30m
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
RSI_LENGTH=14
METRIC_LOOKBACK=40
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
```

#### 15分钟周期 (较激进)
```bash
TIMEFRAME=15m
EMA_FAST_LENGTH=12
EMA_SLOW_LENGTH=26
RSI_LENGTH=10
METRIC_LOOKBACK=60
SCALPING_TAKE_PROFIT_PCT=5.0
SCALPING_STOP_LOSS_PCT=2.0
```

#### 1小时周期 (保守型)
```bash
TIMEFRAME=1h
EMA_FAST_LENGTH=30
EMA_SLOW_LENGTH=100
RSI_LENGTH=21
METRIC_LOOKBACK=30
SCALPING_TAKE_PROFIT_PCT=2.0
SCALPING_STOP_LOSS_PCT=1.0
```

## 使用方法

### 1. 设置环境变量

创建或修改 `.env` 文件:

```bash
STRATEGY_NAME=ultimate_scalping
CONTRACT_SYMBOL=BTC/USDT:USDT
TIMEFRAME=30m
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
RSI_LENGTH=14
SUPERTREND_MULT=3.0
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
SCALPING_REVERSAL_EXIT=true
FIXED_ORDER_SIZE=3.0
```

### 2. 运行策略

```bash
python main.py
```

## 策略特点

### 优势
1. **多重确认**: 结合趋势、动量和价格行为,减少假信号
2. **双向交易**: 支持多空双向操作,适合震荡和趋势市场
3. **回调重入**: 在趋势中捕捉回调机会,提高资金利用率
4. **风险控制**: 内置止损机制和反向信号平仓保护

### 适用场景
- 1分钟到5分钟级别的短线交易
- 流动性好、波动适中的加密货币
- 趋势明确或震荡区间内的市场环境

### 风险提示
1. 剥头皮策略对交易费用敏感,需要考虑手续费成本
2. 需要较低延迟的网络连接
3. 在趋势反转期间可能产生连续亏损
4. 建议先在测试网验证参数后再实盘使用

## 技术说明

### 依赖模块
- `indicators.py`: 提供 EMA、RSI、SuperTrend 计算
- `order_executor.py`: 执行市价单和止损单
- `position_reader.py`: 读取当前仓位状态
- `data.py`: 获取 OHLCV 数据

### 信号去重
策略内置信号去重机制,相同信号不会重复执行,避免频繁交易。

### 止盈实现
由于当前 OrderExecutor 不支持限价止盈单,止盈通过以下方式实现:
- 开启 `SCALPING_REVERSAL_EXIT` 时,反向信号触发平仓
- 手动监控或依赖策略的信号反转

未来版本将支持限价止盈单功能。

## 性能调优建议

1. **参数优化**: 根据交易品种的波动率调整 EMA 和 RSI 周期
2. **止盈止损比例**: 建议保持至少 1:2 的风险回报比
3. **SuperTrend倍数**: 波动大的币种可以适当提高倍数
4. **回测验证**: 使用历史数据回测不同参数组合的表现

## 监控指标

策略运行时会记录以下关键信息:
- 当前信号 (1=多头, -1=空头, 0=平仓)
- 价格、RSI、EMA 值
- 仓位状态和成交信息
- 止盈止损价位

日志示例:
```
[BTC/USDT:USDT] 信号:1 收盘价:43256.5000 RSI:62.34 EMA_fast:43200.12 EMA_slow:43150.45 ST_trend:1
Long SL=42608.65 (1.50%) | TP目标=44554.20 (3.00%)
```

## 常见问题

### Q: 为什么没有止盈单?
A: 当前版本的 OrderExecutor 只支持止损单,止盈通过反向信号平仓实现。未来版本会添加限价止盈单支持。

### Q: 如何调整仓位大小?
A: 通过 `FIXED_ORDER_SIZE` 环境变量设置固定仓位,或修改 config.py 中的风险管理逻辑。

### Q: 可以在哪些交易所使用?
A: 目前支持币安合约 (USDM),其他交易所需要修改 ExchangeClient 适配。

### Q: 如何避免频繁交易?
A: 可以增加 EMA 周期长度,或调整 RSI 阈值 (例如多头 > 60, 空头 < 40)。

## 版本历史

- **v1.0** (2025-01): 初始版本,移植自 TradingView 策略
  - 支持 EMA + RSI + SuperTrend 综合信号
  - 实现反向信号平仓
  - 添加回调重入逻辑
  - 集成止损单管理

## 相关文档

- [SuperTrend Strategy](supertrend.py): 基于 SuperTrend 的自适应策略
- [MACD Triple Filter](macd_triple_filter.py): 基于 MACD 的多重过滤策略
- [配置说明](../config.py): 完整的配置参数文档
