# quant4BTC

面向 OKX 合约的 SuperTrend 自适应量化框架，围绕“一次拉取、一次评估、一次执行”的循环设计，便于扩展指标或切换执行环境。

## 架构与模块

| 模块 | 职责摘要 |
| --- | --- |
| config.py | 统一从环境变量加载运行参数，并提供日志初始化。 |
| exchange_client.py | 对 ccxt.okx 做最薄封装，负责连接、精度、杠杆与仓位模式。 |
| data.py | 拉取 K 线数据并整理成 pandas.DataFrame，剔除未收盘的尾巴。 |
| indicators.py | 计算 ATR 与 SuperTrend 指标。 |
| signals.py | 基于 SuperTrend 输出构造 long/short/flat 信号序列。 |
| evaluator.py | 评估不同因子的绩效，输出多项指标并打分。 |
| selector.py | 根据配置选择因子（遍历、绩效聚类或 regime 聚类）。 |
| position_reader.py | 屏蔽 OKX 多仓位模式差异，返回净仓及明细。 |
| order_executor.py | 统一封装下单与平仓语义，同时记录手续费等信息。 |
| csv_logger.py | 将每次执行结果追加到 CSV。 |
| unner.py | 主控流程：拉取数据、评估、执行、记录。 |
| main.py | 入口脚本，完成初始化并启动主循环。 |

### 主循环概览
1. DataFetcher 拉取行情，清理未收盘数据。  
2. IndicatorEngine 生成 ATR/SuperTrend。  
3. FactorSelector 决定当前使用的 SuperTrend 因子。  
4. SignalBuilder 生成目标仓位，PositionReader 读取当前仓位。  
5. OrderExecutor 触发必要的开平仓动作，并同步写入 CsvLogger。  
6. StrategyRunner.align_and_loop 通过对齐函数约束轮询节奏。

## 环境准备

### 依赖
- Python 3.10+
- pip install -r requirements.txt（若尚未创建，可手动安装：ccxt, pandas, 
umpy）

### 必需环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| CONTRACT_SYMBOL | BTC/USDT:USDT | 交易合约 |
| TIMEFRAME | 5m | K 线周期 |
| FETCH_LIMIT | 900 | 拉取历史根数 |
| CONTRACTS_PER_ORDER | 10 | 单次下单张数 |
| USE_DEMO | 	rue | 是否使用沙盒环境 |
| LOG_LEVEL | INFO | 日志等级 |
| OKX_API_KEY/SECRET/PASSWORD | 空 | OKX API 凭证 |
| LEVERAGE | 5 | 杠杆倍数 |
| MARGIN_MODE | cross | cross 或 isolated |
| POSITION_MODE | hedge | hedge 或 
et |
| CSV_LOG_FILE | 	rade_log.csv | 交易日志文件 |

更多指标、聚类与风控相关参数同样由 Config 读取，可根据需要覆写。

### 代理
若需走代理，设置 HTTP_PROXY / HTTPS_PROXY 即可，否则留空。

## 运行方式

`ash
python main.py
`

程序会：
1. 自动加载环境变量并初始化日志。  
2. 建立 OKX 连接，设置沙盒/实盘、杠杆和仓位模式。  
3. 执行一次完整流程，再通过 lign_to_next_candle 对齐下一根 K 线。

如需自定义对齐逻辑，可修改 main.py 中的 lign_to_next_candle。

## 日志与数据输出
- 控制台：标准日志输出（时间戳 + 等级 + 信息）。  
- CSV：CsvLogger 将每次动作写入 CSV_LOG_FILE 指定的文件，列包含信号、执行价、手续费、因子等。  
- 若需进一步分析，可直接在 Pandas 中读取该 CSV。

## 开发建议
- 保持模块单一职责，新增功能优先扩展现有类，而非引入交叉依赖。  
- 若引入新环境变量，务必在 README 和 Config 默认值中同步。  
- 调试时建议先设为 USE_DEMO=true，验证无误后再切换实盘。  
- 使用 python -m compileall . 或单元测试确保修改未破坏现有逻辑。
