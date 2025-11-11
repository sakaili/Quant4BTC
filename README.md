# quant4BTC

多策略加密货币量化交易框架,支持币安 (Binance) 合约交易。围绕"一次拉取、一次评估、一次执行"的循环设计,便于扩展指标或切换执行环境。

## 特性

- 🚀 **多策略支持**: SuperTrend 自适应策略、Ultimate Scalping 策略
- 🔄 **灵活配置**: 支持 .env 文件、环境变量、系统配置
- 🐧 **Ubuntu 部署**: 完整的 systemd 服务和自动部署脚本
- 📊 **风险管理**: 内置止损、回撤控制、信号过滤
- 📈 **多指标组合**: SuperTrend、EMA、RSI、MACD 等
- 🔐 **安全**: 支持测试网、API 权限限制

## 快速开始

### 本地测试 (Windows/Mac/Linux)

```bash
# 1. 克隆项目
git clone <your-repo>
cd quant4BTC

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
cp .env.template .env
nano .env  # 编辑配置

# 4. 运行
python main.py
```

### Ubuntu 服务器部署

```bash
# 1. 上传代码
scp -r ./quant4BTC ubuntu@your-server:/home/ubuntu/

# 2. SSH 登录并运行部署脚本
ssh ubuntu@your-server
cd /home/ubuntu/quant4BTC
chmod +x deployment/deploy.sh
./deployment/deploy.sh

# 3. 查看状态
sudo systemctl status quant4btc
```

📖 **详细部署指南**: [deployment/QUICKSTART.md](deployment/QUICKSTART.md)

## 可用策略

### 1. Ultimate Scalping Strategy
结合 EMA 趋势、RSI 动量和 SuperTrend 的多空剥头皮策略。

**特点**:
- 多重信号确认 (EMA + RSI + SuperTrend)
- 支持回调重入
- 反向信号自动平仓
- 适合 1-5 分钟级别短线交易

**配置**:
```bash
STRATEGY_NAME=ultimate_scalping
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
RSI_LENGTH=14
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
```

📖 **详细说明**: [strategies/ultimate_scalping_README.md](strategies/ultimate_scalping_README.md)

### 2. SuperTrend Adaptive Strategy
基于 SuperTrend 指标的自适应策略,动态调整参数。

**特点**:
- 自适应因子选择 (KMeans 聚类)
- 市场状态识别
- MACD 过滤
- 风险管理和回撤控制

**配置**:
```bash
STRATEGY_NAME=supertrend
SELECTION=regime_kmeans
USE_MACD_FILTER=true
```

## 架构与模块

| 模块 | 职责摘要 |
| --- | --- |
| config.py | 统一化配置入口，读取环境变量并初始化日志 |
| exchange_client.py | 对 ccxt.okx 的轻量封装，处理账户、杠杆与持仓模式 |
| data.py | 抓取 K 线并整理为 pandas.DataFrame，移除未完成 K 线 |
| indicators.py | 计算 ATR 和 SuperTrend 指标 |
| signals.py | 基于 SuperTrend 生成 long/short/flat 信号 |
| evaluator.py | 评估不同参数表现并输出统计 |
| selector.py | 策略特有的因子选择逻辑（如 regime 聚类） |
| position_reader.py | 读取 OKX 持仓信息，返回净仓/对冲仓数据 |
| order_executor.py | 封装开平仓及风险控制下单 |
| csv_logger.py | 将每次执行结果追加写入 CSV |
| strategies/base.py | 策略基类，封装循环调度与风险管理 |
| strategies/supertrend.py | SuperTrend 策略实现 |
| strategies/ultimate_scalping.py | Ultimate Scalping 策略实现 |
| runner.py | 兼容旧接口，继续导出 SuperTrendStrategy |
| main.py | 程序入口，装配通用组件并运行所选策略 |
| env_loader.py | 环境变量加载器,支持 .env 文件 |

### 主循环概览
1. DataFetcher 拉取行情，清理未收盘数据。
2. IndicatorEngine 生成 ATR/SuperTrend/EMA/RSI。
3. FactorSelector 决定当前使用的 SuperTrend 因子 (仅 SuperTrend 策略)。
4. SignalBuilder 或策略内部逻辑生成交易信号。
5. OrderExecutor 触发必要的开平仓动作，并同步写入 CsvLogger。
6. Strategy.align_and_loop 通过对齐回调驱动主循环。

## 环境配置

### 方式 1: .env 文件 (推荐)

```bash
# 复制模板
cp .env.template .env

# 编辑配置
nano .env
```

### 方式 2: 环境变量

```bash
export STRATEGY_NAME=ultimate_scalping
export BINANCE_API_KEY=your_key
export BINANCE_SECRET=your_secret
export USE_DEMO=true
```

### 必需配置

```bash
# 策略选择
STRATEGY_NAME=ultimate_scalping  # 或 supertrend

# 交易对
CONTRACT_SYMBOL=BTC/USDT:USDT

# API 凭证 (从币安获取)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret

# 使用测试网 (强烈建议先测试)
USE_DEMO=true

# 仓位大小
FIXED_ORDER_SIZE=3.0
```

📖 **完整配置说明**: [ENV_CONFIG.md](ENV_CONFIG.md)

## 依赖安装

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
