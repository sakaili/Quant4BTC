# Ubuntu 部署环境变量配置总结

## 概述

项目现已支持灵活的环境变量配置方式,适合 Ubuntu ECS 服务器部署。

## 环境变量加载机制

### 加载优先级

1. **系统环境变量** (最高优先级)
2. **.env 文件** (推荐用于部署)
3. **代码默认值** (最低优先级)

### 自动加载

当导入 `config.py` 时,会自动加载 `.env` 文件(如果存在):

```python
from config import Config
cfg = Config()  # 自动从 .env 加载环境变量
```

## 配置方式

### 方式 1: 使用 .env 文件 (推荐)

```bash
# 1. 复制模板
cp .env.template .env

# 2. 编辑配置
nano .env

# 3. 运行程序
python main.py  # 自动加载 .env 文件
```

**优点**:
- 配置集中管理
- 易于版本控制 (不提交 .env,只提交 .env.template)
- 修改配置无需重启系统环境变量

### 方式 2: systemd 服务环境变量

在 `quant4btc.service` 文件中设置:

```ini
[Service]
Environment="STRATEGY_NAME=ultimate_scalping"
Environment="BINANCE_API_KEY=your_key"
Environment="USE_DEMO=true"
```

**优点**:
- 与服务绑定
- 系统级管理

### 方式 3: 系统环境变量

```bash
# 临时设置 (当前会话)
export STRATEGY_NAME=ultimate_scalping
export BINANCE_API_KEY=your_key

# 永久设置 (添加到 ~/.bashrc 或 ~/.profile)
echo 'export STRATEGY_NAME=ultimate_scalping' >> ~/.bashrc
source ~/.bashrc
```

**优点**:
- 系统级配置
- 所有应用可访问

## 已创建的文件

### 配置文件
1. **`.env.template`** - 环境变量模板,包含所有可配置参数
2. **`env_loader.py`** - 环境变量加载器,支持 python-dotenv

### 部署文件
1. **`deployment/quant4btc.service`** - systemd 服务配置
2. **`deployment/deploy.sh`** - 自动部署脚本
3. **`deployment/healthcheck.sh`** - 健康检查脚本
4. **`deployment/DEPLOYMENT.md`** - 详细部署文档
5. **`deployment/QUICKSTART.md`** - 快速开始指南

### 依赖文件
1. **`requirements.txt`** - Python 依赖包列表

## 配置修改

修改 `config.py` 以支持自动加载 .env:

```python
# config.py
import logging
import os
from dataclasses import dataclass

# Load environment variables from .env file if it exists
try:
    from env_loader import load_env
    # load_env() is called automatically when env_loader is imported
except ImportError:
    print("⚠ env_loader not found, using system environment variables only")

TRUE_SET = ("1", "true", "yes")

@dataclass(frozen=True)
class Config:
    # ... existing configuration ...
```

## 使用示例

### 开发环境 (Windows)

```bash
# 1. 复制模板
copy .env.template .env

# 2. 编辑 .env
notepad .env

# 3. 运行
python main.py
```

### 生产环境 (Ubuntu)

```bash
# 1. 上传代码
scp -r ./quant4BTC ubuntu@server:/home/ubuntu/

# 2. SSH 登录
ssh ubuntu@server

# 3. 配置环境
cd /home/ubuntu/quant4BTC
cp .env.template .env
nano .env

# 4. 部署
chmod +x deployment/deploy.sh
./deployment/deploy.sh

# 5. 查看状态
sudo systemctl status quant4btc
```

## 核心配置参数

### 策略配置
```bash
STRATEGY_NAME=ultimate_scalping
CONTRACT_SYMBOL=BTC/USDT:USDT
TIMEFRAME=1m
```

### Ultimate Scalping 参数
```bash
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
RSI_LENGTH=14
SUPERTREND_MULT=3.0
SCALPING_TAKE_PROFIT_PCT=3.0
SCALPING_STOP_LOSS_PCT=1.5
SCALPING_REVERSAL_EXIT=true
```

### 交易配置
```bash
FIXED_ORDER_SIZE=3.0
LEVERAGE=10
MODE=long_short
```

### 风险管理
```bash
INITIAL_CAPITAL=500
RISK_PER_TRADE=0.1
DAILY_DRAWDOWN_LIMIT=0.10
OVERALL_DRAWDOWN_LIMIT=0.30
```

### API 配置
```bash
USE_DEMO=true
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret
```

## 依赖安装

```bash
# 安装 python-dotenv (推荐)
pip install python-dotenv

# 或安装所有依赖
pip install -r requirements.txt
```

**注意**: 如果没有安装 python-dotenv,系统会使用内置的备用加载器。

## 验证配置

```bash
# 测试环境变量加载
python -c "from env_loader import load_env; print('OK')"

# 测试配置加载
python -c "from config import Config; cfg = Config(); print(f'Strategy: {cfg.strategy_name}')"
```

## 常见问题

### Q: .env 文件没有被加载?
A: 确保 .env 文件在项目根目录,与 `config.py` 同级。

### Q: 需要重启服务才能加载新配置吗?
A: 是的,修改 .env 后需要重启:
```bash
sudo systemctl restart quant4btc
```

### Q: 如何查看当前加载的配置?
A: 查看日志,启动时会显示配置信息:
```bash
sudo journalctl -u quant4btc | grep "Strategy:"
```

### Q: python-dotenv 是必需的吗?
A: 不是必需的。如果未安装,会使用内置的备用加载器。

## 安全建议

1. **不要提交 .env 文件到 git**
   ```bash
   # 添加到 .gitignore
   echo ".env" >> .gitignore
   ```

2. **保护敏感信息**
   ```bash
   # 限制 .env 文件权限
   chmod 600 .env
   ```

3. **使用测试网进行测试**
   ```bash
   USE_DEMO=true
   ```

4. **限制 API 密钥权限**
   - 只开启交易权限
   - 不要开启提现权限
   - 限制 IP 白名单

## 下一步

查看详细部署指南:
- 快速开始: [deployment/QUICKSTART.md](deployment/QUICKSTART.md)
- 完整文档: [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)
- 策略说明: [strategies/ultimate_scalping_README.md](strategies/ultimate_scalping_README.md)
