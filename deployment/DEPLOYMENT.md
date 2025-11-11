# Ubuntu ECS 服务器部署指南

本指南将帮助你在 Ubuntu ECS 服务器上部署 Quant4BTC 交易机器人。

## 目录
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [配置说明](#配置说明)
- [服务管理](#服务管理)
- [监控和日志](#监控和日志)
- [故障排查](#故障排查)
- [安全建议](#安全建议)

---

## 系统要求

- **操作系统**: Ubuntu 20.04 LTS 或更高版本
- **Python**: 3.9 或更高版本
- **内存**: 至少 1GB RAM
- **磁盘空间**: 至少 2GB 可用空间
- **网络**: 稳定的互联网连接,能够访问币安 API

---

## 快速开始

```bash
# 1. 上传代码到服务器
cd /home/ubuntu
git clone <your-repo> quant4BTC
# 或使用 scp 上传文件

# 2. 进入项目目录
cd quant4BTC

# 3. 复制配置模板
cp .env.template .env

# 4. 编辑配置文件
nano .env
# 至少需要配置:
# - BINANCE_API_KEY
# - BINANCE_SECRET
# - USE_DEMO (建议先使用 true 测试)

# 5. 运行部署脚本
chmod +x deployment/deploy.sh
./deployment/deploy.sh

# 6. 检查服务状态
sudo systemctl status quant4btc
```

---

## 详细步骤

### 1. 准备服务器

#### 1.1 更新系统
```bash
sudo apt update
sudo apt upgrade -y
```

#### 1.2 安装必要软件
```bash
# 安装 Python 3 和 pip
sudo apt install -y python3 python3-pip python3-venv

# 安装 git (如果需要从仓库拉取)
sudo apt install -y git

# 安装其他依赖
sudo apt install -y build-essential libssl-dev libffi-dev
```

#### 1.3 创建应用用户 (可选,推荐)
```bash
# 如果不使用 ubuntu 用户,可以创建专用用户
sudo useradd -m -s /bin/bash trader
sudo su - trader
```

### 2. 上传代码

#### 方法 A: 使用 Git
```bash
cd /home/ubuntu
git clone https://github.com/your-username/quant4BTC.git
cd quant4BTC
```

#### 方法 B: 使用 SCP
```bash
# 在本地机器上运行
scp -r ./quant4BTC ubuntu@your-server-ip:/home/ubuntu/
```

#### 方法 C: 使用 FTP 工具
使用 FileZilla 或 WinSCP 上传整个项目文件夹

### 3. 配置环境变量

```bash
cd /home/ubuntu/quant4BTC

# 复制模板
cp .env.template .env

# 编辑配置
nano .env
```

#### 必须配置的参数:

```bash
# 选择策略
STRATEGY_NAME=ultimate_scalping  # 或 supertrend

# 交易对
CONTRACT_SYMBOL=BTC/USDT:USDT

# API 凭证 (从币安获取)
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_SECRET=your_actual_secret_here

# 使用测试网 (强烈建议先测试)
USE_DEMO=true

# 仓位大小
FIXED_ORDER_SIZE=3.0
```

保存并退出 (Ctrl+X, 然后 Y, 然后 Enter)

### 4. 创建 requirements.txt

如果项目中没有 requirements.txt,创建一个:

```bash
cat > requirements.txt << 'EOF'
ccxt>=4.0.0
pandas>=1.5.0
numpy>=1.24.0
python-dotenv>=1.0.0
scikit-learn>=1.2.0
EOF
```

### 5. 运行部署脚本

```bash
# 给脚本执行权限
chmod +x deployment/deploy.sh

# 运行部署
./deployment/deploy.sh
```

脚本会自动:
1. 创建 Python 虚拟环境
2. 安装依赖包
3. 测试配置加载
4. 安装 systemd 服务
5. 启动服务 (如果你选择)

### 6. 手动启动服务 (如果脚本没有启动)

```bash
sudo systemctl enable quant4btc.service
sudo systemctl start quant4btc.service
sudo systemctl status quant4btc.service
```

---

## 配置说明

### 环境变量优先级

系统会按以下优先级加载配置:
1. **系统环境变量** (最高优先级)
2. **.env 文件**
3. **代码中的默认值** (最低优先级)

### 主要配置参数

#### 策略选择
```bash
# 可选值: supertrend, ultimate_scalping
STRATEGY_NAME=ultimate_scalping
```

#### Ultimate Scalping 策略参数
```bash
EMA_FAST_LENGTH=20          # 快速 EMA 周期
EMA_SLOW_LENGTH=50          # 慢速 EMA 周期
RSI_LENGTH=14               # RSI 周期
SUPERTREND_MULT=3.0         # SuperTrend 倍数
SCALPING_TAKE_PROFIT_PCT=3.0   # 止盈百分比
SCALPING_STOP_LOSS_PCT=1.5     # 止损百分比
SCALPING_REVERSAL_EXIT=true    # 反向信号平仓
```

#### 风险管理
```bash
INITIAL_CAPITAL=500              # 初始资金
RISK_PER_TRADE=0.1              # 每笔交易风险比例
DAILY_DRAWDOWN_LIMIT=0.10       # 日回撤限制
OVERALL_DRAWDOWN_LIMIT=0.30     # 总回撤限制
STOP_LOSS_PCT=0.5               # 止损百分比
```

#### 交易模式
```bash
MODE=long_short    # long_flat (只做多) 或 long_short (多空双向)
LEVERAGE=10        # 杠杆倍数
MARGIN_MODE=cross  # cross (全仓) 或 isolated (逐仓)
POSITION_MODE=hedge # hedge (双向持仓) 或 net (单向持仓)
```

---

## 服务管理

### 基本命令

```bash
# 启动服务
sudo systemctl start quant4btc

# 停止服务
sudo systemctl stop quant4btc

# 重启服务
sudo systemctl restart quant4btc

# 查看服务状态
sudo systemctl status quant4btc

# 开机自启动
sudo systemctl enable quant4btc

# 禁用开机自启动
sudo systemctl disable quant4btc

# 重新加载服务配置
sudo systemctl daemon-reload
```

### 修改服务配置

如果需要修改服务配置:

```bash
# 编辑服务文件
sudo nano /etc/systemd/system/quant4btc.service

# 重新加载配置
sudo systemctl daemon-reload

# 重启服务
sudo systemctl restart quant4btc
```

---

## 监控和日志

### 查看实时日志

```bash
# 查看 systemd 日志 (推荐)
sudo journalctl -u quant4btc -f

# 查看最近 100 行日志
sudo journalctl -u quant4btc -n 100

# 查看特定时间的日志
sudo journalctl -u quant4btc --since "2025-01-01 00:00:00"

# 查看应用日志文件
tail -f /home/ubuntu/quant4BTC/logs/bot.log

# 查看错误日志
tail -f /home/ubuntu/quant4BTC/logs/bot.error.log
```

### 查看交易记录

```bash
# 查看 CSV 交易日志
cat /home/ubuntu/quant4BTC/trade_log_*.csv

# 实时监控交易日志
tail -f /home/ubuntu/quant4BTC/trade_log_*.csv
```

### 日志轮转

创建日志轮转配置以防止日志文件过大:

```bash
sudo nano /etc/logrotate.d/quant4btc
```

添加以下内容:

```
/home/ubuntu/quant4BTC/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ubuntu ubuntu
}
```

---

## 故障排查

### 服务无法启动

1. **查看详细错误信息**:
```bash
sudo journalctl -u quant4btc -n 50 --no-pager
```

2. **检查配置文件**:
```bash
cd /home/ubuntu/quant4BTC
source venv/bin/activate
python3 -c "from config import Config; cfg = Config(); print(cfg)"
```

3. **检查 Python 环境**:
```bash
/home/ubuntu/quant4BTC/venv/bin/python --version
/home/ubuntu/quant4BTC/venv/bin/pip list
```

### API 连接失败

1. **检查网络连接**:
```bash
ping api.binance.com
curl -I https://api.binance.com/api/v3/ping
```

2. **检查 API 密钥**:
```bash
# 确保 .env 文件中有正确的 API 密钥
cat /home/ubuntu/quant4BTC/.env | grep BINANCE
```

3. **检查防火墙**:
```bash
sudo ufw status
# 如果需要允许出站连接
sudo ufw allow out 443/tcp
```

### 权限问题

```bash
# 确保正确的文件权限
cd /home/ubuntu
sudo chown -R ubuntu:ubuntu quant4BTC
chmod +x quant4BTC/deployment/*.sh
```

### Python 依赖问题

```bash
# 重新安装依赖
cd /home/ubuntu/quant4BTC
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 安全建议

### 1. API 密钥安全

- **限制 IP**: 在币安 API 设置中限制只允许你的服务器 IP
- **权限最小化**: 只授予交易必需的权限,不要开启提现权限
- **使用测试网**: 正式交易前先在测试网验证策略
- **定期更换**: 定期更换 API 密钥

### 2. 服务器安全

```bash
# 配置防火墙
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from your-ip to any port 22

# 禁用密码登录,使用 SSH 密钥
sudo nano /etc/ssh/sshd_config
# 设置: PasswordAuthentication no
sudo systemctl restart sshd

# 自动安全更新
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

### 3. 监控和告警

设置告警通知 (可选):

```bash
# 安装监控工具
sudo apt install monitoring-plugins

# 配置邮件通知
sudo apt install mailutils

# 在策略异常时发送邮件
# (需要在代码中添加邮件通知逻辑)
```

### 4. 备份

```bash
# 创建备份脚本
cat > /home/ubuntu/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/quant4btc_$DATE.tar.gz" \
    /home/ubuntu/quant4BTC/.env \
    /home/ubuntu/quant4BTC/trade_log*.csv \
    /home/ubuntu/quant4BTC/logs/
# 只保留最近 7 天的备份
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /home/ubuntu/backup.sh

# 添加到 crontab (每天凌晨 3 点备份)
crontab -e
# 添加: 0 3 * * * /home/ubuntu/backup.sh
```

---

## 更新和维护

### 更新代码

```bash
cd /home/ubuntu/quant4BTC

# 停止服务
sudo systemctl stop quant4btc

# 拉取最新代码 (如果使用 Git)
git pull

# 或重新上传文件
# scp ...

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 重启服务
sudo systemctl start quant4btc
```

### 性能优化

监控资源使用:
```bash
# CPU 和内存使用
top -p $(pgrep -f "python.*main.py")

# 磁盘使用
df -h
du -sh /home/ubuntu/quant4BTC/*
```

---

## 常见问题 (FAQ)

### Q: 如何切换策略?
A: 编辑 .env 文件,修改 `STRATEGY_NAME`,然后重启服务:
```bash
nano /home/ubuntu/quant4BTC/.env
sudo systemctl restart quant4btc
```

### Q: 如何同时运行多个交易对?
A: 在 .env 文件中设置:
```bash
CONTRACT_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT
```

### Q: 如何从测试网切换到正式网?
A: 修改 .env 文件:
```bash
USE_DEMO=false
```
**重要**: 确保在测试网充分验证后再切换!

### Q: 服务器重启后机器人会自动启动吗?
A: 如果运行了 `sudo systemctl enable quant4btc`,会自动启动

### Q: 如何临时停止交易但不关闭服务?
A: 可以修改策略参数使其不产生信号,或者直接停止服务:
```bash
sudo systemctl stop quant4btc
```

---

## 技术支持

如果遇到问题:
1. 查看日志: `sudo journalctl -u quant4btc -n 100`
2. 检查配置: `cat /home/ubuntu/quant4BTC/.env`
3. 测试网络: `ping api.binance.com`
4. 查阅文档: 阅读策略 README 文件

---

## 附录

### A. 完整的 requirements.txt 示例

```txt
ccxt>=4.0.0
pandas>=1.5.0
numpy>=1.24.0
python-dotenv>=1.0.0
scikit-learn>=1.2.0
```

### B. 示例 cron 任务

```bash
# 编辑 crontab
crontab -e

# 每小时检查服务状态
0 * * * * systemctl is-active --quiet quant4btc || systemctl start quant4btc

# 每天凌晨 3 点备份
0 3 * * * /home/ubuntu/backup.sh

# 每天凌晨 4 点清理旧日志
0 4 * * * find /home/ubuntu/quant4BTC/logs -name "*.log" -mtime +7 -delete
```

### C. 监控脚本示例

```bash
#!/bin/bash
# health_check.sh
if ! systemctl is-active --quiet quant4btc; then
    echo "Service is down! Restarting..."
    systemctl start quant4btc
    echo "Service restarted at $(date)" >> /home/ubuntu/restart.log
fi
```

---

**祝交易顺利!** 🚀
