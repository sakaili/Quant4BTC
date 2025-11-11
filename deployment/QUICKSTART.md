# 快速部署指南 (5 分钟)

## 第 1 步: 上传代码到服务器

```bash
# 方法 1: 使用 scp (在本地机器运行)
scp -r ./quant4BTC ubuntu@your-server-ip:/home/ubuntu/

# 方法 2: 使用 git (在服务器运行)
cd /home/ubuntu
git clone <your-repo-url> quant4BTC
```

## 第 2 步: 安装 Python 依赖

```bash
# SSH 登录到服务器
ssh ubuntu@your-server-ip

# 安装 Python 和必要工具
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

## 第 3 步: 配置环境变量

```bash
cd /home/ubuntu/quant4BTC

# 复制配置模板
cp .env.template .env

# 编辑配置 (使用 nano 或 vim)
nano .env
```

**必须修改的配置**:
```bash
STRATEGY_NAME=ultimate_scalping
BINANCE_API_KEY=your_key_here
BINANCE_SECRET=your_secret_here
USE_DEMO=true  # 测试网,正式交易改为 false
FIXED_ORDER_SIZE=3.0
```

保存: `Ctrl+X`, 然后 `Y`, 然后 `Enter`

## 第 4 步: 运行部署脚本

```bash
chmod +x deployment/deploy.sh
./deployment/deploy.sh
```

按提示操作,脚本会自动完成所有设置。

## 第 5 步: 验证运行

```bash
# 查看服务状态
sudo systemctl status quant4btc

# 查看实时日志
sudo journalctl -u quant4btc -f

# 按 Ctrl+C 退出日志查看
```

---

## 常用命令

```bash
# 启动
sudo systemctl start quant4btc

# 停止
sudo systemctl stop quant4btc

# 重启
sudo systemctl restart quant4btc

# 查看状态
sudo systemctl status quant4btc

# 查看日志
sudo journalctl -u quant4btc -f
```

---

## 切换策略

```bash
# 编辑配置
nano /home/ubuntu/quant4BTC/.env

# 修改 STRATEGY_NAME
# 可选: ultimate_scalping, supertrend

# 重启服务
sudo systemctl restart quant4btc
```

---

## 从测试网切换到正式交易

⚠️ **重要**: 请先在测试网充分验证策略!

```bash
# 1. 编辑配置
nano /home/ubuntu/quant4BTC/.env

# 2. 修改
USE_DEMO=false

# 3. 确认 API 密钥是正式网的密钥

# 4. 重启服务
sudo systemctl restart quant4btc

# 5. 密切监控日志
sudo journalctl -u quant4btc -f
```

---

## 故障排查

### 服务启动失败

```bash
# 查看错误详情
sudo journalctl -u quant4btc -n 50

# 手动测试
cd /home/ubuntu/quant4BTC
source venv/bin/activate
python3 main.py
```

### API 连接失败

```bash
# 测试网络
ping api.binance.com

# 检查 API 密钥
cat /home/ubuntu/quant4BTC/.env | grep BINANCE
```

### 查看交易记录

```bash
# 查看 CSV 日志
cat /home/ubuntu/quant4BTC/trade_log_*.csv

# 实时监控
tail -f /home/ubuntu/quant4BTC/trade_log_*.csv
```

---

## 需要帮助?

查看完整文档: [deployment/DEPLOYMENT.md](DEPLOYMENT.md)
