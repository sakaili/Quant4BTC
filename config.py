# config.py
import logging
import os
from dataclasses import dataclass

TRUE_SET = ('1', 'true', 'yes')


@dataclass(frozen=True)
class Config:
    """集中维护策略运行所需的全部环境参数。"""

    # 市场与数据
    symbol: str = os.getenv('CONTRACT_SYMBOL', 'BTC/USDT:USDT')
    timeframe: str = os.getenv('TIMEFRAME', '5m')
    fetch_limit: int = int(os.getenv('FETCH_LIMIT', '900'))

    # 下单与模拟
    contracts_per_order: int = int(os.getenv('CONTRACTS_PER_ORDER', '10'))

    # 运行环境：仅保留沙盒/实盘
    use_demo: bool = os.getenv('USE_DEMO', 'true').lower() in TRUE_SET
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    strategy_name: str = os.getenv('STRATEGY_NAME', 'macd_triple_filter') #macd_triple_filter supertrend

    # 代理
    http_proxy: str = os.getenv('HTTP_PROXY', '')
    https_proxy: str = os.getenv('HTTPS_PROXY', '')

    # OKX 密钥
    okx_api_key: str = os.getenv('OKX_API_KEY', '')
    okx_secret: str = os.getenv('OKX_SECRET', '')
    okx_password: str = os.getenv('OKX_PASSWORD', '')

    # 合约参数
    leverage: int = int(os.getenv('LEVERAGE', '10'))
    margin_mode: str = os.getenv('MARGIN_MODE', 'cross')      # cross/isolated
    position_mode: str = os.getenv('POSITION_MODE', 'hedge')  # net/hedge

    # 指标/选参
    metric_lookback: int = int(os.getenv('METRIC_LOOKBACK', '300'))
    regime_lookback: int = int(os.getenv('REGIME_LOOKBACK', '600'))
    feature_ema: int = int(os.getenv('FEATURE_EMA', '20'))
    n_clusters: int = int(os.getenv('N_CLUSTERS', '3'))

    atr_length: int = int(os.getenv('ATR_LENGTH', '5'))
    min_mult: float = float(os.getenv('MIN_MULT', '1.0'))
    max_mult: float = float(os.getenv('MAX_MULT', '5.0'))
    step: float = float(os.getenv('STEP', '0.5'))
    selection: str = os.getenv('SELECTION', 'regime_kmeans')

    macd_fast_length: int = int(os.getenv('MACD_FAST_LENGTH', '12'))
    macd_slow_length: int = int(os.getenv('MACD_SLOW_LENGTH', '26'))
    macd_signal_length: int = int(os.getenv('MACD_SIGNAL_LENGTH', '9'))
    macd_regime_ma_length: int = int(os.getenv('MACD_REGIME_MA_LENGTH', '200'))
    macd_hist_confirm_bars: int = int(os.getenv('MACD_HIST_CONFIRM_BARS', '2'))
    macd_atr_min: float = float(os.getenv('MACD_ATR_MIN', '0.0008'))
    macd_atr_max: float = float(os.getenv('MACD_ATR_MAX', '0.02'))
    macd_atr_stop_multiple: float = float(os.getenv('MACD_ATR_STOP_MULTIPLE', '2.0'))

    # 策略模式
    mode: str = os.getenv('MODE', 'long_short')  # long_flat / long_short
    rr: float = float(os.getenv('RR', '2.0'))

    # 成本估算（仅用于评估）
    slippage_rate: float = float(os.getenv('SLIPPAGE_RATE', '0.0001'))
    fee_rate: float = float(os.getenv('FEE_RATE', '0.0005'))
    turnover_penalty: float = float(os.getenv('TURNOVER_PENALTY', '0.0'))

    # 日志
    csv_log_file: str = os.getenv('CSV_LOG_FILE', 'trade_log.csv')
    symbols_env: str = os.getenv('CONTRACT_SYMBOLS', '')

    # 风险管理
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '500'))
    risk_per_trade: float = float(os.getenv('RISK_PER_TRADE', '0.1'))
    daily_drawdown_limit: float = float(os.getenv('DAILY_DRAWDOWN_LIMIT', '0.10'))
    overall_drawdown_limit: float = float(os.getenv('OVERALL_DRAWDOWN_LIMIT', '0.30'))
    stop_loss_pct: float = float(os.getenv('STOP_LOSS_PCT', '0.5'))

    # 信号质量控制
    signal_confirm: int = int(os.getenv('SIGNAL_CONFIRM', '1'))
    band_eps: float = float(os.getenv('BAND_EPS', '0.0005'))
    factor_hold_bars: int = int(os.getenv('FACTOR_HOLD_BARS', '10'))
    factor_sticky: float = float(os.getenv('FACTOR_STICKY', '0.1'))

    # 聚类/标准化
    zscore_window: int = int(os.getenv('ZSCORE_WINDOW', '500'))
    min_regime_samples: int = int(os.getenv('MIN_REGIME_SAMPLES', '50'))
    min_cluster_frac: float = float(os.getenv('MIN_CLUSTER_FRAC', '0.05'))

    backtest_trade_size: float = float(os.getenv('BACKTEST_TRADE_SIZE', '0.01'))

    def proxies(self):
        """根据配置拼接代理设置，未配置时返回 None。"""
        if self.http_proxy or self.https_proxy:
            return {'http': self.http_proxy, 'https': self.https_proxy}
        return None

    @property
    def symbol_list(self) -> list[str]:
        raw = self.symbols_env.strip()
        if not raw:
            return [self.symbol]
        parts = [p.strip() for p in raw.split(',')]
        symbols = [p for p in parts if p]
        return symbols or [self.symbol]


def setup_logging(level: str):
    """初始化基础日志输出，统一日志等级与格式。"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("adaptive_supertrend_okx_oop")
