# config.py
import logging
import os
from dataclasses import dataclass

TRUE_SET = ("1", "true", "yes")


@dataclass(frozen=True)
class Config:
    """Centralised runtime configuration for strategies, execution, and backtests."""

    # Market configuration
    symbol: str = os.getenv("CONTRACT_SYMBOL", "BTC/USDC")
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    fetch_limit: int = int(os.getenv("FETCH_LIMIT", "900"))

    # Trading model
    contracts_per_order: int = int(os.getenv("CONTRACTS_PER_ORDER", "10"))
    fixed_order_size: float = float(os.getenv("FIXED_ORDER_SIZE", "0.01"))

    # Environment toggles
    use_demo: bool = os.getenv("USE_DEMO", "false").lower() in TRUE_SET
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    strategy_name: str = os.getenv("STRATEGY_NAME", "supertrend")
    use_macd_filter: bool = os.getenv("USE_MACD_FILTER", "false").lower() in TRUE_SET

    # Proxy settings
    http_proxy: str = os.getenv("HTTP_PROXY", "")
    https_proxy: str = os.getenv("HTTPS_PROXY", "")

    # Binance API credentials
    binance_api_key: str = os.getenv("BINANCE_API_KEY", os.getenv("OKX_API_KEY", ""))
    binance_secret: str = os.getenv("BINANCE_SECRET", os.getenv("OKX_SECRET", ""))
    binance_recv_window: int = int(os.getenv("BINANCE_RECV_WINDOW", "5000"))

    # Leverage & position configuration
    leverage: int = int(os.getenv("LEVERAGE", "10"))
    margin_mode: str = os.getenv("MARGIN_MODE", "cross")  # cross / isolated
    position_mode: str = os.getenv("POSITION_MODE", "hedge")  # hedge / net

    # Indicator / selection parameters
    metric_lookback: int = int(os.getenv("METRIC_LOOKBACK", "40"))
    regime_lookback: int = int(os.getenv("REGIME_LOOKBACK", "40"))
    feature_ema: int = int(os.getenv("FEATURE_EMA", "12"))
    n_clusters: int = int(os.getenv("N_CLUSTERS", "3"))

    atr_length: int = int(os.getenv("ATR_LENGTH", "5"))
    min_mult: float = float(os.getenv("MIN_MULT", "0.8"))
    max_mult: float = float(os.getenv("MAX_MULT", "3.5"))
    step: float = float(os.getenv("STEP", "0.5"))
    selection: str = os.getenv("SELECTION", "regime_kmeans")

    macd_fast_length: int = int(os.getenv("MACD_FAST_LENGTH", "12"))
    macd_slow_length: int = int(os.getenv("MACD_SLOW_LENGTH", "26"))
    macd_signal_length: int = int(os.getenv("MACD_SIGNAL_LENGTH", "9"))
    macd_regime_ma_length: int = int(os.getenv("MACD_REGIME_MA_LENGTH", "200"))
    macd_hist_confirm_bars: int = int(os.getenv("MACD_HIST_CONFIRM_BARS", "2"))
    macd_atr_min: float = float(os.getenv("MACD_ATR_MIN", "0.0008"))
    macd_atr_max: float = float(os.getenv("MACD_ATR_MAX", "0.02"))
    macd_atr_stop_multiple: float = float(os.getenv("MACD_ATR_STOP_MULTIPLE", "2.0"))

    # Risk mode
    mode: str = os.getenv("MODE", "long_short")  # long_flat / long_short
    rr: float = float(os.getenv("RR", "2.0"))

    # Cost model (slippage, fees, penalties)
    slippage_rate: float = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.0000275"))
    turnover_penalty: float = float(os.getenv("TURNOVER_PENALTY", "0.0"))

    # Logging
    csv_log_file: str = os.getenv("CSV_LOG_FILE", "trade_log.csv")
    symbols_env: str = os.getenv("CONTRACT_SYMBOLS", "")

    # Risk controls
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "500"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.1"))
    daily_drawdown_limit: float = float(os.getenv("DAILY_DRAWDOWN_LIMIT", "0.10"))
    overall_drawdown_limit: float = float(os.getenv("OVERALL_DRAWDOWN_LIMIT", "0.30"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.5"))

    # Signal controls
    signal_confirm: int = int(os.getenv("SIGNAL_CONFIRM", "1"))
    band_eps: float = float(os.getenv("BAND_EPS", "0.0005"))
    factor_hold_bars: int = int(os.getenv("FACTOR_HOLD_BARS", "1"))
    factor_sticky: float = float(os.getenv("FACTOR_STICKY", "0.1"))
    force_factor_recalc: bool = os.getenv("FORCE_FACTOR_RECALC", "true").lower() in TRUE_SET
    cooldown_loss_pct: float = float(os.getenv("COOLDOWN_LOSS_PCT", "0.0"))
    cooldown_loss_amount: float = float(os.getenv("COOLDOWN_LOSS_AMOUNT", "10.0"))
    cooldown_duration_minutes: int = int(os.getenv("COOLDOWN_DURATION_MINUTES", "60"))

    # Normalisation
    zscore_window: int = int(os.getenv("ZSCORE_WINDOW", "12"))
    min_regime_samples: int = int(os.getenv("MIN_REGIME_SAMPLES", "8"))
    min_cluster_frac: float = float(os.getenv("MIN_CLUSTER_FRAC", "0.05"))

    backtest_trade_size: float = float(os.getenv("BACKTEST_TRADE_SIZE", "0.02"))
    dump_kmeans_debug: bool = os.getenv("DUMP_KMEANS_DEBUG", "false").lower() in TRUE_SET
    kmeans_debug_dir: str = os.getenv("KMEANS_DEBUG_DIR", "kmeans_debug")

    def proxies(self):
        """Return optional proxy settings in ccxt format."""

        if self.http_proxy or self.https_proxy:
            return {"http": self.http_proxy, "https": self.https_proxy}
        return None

    @property
    def symbol_list(self) -> list[str]:
        raw = self.symbols_env.strip()
        if not raw:
            return [self.symbol]
        parts = [p.strip() for p in raw.split(",")]
        symbols = [p for p in parts if p]
        return symbols or [self.symbol]


def setup_logging(level: str):
    """Initialise logging with a consistent formatter."""

    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("adaptive_supertrend_binance")
