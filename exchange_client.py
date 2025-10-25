# exchange_client.py
import ccxt
from typing import Any, Dict
from config import Config


class ExchangeClient:
    """最薄封装 ccxt.okx，负责建立连接并管理交易所状态。"""

    def __init__(self, cfg: Config, logger):
        """初始化 okx 客户端，处理沙盒/实盘、杠杆和持仓模式设置。"""
        self.cfg = cfg
        self.logger = logger
        self.exchange = ccxt.okx({
            'apiKey': cfg.okx_api_key,
            'secret': cfg.okx_secret,
            'password': cfg.okx_password,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
            'proxies': cfg.proxies(),
            'timeout': 20000
        })
        self.exchange.set_sandbox_mode(bool(cfg.use_demo))
        self.exchange.load_markets()
        # 模式与杠杆
        try:
            hedged = (cfg.position_mode.lower() == 'hedge')
            self.exchange.set_position_mode(hedged, cfg.symbol)
            self.logger.info(f"已设置 POSITION_MODE={cfg.position_mode}")
        except Exception as e:
            self.logger.warning(f"设置 POSITION_MODE 失败：{e}")
        try:
            self.exchange.set_leverage(cfg.leverage, cfg.symbol, params={'mgnMode': cfg.margin_mode})
            self.logger.info(f"已设置 LEVERAGE={cfg.leverage}, MARGIN_MODE={cfg.margin_mode}")
        except Exception as e:
            self.logger.warning(f"设置杠杆失败：{e}")
        self.market_info = self.exchange.market(cfg.symbol)

    def amount_to_precision(self, n: float) -> float:
        """按交易所规则修正合约张数，失败时回退到四舍五入。"""
        try:
            return float(self.exchange.amount_to_precision(self.cfg.symbol, n))
        except Exception:
            return float(int(round(n)))

    def price_to_precision(self, p: float) -> float:
        """按交易所 tick size 修正价格。"""
        return float(self.exchange.price_to_precision(self.cfg.symbol, p))

    def min_contracts(self) -> float:
        """读取交易所要求的最小下单数量，不可用时落到 1 张。"""
        try:
            min_amt = self.market_info.get('limits', {}).get('amount', {}).get('min', None)
            if min_amt is not None:
                return max(1.0, float(min_amt))
        except Exception:
            pass
        return 1.0

    # 数据与账户
    def fetch_ohlcv(self, limit: int):
        """获取最新 K 线数据，列顺序与 ccxt 保持一致。"""
        return self.exchange.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, limit=limit)

    def fetch_ticker_last(self) -> float:
        """读取最新成交价用于风控或下单。"""
        t = self.exchange.fetch_ticker(self.cfg.symbol)
        return float(t['last'])

    def fetch_balance_swap(self) -> Dict[str, Any] | None:
        """拉取 swap 账户的资产信息，捕获异常避免崩溃。"""
        try:
            return self.exchange.fetch_balance({'type': 'swap'})
        except Exception as e:
            self.logger.error(f"获取合约余额失败: {e}")
            return None

    def fetch_positions(self):
        """查询目标合约的全部持仓，失败返回空列表。"""
        try:
            return self.exchange.fetch_positions([self.cfg.symbol])
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return []

    # 下单
    def create_market_order(
        self,
        side: str,
        amount: int,
        reduce_only: bool,
        pos_side: str | None,
    ):
        """构造带有 margin/posSide 参数的市价单。"""
        params = {'tdMode': self.cfg.margin_mode, 'reduceOnly': bool(reduce_only)}
        if self.cfg.position_mode.lower() == 'hedge' and pos_side in ('long', 'short'):
            params['posSide'] = pos_side
        return self.exchange.create_order(
            symbol=self.cfg.symbol,
            type='market',
            side=side,
            amount=amount,
            price=None,
            params=params
        )

    def create_stop_market_order(
        self,
        side: str,
        amount: int,
        trigger_price: float,
        pos_side: str | None,
    ):
        """下达条件止损市价单，挂在交易所侧执行。"""
        params: dict = {
            'tdMode': self.cfg.margin_mode,
            'reduceOnly': True,
            'stopLossPrice': float(self.price_to_precision(trigger_price)),
            'slOrdPx': '-1',
            'slTriggerPxType': 'last',
        }
        if self.cfg.position_mode.lower() == 'hedge' and pos_side in ('long', 'short'):
            params['posSide'] = pos_side
        return self.exchange.create_order(
            symbol=self.cfg.symbol,
            type='conditional',
            side=side,
            amount=amount,
            price=None,
            params=params,
        )

    def create_take_profit_order(
        self,
        side: str,
        amount: int,
        trigger_price: float,
        pos_side: str | None,
    ):
        """挂出止盈市价单。"""
        params: dict = {
            'tdMode': self.cfg.margin_mode,
            'reduceOnly': True,
            'tpTriggerPx': float(self.price_to_precision(trigger_price)),
            'tpTriggerPxType': 'last',
            'tpOrdPx': '-1',  # -1 代表市价执行
        }
        if self.cfg.position_mode.lower() == 'hedge' and pos_side in ('long', 'short'):
            params['posSide'] = pos_side
        return self.exchange.create_order(
            symbol=self.cfg.symbol,
            type='conditional',
            side=side,
            amount=amount,
            price=None,
            params=params,
        )
