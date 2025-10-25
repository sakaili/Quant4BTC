# order_executor.py
from typing import Optional
from config import Config
from exchange_client import ExchangeClient


class OrderExecutor:
    """统一封装开平仓逻辑，确保沙盒/实盘切换透明。"""

    def __init__(self, cfg: Config, exch: ExchangeClient, logger):
        """保存配置、交易所和日志器，供随时下单与记账。"""
        self.cfg = cfg
        self.exch = exch
        self.logger = logger

    def market_info(self):
        """返回交易所缓存的合约元数据。"""
        return getattr(self.exch, 'market_info', {})

    def account_snapshot(self) -> dict:
        """获取账户权益与可用保证金等信息。"""
        bal = self.exch.fetch_balance_swap()
        if not bal:
            return {'equity': 0.0, 'free': 0.0, 'used': 0.0}
        usdt_free = float(bal['free'].get('USDT', 0.0))
        usdt_used = float(bal['used'].get('USDT', 0.0))
        return {'equity': usdt_free + usdt_used, 'free': usdt_free, 'used': usdt_used}

    def account_equity(self) -> float:
        """计算合约账户权益（可用+占用），失败返回 0。"""
        snap = self.account_snapshot()
        return float(snap['equity'])

    def available_margin(self) -> float:
        """返回当前可用保证金。"""
        snap = self.account_snapshot()
        return float(snap['free'])

    def place(
        self,
        side: str,
        n_contracts: int,
        last_price: Optional[float],
        reduce_only: bool,
        pos_side: str | None,
    ):
        """构造统一的市价单请求并处理返回值。"""
        amt = max(int(round(self.exch.amount_to_precision(n_contracts))), int(self.exch.min_contracts()))
        ps = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_market_order(
                side=side.lower().strip(),
                amount=amt,
                reduce_only=reduce_only,
                pos_side=ps,
            )
            filled_raw = order.get('filled', 0)
            if filled_raw in (None, ''):
                filled_raw = order.get('amount', amt)
            try:
                filled = float(filled_raw)
            except (TypeError, ValueError):
                filled = float(amt)

            price_candidates = (
                order.get('average'),
                order.get('price'),
                last_price,
            )
            avg_val = None
            for cand in price_candidates:
                if cand not in (None, ''):
                    avg_val = cand
                    break
            try:
                avg = float(avg_val) if avg_val is not None else float(last_price or 0.0)
            except (TypeError, ValueError):
                avg = float(last_price or 0.0)

            fee_cost = 0.0
            if isinstance(order.get('fees', None), list) and len(order['fees']) > 0:
                fee_cost = sum(float(f.get('cost', 0) or 0) for f in order['fees'])
            order_id = order.get('id', '')
            self.logger.info(f"【OKX合约】下单成功 id={order_id} filled={filled} avg={avg:.6f} fee≈{fee_cost}")
            return {'status': 'ok', 'side': side, 'amount': filled if filled else amt, 'price': avg, 'fee': fee_cost, 'order_id': order_id}
        except Exception as e:
            self.logger.exception(f"【OKX合约】下单异常 {e}")
            return {'status': 'error', 'reason': str(e)}

    def place_stop(self, side: str, amount: int, trigger_price: float, pos_side: str | None):
        """在交易所挂出止损市价单。"""
        amt = max(int(round(self.exch.amount_to_precision(amount))), int(self.exch.min_contracts()))
        ps = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_stop_market_order(
                side=side.lower().strip(),
                amount=amt,
                trigger_price=trigger_price,
                pos_side=ps,
            )
            order_id = order.get('id', '')
            self.logger.info(f"【OKX合约】挂止损成功 id={order_id} trigger={trigger_price:.6f}")
            return {'status': 'ok', 'order_id': order_id}
        except Exception as e:
            self.logger.error(f"挂止损失败: {e}")
            return {'status': 'error', 'reason': str(e)}

    def place_take_profit(self, side: str, amount: int, trigger_price: float, pos_side: str | None):
        """在交易所挂出止盈市价单。"""
        amt = max(int(round(self.exch.amount_to_precision(amount))), int(self.exch.min_contracts()))
        ps = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_take_profit_order(
                side=side.lower().strip(),
                amount=amt,
                trigger_price=trigger_price,
                pos_side=ps,
            )
            order_id = order.get('id', '')
            self.logger.info(f"【OKX合约】挂止盈成功 id={order_id} trigger={trigger_price:.6f}")
            return {'status': 'ok', 'order_id': order_id}
        except Exception as e:
            self.logger.error(f"挂止盈失败: {e}")
            return {'status': 'error', 'reason': str(e)}

    def cancel_all_conditional(self):
        """取消所有条件单。"""
        return self.exch.cancel_all_conditional_orders()

    # 语义方法
    def open_long(self, amt: int, last_price: float):
        """发出开多单，自动匹配 Hedge/Net 模式。"""
        ps = 'long' if self.cfg.position_mode.lower() == 'hedge' else None
        return self.place('buy', amt, last_price, reduce_only=False, pos_side=ps)

    def open_short(self, amt: int, last_price: float):
        """发出开空单，自动匹配 Hedge/Net 模式。"""
        ps = 'short' if self.cfg.position_mode.lower() == 'hedge' else None
        return self.place('sell', amt, last_price, reduce_only=False, pos_side=ps)

    def close_long(self, amt: int, last_price: float):
        """按张数减少多头。"""
        if amt <= 0:
            return None
        ps = 'long' if self.cfg.position_mode.lower() == 'hedge' else None
        return self.place('sell', amt, last_price, reduce_only=True, pos_side=ps)

    def close_short(self, amt: int, last_price: float):
        """按张数减少空头。"""
        if amt <= 0:
            return None
        ps = 'short' if self.cfg.position_mode.lower() == 'hedge' else None
        return self.place('buy', amt, last_price, reduce_only=True, pos_side=ps)

    def close_long_all(self, last_price: float, long_amount: int):
        """按张数一次性平掉所有多单，若无仓位直接返回。"""
        if long_amount <= 0:
            return None
        return self.close_long(long_amount, last_price)

    def close_short_all(self, last_price: float, short_amount: int):
        """按张数一次性平掉所有空单，若无仓位直接返回。"""
        if short_amount <= 0:
            return None
        return self.close_short(short_amount, last_price)
