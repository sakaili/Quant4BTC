# order_executor.py
from typing import Optional

from config import Config
from exchange_client import ExchangeClient


class OrderExecutor:
    """Unified wrapper around order placement and risk utilities."""

    def __init__(self, cfg: Config, exch: ExchangeClient, logger):
        self.cfg = cfg
        self.exch = exch
        self.logger = logger

    def market_info(self):
        return getattr(self.exch, "market_info", {})

    def account_snapshot(self) -> dict:
        balance = self.exch.fetch_balance_swap()
        if not balance:
            return {"equity": 0.0, "free": 0.0, "used": 0.0}
        quote_ccy = self.cfg.symbol.split("/")[-1].upper()
        try:
            free = float(balance["free"].get(quote_ccy, 0.0))
            used = float(balance["used"].get(quote_ccy, 0.0))
        except Exception:
            free = (
                float(balance.get("free", {}).get(quote_ccy, 0.0))
                if isinstance(balance.get("free"), dict)
                else 0.0
            )
            used = (
                float(balance.get("used", {}).get(quote_ccy, 0.0))
                if isinstance(balance.get("used"), dict)
                else 0.0
            )
        return {"equity": free + used, "free": free, "used": used}

    def account_equity(self) -> float:
        return float(self.account_snapshot()["equity"])

    def available_margin(self) -> float:
        return float(self.account_snapshot()["free"])

    def _normalise_amount(self, qty: float) -> float:
        precise = float(self.exch.amount_to_precision(qty))
        minimum = float(self.exch.min_contracts())
        if minimum > 0:
            return max(precise, minimum)
        return precise

    def place(
        self,
        side: str,
        n_contracts: float,
        last_price: Optional[float],
        reduce_only: bool,
        pos_side: str | None,
    ):
        qty = self._normalise_amount(n_contracts)
        position_side = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_market_order(
                side=side.lower().strip(),
                amount=qty,
                reduce_only=reduce_only,
                pos_side=position_side,
            )
            filled_raw = order.get("filled", order.get("amount", qty))
            try:
                filled = float(filled_raw)
            except (TypeError, ValueError):
                filled = float(qty)

            avg_candidates = (order.get("average"), order.get("price"), last_price)
            avg_price = None
            for candidate in avg_candidates:
                if candidate not in (None, ""):
                    avg_price = candidate
                    break
            try:
                avg = float(avg_price) if avg_price is not None else float(last_price or 0.0)
            except (TypeError, ValueError):
                avg = float(last_price or 0.0)

            fee_cost = 0.0
            if isinstance(order.get("fees"), list) and order["fees"]:
                fee_cost = sum(float(f.get("cost", 0) or 0) for f in order["fees"])

            order_id = order.get("id", "")
            self.logger.info(
                f"[Binance] market order id={order_id} side={side} qty={filled:.6f} avg={avg:.6f} fee={fee_cost}"
            )
            return {
                "status": "ok",
                "side": side,
                "amount": filled if filled else qty,
                "price": avg,
                "fee": fee_cost,
                "order_id": order_id,
            }
        except Exception as exc:
            self.logger.exception(f"[Binance] market order failed: {exc}")
            return {"status": "error", "reason": str(exc)}

    def place_stop(self, side: str, amount: float, trigger_price: float, pos_side: str | None):
        qty = self._normalise_amount(amount)
        position_side = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_stop_market_order(
                side=side.lower().strip(),
                amount=qty,
                trigger_price=trigger_price,
                pos_side=position_side,
            )
            order_id = order.get("id", "")
            self.logger.info(f"[Binance] stop order id={order_id} trigger={trigger_price:.6f}")
            return {"status": "ok", "order_id": order_id}
        except Exception as exc:
            self.logger.error(f"Stop order failed: {exc}")
            return {"status": "error", "reason": str(exc)}

    def cancel_all_conditional(self):
        return self.exch.cancel_all_conditional_orders()

    def open_long(self, amt: float, last_price: float):
        position_side = "long" if self.cfg.position_mode.lower() == "hedge" else None
        return self.place("buy", amt, last_price, reduce_only=False, pos_side=position_side)

    def open_short(self, amt: float, last_price: float):
        position_side = "short" if self.cfg.position_mode.lower() == "hedge" else None
        return self.place("sell", amt, last_price, reduce_only=False, pos_side=position_side)

    def close_long(self, amt: float, last_price: float):
        if amt <= 0:
            return None
        position_side = "long" if self.cfg.position_mode.lower() == "hedge" else None
        return self.place("sell", amt, last_price, reduce_only=True, pos_side=position_side)

    def close_short(self, amt: float, last_price: float):
        if amt <= 0:
            return None
        position_side = "short" if self.cfg.position_mode.lower() == "hedge" else None
        return self.place("buy", amt, last_price, reduce_only=True, pos_side=position_side)

    def close_long_all(self, last_price: float, long_amount: float):
        if long_amount <= 0:
            return None
        return self.close_long(long_amount, last_price)

    def close_short_all(self, last_price: float, short_amount: float):
        if short_amount <= 0:
            return None
        return self.close_short(short_amount, last_price)
