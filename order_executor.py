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
        quote_ccy = self.cfg.symbol.split("/")[-1].split(":")[0].upper()
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

    def place_stop(self, side: str, amount: float, trigger_price: float, pos_side: str | None, reduce_only: bool = True):
        qty = self._normalise_amount(amount)
        position_side = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_stop_market_order(
                side=side.lower().strip(),
                amount=qty,
                trigger_price=trigger_price,
                pos_side=position_side,
                reduce_only=reduce_only,
            )
            order_id = order.get("id", "")
            order_type = "stop_loss" if reduce_only else "stop_reverse"
            self.logger.info(f"[Binance] {order_type} order id={order_id} trigger={trigger_price:.6f}")
            return {"status": "ok", "order_id": order_id}
        except Exception as exc:
            self.logger.error(f"Stop order failed: {exc}")
            return {"status": "error", "reason": str(exc)}

    def place_take_profit(self, side: str, amount: float, trigger_price: float, pos_side: str | None, reduce_only: bool = True):
        """Place a take profit order."""
        qty = self._normalise_amount(amount)
        position_side = pos_side.lower().strip() if pos_side else None
        try:
            order = self.exch.create_take_profit_order(
                side=side.lower().strip(),
                amount=qty,
                trigger_price=trigger_price,
                pos_side=position_side,
                reduce_only=reduce_only,
            )
            order_id = order.get("id", "")
            self.logger.info(f"[Binance] take_profit order id={order_id} trigger={trigger_price:.6f}")
            return {"status": "ok", "order_id": order_id}
        except Exception as exc:
            self.logger.error(f"Take profit order failed: {exc}")
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

    def get_bbo(self) -> dict:
        """Get Best Bid and Best Offer (BBO) prices from order book.

        Returns:
            dict: {
                "bid": float,  # Best bid price
                "ask": float,  # Best ask price
                "bid_size": float,  # Best bid quantity
                "ask_size": float,  # Best ask quantity
            }
        """
        try:
            orderbook = self.exch.fetch_order_book(limit=5)
            if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
                raise ValueError("Invalid order book response")

            if not orderbook["bids"] or not orderbook["asks"]:
                raise ValueError("Empty order book")

            best_bid = orderbook["bids"][0]
            best_ask = orderbook["asks"][0]

            return {
                "bid": float(best_bid[0]),
                "ask": float(best_ask[0]),
                "bid_size": float(best_bid[1]),
                "ask_size": float(best_ask[1]),
            }
        except Exception as exc:
            self.logger.error(f"Failed to fetch BBO: {exc}")
            raise

    def place_limit_order(
        self,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool,
        pos_side: str | None,
    ) -> dict:
        """Place a limit order (Maker order).

        Args:
            side: "buy" or "sell"
            amount: Order quantity in contracts
            price: Limit price
            reduce_only: True for closing positions only
            pos_side: "long" or "short" for hedge mode, None for net mode

        Returns:
            dict: {
                "status": "ok" or "error",
                "order_id": str,
                "reason": str (only if error),
            }
        """
        qty = self._normalise_amount(amount)
        position_side = pos_side.lower().strip() if pos_side else None

        try:
            order = self.exch.create_limit_order(
                side=side.lower().strip(),
                amount=qty,
                price=price,
                reduce_only=reduce_only,
                pos_side=position_side,
            )
            order_id = order.get("id", "")
            self.logger.info(
                f"[Binance] limit order id={order_id} side={side} qty={qty:.6f} price={price:.6f}"
            )
            return {"status": "ok", "order_id": order_id}
        except Exception as exc:
            self.logger.error(f"Limit order failed: {exc}")
            return {"status": "error", "reason": str(exc)}

    def check_order_status(self, order_id: str) -> dict:
        """Check the status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            dict: {
                "status": "open" | "closed" | "canceled" | "error",
                "filled": float,  # Filled quantity
                "remaining": float,  # Remaining quantity
                "price": float,  # Average fill price (if filled)
                "reason": str,  # Error reason if status is "error"
            }
        """
        try:
            order = self.exch.fetch_order(order_id)

            status = order.get("status", "").lower()
            filled = float(order.get("filled", 0.0))
            remaining = float(order.get("remaining", 0.0))
            avg_price = float(order.get("average", 0.0) or 0.0)

            return {
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "price": avg_price,
            }
        except Exception as exc:
            self.logger.error(f"Failed to check order status: {exc}")
            return {"status": "error", "reason": str(exc)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if successfully canceled, False otherwise
        """
        try:
            self.exch.cancel_order(order_id)
            self.logger.info(f"[Binance] Canceled order id={order_id}")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to cancel order {order_id}: {exc}")
            return False
