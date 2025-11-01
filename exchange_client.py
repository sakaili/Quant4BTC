# exchange_client.py
import ccxt
from typing import Any, Dict

from config import Config


class ExchangeClient:
    """Thin wrapper around ccxt for Binance USD-M futures."""

    def __init__(self, cfg: Config, logger):
        self.cfg = cfg
        self.logger = logger
        options = {
            "apiKey": cfg.binance_api_key,
            "secret": cfg.binance_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "defaultSubType": "linear",
                "adjustForTimeDifference": True,
                "recvWindow": cfg.binance_recv_window,
                "testnet": bool(cfg.use_demo),
            },
            "proxies": cfg.proxies(),
            "timeout": 20000,
        }

        self.exchange = ccxt.binanceusdm(options)

        try:
            # ccxt toggles sandbox endpoints for binance via this helper
            self.exchange.set_sandbox_mode(bool(cfg.use_demo))
        except Exception as exc:  # pragma: no cover - depends on ccxt version
            self.logger.warning(f"Failed to enable Binance sandbox mode: {exc}")

        self.exchange.load_markets()

        try:
            hedged = cfg.position_mode.lower() == "hedge"
            # ccxt normalises hedge=True/False for Binance dual-side mode
            self.exchange.set_position_mode(hedged, symbol=cfg.symbol)
            self.logger.info(f"Set POSITION_MODE={cfg.position_mode}")
        except Exception as exc:
            self.logger.warning(f"Unable to set position mode: {exc}")

        try:
            margin_mode = cfg.margin_mode.lower()
            if margin_mode in {"isolated", "cross"}:
                self.exchange.set_margin_mode(margin_mode, symbol=cfg.symbol)
                self.logger.info(f"Set MARGIN_MODE={cfg.margin_mode}")
        except Exception as exc:
            self.logger.warning(f"Unable to set margin mode: {exc}")

        try:
            self.exchange.set_leverage(cfg.leverage, symbol=cfg.symbol)
            self.logger.info(f"Set LEVERAGE={cfg.leverage}")
        except Exception as exc:
            self.logger.warning(f"Unable to set leverage: {exc}")

        self.market_info = self.exchange.market(cfg.symbol)

    def amount_to_precision(self, n: float) -> float:
        """Clamp contract amount to exchange precision."""

        try:
            return float(self.exchange.amount_to_precision(self.cfg.symbol, n))
        except Exception:
            return float(n)

    def price_to_precision(self, p: float) -> float:
        """Clamp price to the instrument tick size."""

        return float(self.exchange.price_to_precision(self.cfg.symbol, p))

    def min_contracts(self) -> float:
        """Return the minimum order quantity supported by the market."""

        try:
            min_amt = self.market_info.get("limits", {}).get("amount", {}).get("min")
            if min_amt is not None:
                return float(min_amt)
        except Exception:
            pass
        return 0.0

    # ---------- Market data ----------
    def fetch_ohlcv(self, limit: int):
        """Fetch OHLCV data using the configured timeframe."""

        return self.exchange.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, limit=limit)

    def fetch_ticker_last(self) -> float:
        ticker = self.exchange.fetch_ticker(self.cfg.symbol)
        return float(ticker["last"])

    def fetch_balance_swap(self) -> Dict[str, Any] | None:
        """Fetch futures account balances."""

        try:
            return self.exchange.fetch_balance({"type": "future"})
        except Exception as exc:
            self.logger.error(f"Failed to fetch futures balance: {exc}")
            return None

    def fetch_positions(self):
        try:
            return self.exchange.fetch_positions([self.cfg.symbol])
        except Exception as exc:
            self.logger.error(f"Failed to fetch positions: {exc}")
            return []

    # ---------- Trading ----------
    def _hedge_side_param(self, pos_side: str | None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.cfg.position_mode.lower() == "hedge" and pos_side in {"long", "short"}:
            params["positionSide"] = "LONG" if pos_side == "long" else "SHORT"
        return params

    def create_market_order(
        self,
        side: str,
        amount: float,
        reduce_only: bool,
        pos_side: str | None,
    ):
        params: Dict[str, Any] = {}
        hedge_params = self._hedge_side_param(pos_side)
        params.update(hedge_params)
        if reduce_only and not hedge_params:
            params["reduceOnly"] = True
        return self.exchange.create_order(
            symbol=self.cfg.symbol,
            type="market",
            side=side,
            amount=amount,
            price=None,
            params=params,
        )

    def create_stop_market_order(
        self,
        side: str,
        amount: float,
        trigger_price: float,
        pos_side: str | None,
    ):
        params: Dict[str, Any] = {
            "stopPrice": float(self.price_to_precision(trigger_price)),
            "workingType": "CONTRACT_PRICE",
        }
        hedge_params = self._hedge_side_param(pos_side)
        params.update(hedge_params)
        if not hedge_params:
            params["reduceOnly"] = True
        return self.exchange.create_order(
            symbol=self.cfg.symbol,
            type="STOP_MARKET",
            side=side,
            amount=amount,
            price=None,
            params=params,
        )

    def cancel_all_conditional_orders(self):
        try:
            cancelled = self.exchange.cancel_all_orders(self.cfg.symbol)
            return {"status": "ok", "cancelled": len(cancelled) if cancelled else 0}
        except Exception as exc:
            self.logger.error(f"Failed to cancel stop orders: {exc}")
            return {"status": "error", "reason": str(exc)}
