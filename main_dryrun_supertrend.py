from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from typing import Optional

from config import Config, setup_logging
from csv_logger import CsvLogger
from data import DataFetcher
from exchange_client import ExchangeClient
from indicators import IndicatorEngine
from selector import FactorSelector
from signals import SignalBuilder
from strategies.supertrend import SuperTrendStrategy


TAKE_PROFIT_PCT = 0.016  # +1.6%
STOP_LOSS_PCT = 0.008  # -0.8%


class _StubExchange:
    def __init__(self, min_contracts: float):
        self._min_contracts = max(0.0, float(min_contracts))

    def min_contracts(self) -> float:
        return self._min_contracts


class DryRunLedger:
    """Track dry-run equity, positions, and stop orders."""

    def __init__(self, cfg: Config, logger):
        self.cfg = cfg
        self.logger = logger
        self.initial_capital = float(cfg.initial_capital)
        self.cash = self.initial_capital
        self.last_price: Optional[float] = None
        self.long_amt = 0.0
        self.short_amt = 0.0
        self.long_entry: Optional[float] = None
        self.short_entry: Optional[float] = None
        self.long_tp: Optional[float] = None
        self.long_sl: Optional[float] = None
        self.short_tp: Optional[float] = None
        self.short_sl: Optional[float] = None
        self.cost_rate = float(cfg.fee_rate + cfg.slippage_rate)

    def equity(self) -> float:
        unreal = self._unrealized_pnl()
        return self.cash + unreal

    def _unrealized_pnl(self) -> float:
        if self.last_price is None:
            return 0.0
        pnl = 0.0
        if self.long_amt > 0 and self.long_entry:
            pnl += (self.last_price - self.long_entry) * self.long_amt
        if self.short_amt > 0 and self.short_entry:
            pnl += (self.short_entry - self.last_price) * self.short_amt
        return pnl

    def update_price(self, price: float) -> None:
        self.last_price = price
        self._maybe_flatten_long(price)
        self._maybe_flatten_short(price)

    def _maybe_flatten_long(self, price: float) -> None:
        if self.long_amt <= 0 or self.long_entry is None:
            return
        if self.long_sl and price <= self.long_sl:
            self.close_long(self.long_amt, self.long_sl, reason="stop_loss")
        elif self.long_tp and price >= self.long_tp:
            self.close_long(self.long_amt, self.long_tp, reason="take_profit")

    def _maybe_flatten_short(self, price: float) -> None:
        if self.short_amt <= 0 or self.short_entry is None:
            return
        if self.short_sl and price >= self.short_sl:
            self.close_short(self.short_amt, self.short_sl, reason="stop_loss")
        elif self.short_tp and price <= self.short_tp:
            self.close_short(self.short_amt, self.short_tp, reason="take_profit")

    def open_long(self, qty: float, price: float) -> Optional[float]:
        qty = max(0.0, float(qty))
        if qty <= 0 or price <= 0:
            return None
        self._apply_cost(price, qty)
        total = self.long_amt + qty
        if self.long_amt <= 0 or self.long_entry is None:
            self.long_entry = price
        else:
            self.long_entry = ((self.long_entry * self.long_amt) + (price * qty)) / total
        self.long_amt = total
        self.long_sl = self.long_entry * (1.0 - STOP_LOSS_PCT)
        self.long_tp = self.long_entry * (1.0 + TAKE_PROFIT_PCT)
        self.logger.info(
            "[DRY] Open LONG qty=%.6f price=%.4f tp=%.4f sl=%.4f",
            qty,
            price,
            self.long_tp,
            self.long_sl,
        )
        return price

    def open_short(self, qty: float, price: float) -> Optional[float]:
        qty = max(0.0, float(qty))
        if qty <= 0 or price <= 0:
            return None
        self._apply_cost(price, qty)
        total = self.short_amt + qty
        if self.short_amt <= 0 or self.short_entry is None:
            self.short_entry = price
        else:
            self.short_entry = ((self.short_entry * self.short_amt) + (price * qty)) / total
        self.short_amt = total
        self.short_sl = self.short_entry * (1.0 + STOP_LOSS_PCT)
        self.short_tp = self.short_entry * (1.0 - TAKE_PROFIT_PCT)
        self.logger.info(
            "[DRY] Open SHORT qty=%.6f price=%.4f tp=%.4f sl=%.4f",
            qty,
            price,
            self.short_tp,
            self.short_sl,
        )
        return price

    def close_long(self, qty: float, price: float, reason: str = "manual") -> Optional[float]:
        qty = min(float(qty), self.long_amt)
        if qty <= 0 or price <= 0 or self.long_entry is None:
            return None
        fee = self._fee(price, qty)
        pnl = ((price - self.long_entry) * qty) - fee
        self.cash += pnl
        self.long_amt -= qty
        if self.long_amt <= 1e-9:
            self.long_amt = 0.0
            self.long_entry = None
            self.long_sl = None
            self.long_tp = None
        self.logger.info(
            "[DRY] Close LONG qty=%.6f price=%.4f reason=%s pnl=%.4f",
            qty,
            price,
            reason,
            pnl,
        )
        return price

    def close_short(self, qty: float, price: float, reason: str = "manual") -> Optional[float]:
        qty = min(float(qty), self.short_amt)
        if qty <= 0 or price <= 0 or self.short_entry is None:
            return None
        fee = self._fee(price, qty)
        pnl = ((self.short_entry - price) * qty) - fee
        self.cash += pnl
        self.short_amt -= qty
        if self.short_amt <= 1e-9:
            self.short_amt = 0.0
            self.short_entry = None
            self.short_sl = None
            self.short_tp = None
        self.logger.info(
            "[DRY] Close SHORT qty=%.6f price=%.4f reason=%s pnl=%.4f",
            qty,
            price,
            reason,
            pnl,
        )
        return price

    def _apply_cost(self, price: float, qty: float) -> None:
        fee = self._fee(price, qty)
        self.cash -= fee

    def _fee(self, price: float, qty: float) -> float:
        return price * qty * self.cost_rate


class DryRunPositionReader:
    """Expose ledger positions through the interface strategy expects."""

    def __init__(self, ledger: DryRunLedger):
        self.ledger = ledger

    def _hedge_amounts(self) -> tuple[float, float]:
        return float(self.ledger.long_amt), float(self.ledger.short_amt)

    def net_sign(self) -> int:
        long_amt, short_amt = self._hedge_amounts()
        net = long_amt - short_amt
        if abs(net) < 1e-9:
            return 0
        return 1 if net > 0 else -1


class DryRunOrderExecutor:
    """Order executor that simulates fills and enforces fixed TP/SL."""

    def __init__(self, cfg: Config, logger, ledger: DryRunLedger):
        self.cfg = cfg
        self.logger = logger
        self.ledger = ledger
        self._order_seq = 0
        self._market_info = {"contractSize": 1.0}
        self.exch = _StubExchange(min_contracts=0.0)

    def _next_order_id(self, prefix: str) -> str:
        self._order_seq += 1
        return f"{prefix}-{self._order_seq:06d}"

    def market_info(self):
        return self._market_info

    def account_snapshot(self) -> dict:
        equity = self.ledger.equity()
        free = max(0.0, self.ledger.cash)
        return {"equity": equity, "free": free, "used": max(0.0, equity - free)}

    def account_equity(self) -> float:
        return self.ledger.equity()

    def available_margin(self) -> float:
        return max(0.0, self.ledger.cash)

    def on_price_tick(self, price: float) -> None:
        self.ledger.update_price(price)

    def open_long(self, amt: float, last_price: float):
        fill = self.ledger.open_long(amt, last_price)
        if fill is None:
            return {"status": "error", "reason": "open_long_failed"}
        return {
            "status": "ok",
            "side": "buy",
            "amount": float(amt),
            "price": float(fill),
            "order_id": self._next_order_id("BUY"),
            "fee": None,
        }

    def open_short(self, amt: float, last_price: float):
        fill = self.ledger.open_short(amt, last_price)
        if fill is None:
            return {"status": "error", "reason": "open_short_failed"}
        return {
            "status": "ok",
            "side": "sell",
            "amount": float(amt),
            "price": float(fill),
            "order_id": self._next_order_id("SELL"),
            "fee": None,
        }

    def close_long(self, amt: float, last_price: float):
        fill = self.ledger.close_long(amt, last_price)
        if fill is None:
            return {"status": "skip"}
        return {
            "status": "ok",
            "side": "sell",
            "amount": float(amt),
            "price": float(fill),
            "order_id": self._next_order_id("CLS"),
            "fee": None,
        }

    def close_short(self, amt: float, last_price: float):
        fill = self.ledger.close_short(amt, last_price)
        if fill is None:
            return {"status": "skip"}
        return {
            "status": "ok",
            "side": "buy",
            "amount": float(amt),
            "price": float(fill),
            "order_id": self._next_order_id("CLS"),
            "fee": None,
        }

    def close_long_all(self, last_price: float, long_amount: float):
        return self.close_long(long_amount, last_price)

    def close_short_all(self, last_price: float, short_amount: float):
        return self.close_short(short_amount, last_price)

    def cancel_all_conditional(self):
        return {"status": "ok", "cancelled": 0}

    def place_stop(self, *args, **kwargs):
        self.logger.info("[DRY] place_stop ignored (simulated)")
        return {"status": "ok", "order_id": self._next_order_id("STOP")}

    def place_take_profit(self, *args, **kwargs):
        self.logger.info("[DRY] place_take_profit ignored (simulated)")
        return {"status": "ok", "order_id": self._next_order_id("TP")}


def _wait_for_next_candle(cfg: Config, logger) -> None:
    tf_sec = DataFetcher.timeframe_seconds(cfg.timeframe)
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    remain = tf_sec - (epoch % tf_sec)
    if remain > 0:
        logger.info("Waiting %.2fs to sync with next %s candle", remain, cfg.timeframe)
        time.sleep(remain)


def _install_signal_handlers(logger):
    def _handler(signum, _frame):
        logger.info("Received signal %s, exiting dry-run loop", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main():
    cfg = Config()
    base_logger = setup_logging(cfg.log_level).getChild("dryrun_supertrend")
    _install_signal_handlers(base_logger)

    exchange = ExchangeClient(cfg, base_logger)
    data_fetcher = DataFetcher(cfg, exchange, base_logger)
    indicator_engine = IndicatorEngine(cfg)
    signal_builder = SignalBuilder(cfg)
    factor_selector = FactorSelector(cfg)
    ledger = DryRunLedger(cfg, base_logger)
    order_executor = DryRunOrderExecutor(cfg, base_logger, ledger)
    position_reader = DryRunPositionReader(ledger)
    csv_logger = CsvLogger(f"dryrun_{cfg.csv_log_file}", base_logger)

    strategy = SuperTrendStrategy(
        cfg=cfg,
        logger=base_logger,
        data_fetcher=data_fetcher,
        indicator_engine=indicator_engine,
        signal_builder=signal_builder,
        factor_selector=factor_selector,
        order_executor=order_executor,
        position_reader=position_reader,
        csv_logger=csv_logger,
    )

    _wait_for_next_candle(cfg, base_logger)

    tf_sec = DataFetcher.timeframe_seconds(cfg.timeframe)
    base_logger.info(
        "Starting dry-run loop for %s @ %s (TP=+%.2f%%, SL=-%.2f%%)",
        cfg.symbol,
        cfg.timeframe,
        TAKE_PROFIT_PCT * 100,
        STOP_LOSS_PCT * 100,
    )

    while True:
        start = time.time()
        try:
            strategy.run_once(equity=ledger.equity())
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            base_logger.error("Dry-run cycle failed: %s", exc, exc_info=True)
        elapsed = time.time() - start
        sleep_time = max(0.0, tf_sec - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
