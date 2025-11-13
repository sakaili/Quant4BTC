from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Callable, Optional


class Strategy(ABC):
    """Base class providing shared lifecycle, risk control, and logging."""

    def __init__(self, cfg, logger, order_executor, position_reader, csv_logger):
        self.cfg = cfg
        self.logger = logger
        self.exec = order_executor
        self.pos_reader = position_reader
        self.csv = csv_logger

        self._daily_anchor_equity: Optional[float] = None
        self._daily_anchor_date: Optional[date] = None
        self._daily_suspend_until: Optional[datetime] = None
        self._global_kill = False

    @abstractmethod
    def run_once(self) -> None:
        """Execute one strategy cycle."""

    def align_and_loop(self, align_fn: Callable[[], None]) -> None:
        mode_str = "Binance Testnet" if self.cfg.use_demo else "Binance Futures"
        self.logger.info(
            "启动循环 MODE=%s, SYMBOL=%s, TF=%s, SELECTION=%s, STRAT_MODE=%s",
            mode_str,
            self.cfg.symbol,
            self.cfg.timeframe,
            getattr(self.cfg, "selection", "-"),
            getattr(self.cfg, "mode", "-"),
        )

        try:
            self.run_once()
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("首次 run_once 失败: %s", exc)

        while True:
            align_fn()
            if self._global_kill:
                self.logger.error("全局 Kill Switch 已触发, 停止循环")
                break
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("运行异常: %s", exc)
                time.sleep(5)

    def _assess_drawdown(self, equity: float) -> Optional[str]:
        if equity <= 0:
            return None

        now = datetime.utcnow()
        if self._daily_anchor_date != now.date():
            self._daily_anchor_date = now.date()
            self._daily_anchor_equity = equity
            self._daily_suspend_until = None
        else:
            self._daily_anchor_equity = max(self._daily_anchor_equity or equity, equity)

        overall_floor = self.cfg.initial_capital * (1.0 - self.cfg.overall_drawdown_limit)
        if equity <= overall_floor:
            self._global_kill = False
            return "overall"

        if self._daily_suspend_until and now < self._daily_suspend_until:
            return "daily_active"

        anchor = self._daily_anchor_equity or equity
        daily_floor = anchor * (1.0 - self.cfg.daily_drawdown_limit)
        if equity <= daily_floor:
            self._daily_suspend_until = now + timedelta(hours=24)
            return "daily_trigger"
        return None

    def _flatten_positions(self, long_amt: int, short_amt: int, last_price: float) -> None:
        if long_amt > 0:
            self.logger.info("KillSwitch: 平掉多头 %s 张", long_amt)
            self.exec.close_long(long_amt, last_price)
        if short_amt > 0:
            self.logger.info("KillSwitch: 平掉空头 %s 张", short_amt)
            self.exec.close_short(short_amt, last_price)
        self.exec.cancel_all_conditional()
