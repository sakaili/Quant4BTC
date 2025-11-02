from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from strategies.base import Strategy

if TYPE_CHECKING:
    from config import Config
    from csv_logger import CsvLogger
    from data import DataFetcher
    from indicators import IndicatorEngine
    from order_executor import OrderExecutor
    from position_reader import PositionReader
    from selector import FactorSelector
    from signals import SignalBuilder


class ThermostatStrategy(Strategy):
    """恒温器策略：CMI 判势，趋势用布林带，震荡用 KD。"""

    def __init__(
        self,
        cfg: Config,
        logger,
        data_fetcher: DataFetcher,
        indicator_engine: IndicatorEngine,
        signal_builder: SignalBuilder,
        factor_selector: FactorSelector,
        order_executor: OrderExecutor,
        position_reader: PositionReader,
        csv_logger: CsvLogger,
    ) -> None:
        super().__init__(cfg, logger, order_executor, position_reader, csv_logger)
        self.fetcher = data_fetcher
        self.ind = indicator_engine
        self.sbuilder = signal_builder
        self.selector = factor_selector
        self._mode: Optional[str] = None
        self._trend_entry_price: Optional[float] = None
        self._range_entry_price: Optional[float] = None

    def _determine_contracts(self) -> int:
        base = int(max(1, getattr(self.cfg, "contracts_per_order", 1)))
        try:
            min_contracts = int(max(1, self.exec.exch.min_contracts()))
        except Exception:
            min_contracts = 1
        return max(base, min_contracts)

    @staticmethod
    def _infer_fill(resp, fallback: float) -> float:
        if not resp or resp.get("status") != "ok":
            return fallback
        price_raw = resp.get("price")
        try:
            return float(price_raw)
        except (TypeError, ValueError):
            return fallback

    def run_once(self) -> None:
        df = self.fetcher.fetch_ohlcv_df()
        if df.empty:
            self.logger.warning("未获取到数据，跳过本轮")
            return
        df = self.fetcher.drop_unclosed_tail(df)

        cmi_period = max(1, int(getattr(self.cfg, "thermostat_cmi_period", 30)))
        bb_period = max(1, int(getattr(self.cfg, "thermostat_bb_period", 20)))
        kd_period = max(1, int(getattr(self.cfg, "thermostat_kd_period", 14)))
        min_required = max(cmi_period, bb_period, kd_period) + 2
        if len(df) < min_required:
            self.logger.warning("数据长度不足，需至少 %s 根K线", min_required)
            return

        cmi_series = self.ind.compute_cmi(df, cmi_period)
        bb_df = self.ind.compute_bbands(df, bb_period, getattr(self.cfg, "thermostat_bb_std", 2.0))
        kd_df = self.ind.compute_kd(
            df,
            kd_period,
            getattr(self.cfg, "thermostat_kd_smooth", 3),
        )

        work = df.copy()
        work["CMI"] = cmi_series
        work = work.join(bb_df)
        work = work.join(kd_df)
        work = work.dropna(subset=["CMI", "bb_middle", "kd_k", "kd_d"])
        if work.empty:
            self.logger.warning("指标计算结果为空，跳过")
            return

        last = work.iloc[-1]
        last_close = float(last["Close"])
        cmi_value = float(last["CMI"])
        upper = float(last["bb_upper"])
        middle = float(last["bb_middle"])
        lower = float(last["bb_lower"])
        k_value = float(last["kd_k"])
        d_value = float(last["kd_d"])

        cmi_threshold = float(getattr(self.cfg, "thermostat_cmi_threshold", 20.0))
        mode = "trend" if cmi_value >= cmi_threshold else "range"

        kd_low = float(getattr(self.cfg, "thermostat_kd_low", 30.0))
        kd_high = float(getattr(self.cfg, "thermostat_kd_high", 70.0))
        tp_pct = max(0.0, float(getattr(self.cfg, "thermostat_range_take_profit_pct", 0.0)))
        allow_short = self.cfg.mode.lower() != "long_flat"

        long_amt, short_amt = self.pos_reader._hedge_amounts()
        net_sign = self.pos_reader.net_sign()

        equity = self.exec.account_equity()
        draw_state = self._assess_drawdown(equity)
        if draw_state:
            msg = {
                "overall": "Thermostat: 触发总回撤 Kill Switch，清仓退出",
                "daily_trigger": "Thermostat: 当日回撤触发，暂停 24 小时",
                "daily_active": "Thermostat: 暂停冷却中，保持空仓",
            }[draw_state]
            self.logger.error(msg)
            self._flatten_positions(long_amt, short_amt, last_close)
            self.exec.cancel_all_conditional()
            self._trend_entry_price = None
            self._range_entry_price = None
            self._mode = mode
            return

        actions: list[str] = []
        prices: list[float] = []
        fees: list[float] = []
        order_ids: list[str] = []

        def record(resp, label: str) -> Optional[float]:
            if not resp:
                return None
            if resp.get("status") == "ok":
                actions.append(label)
                price_field = resp.get("price")
                if price_field is not None:
                    try:
                        price_val = float(price_field)
                        prices.append(price_val)
                    except (TypeError, ValueError):
                        price_val = None
                else:
                    price_val = None
                fee_field = resp.get("fee")
                if fee_field is not None:
                    try:
                        fees.append(float(fee_field))
                    except (TypeError, ValueError):
                        pass
                if resp.get("order_id"):
                    order_ids.append(str(resp["order_id"]))
                return price_val
            return None

        previous_mode = self._mode
        if previous_mode and previous_mode != mode:
            if previous_mode == "trend" and mode == "range":
                if long_amt > 0:
                    resp = self.exec.close_long(long_amt, last_close)
                    record(resp, f"mode_switch_close_long_{long_amt}")
                    long_amt = 0
                if short_amt > 0:
                    resp = self.exec.close_short(short_amt, last_close)
                    record(resp, f"mode_switch_close_short_{short_amt}")
                    short_amt = 0
                self.exec.cancel_all_conditional()
                net_sign = 0
                self._trend_entry_price = None
            elif previous_mode == "range" and mode == "trend":
                trend_dir = 0
                if last_close >= upper:
                    trend_dir = 1
                elif last_close <= lower and allow_short:
                    trend_dir = -1
                if net_sign != 0:
                    aligned = trend_dir != 0 and net_sign == trend_dir
                    if not aligned:
                        if long_amt > 0:
                            resp = self.exec.close_long(long_amt, last_close)
                            record(resp, f"mode_switch_close_long_{long_amt}")
                            long_amt = 0
                        if short_amt > 0:
                            resp = self.exec.close_short(short_amt, last_close)
                            record(resp, f"mode_switch_close_short_{short_amt}")
                            short_amt = 0
                        self.exec.cancel_all_conditional()
                        net_sign = 0
                self._range_entry_price = None

        target_signal = net_sign
        desired_contracts = max(long_amt, short_amt)

        if mode == "trend":
            if target_signal > 0:
                if last_close <= middle:
                    target_signal = 0
                else:
                    target_signal = 1
            elif target_signal < 0:
                if last_close >= middle:
                    target_signal = 0
                else:
                    target_signal = -1
            else:
                if last_close >= upper:
                    target_signal = 1
                elif allow_short and last_close <= lower:
                    target_signal = -1
                else:
                    target_signal = 0
            if target_signal == -1 and not allow_short:
                target_signal = 0
            desired_contracts = self._determine_contracts() if target_signal != 0 else 0
        else:
            if target_signal > 0:
                exit_long = (k_value < d_value) or (d_value >= kd_high)
                if self._range_entry_price and tp_pct > 0:
                    exit_long = exit_long or (last_close >= self._range_entry_price * (1.0 + tp_pct))
                if exit_long:
                    target_signal = 0
            elif target_signal < 0:
                exit_short = (k_value > d_value) or (d_value <= kd_low)
                if self._range_entry_price and tp_pct > 0:
                    exit_short = exit_short or (last_close <= self._range_entry_price * (1.0 - tp_pct))
                if exit_short:
                    target_signal = 0
            else:
                if k_value > d_value and d_value < kd_low:
                    target_signal = 1
                elif allow_short and k_value < d_value and d_value > kd_high:
                    target_signal = -1
                else:
                    target_signal = 0
            if target_signal == -1 and not allow_short:
                target_signal = 0
            desired_contracts = self._determine_contracts() if target_signal != 0 else 0

        desired_long = desired_contracts if target_signal == 1 else 0
        desired_short = desired_contracts if target_signal == -1 else 0

        reduce_long = max(0, long_amt - desired_long)
        if reduce_long > 0:
            resp = self.exec.close_long(reduce_long, last_close)
            record(resp, f"close_long_{reduce_long}")
            long_amt -= reduce_long
        reduce_short = max(0, short_amt - desired_short)
        if reduce_short > 0:
            resp = self.exec.close_short(reduce_short, last_close)
            record(resp, f"close_short_{reduce_short}")
            short_amt -= reduce_short

        current_long = long_amt
        current_short = short_amt

        add_long = max(0, desired_long - current_long)
        if add_long > 0:
            resp = self.exec.open_long(add_long, last_close)
            fill = record(resp, f"open_long_{add_long}")
            current_long += add_long
            price = fill if fill is not None else last_close
            if mode == "range":
                self._range_entry_price = price
            else:
                self._trend_entry_price = price
        elif desired_long == 0:
            if mode == "range":
                self._range_entry_price = None
            else:
                self._trend_entry_price = None

        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            resp = self.exec.open_short(add_short, last_close)
            fill = record(resp, f"open_short_{add_short}")
            current_short += add_short
            price = fill if fill is not None else last_close
            if mode == "range":
                self._range_entry_price = price
            else:
                self._trend_entry_price = price
        elif desired_short == 0 and current_short == 0:
            if mode == "range":
                self._range_entry_price = None
            else:
                self._trend_entry_price = None

        if target_signal == 0:
            self._trend_entry_price = None
            self._range_entry_price = None

        action_str = "|".join(actions) if actions else None
        exec_price = prices[-1] if prices else None
        fee_sum = sum(fees) if fees else None
        order_id = "|".join(order_ids) if order_ids else None

        mode_str = "OKX-DEMO-SWAP" if self.cfg.use_demo else "OKX-SWAP"
        self.csv.append(
            {
                "timestamp": datetime.now(),
                "signal": target_signal,
                "close": last_close,
                "position": self.pos_reader.net_sign(),
                "action": action_str,
                "exec_price": exec_price,
                "fee": fee_sum,
                "order_id": order_id,
                "stop_loss": None,
                "take_profit": None,
                "best_factor": None,
                "equity": equity,
                "mode": mode_str,
            }
        )

        self.logger.info(
            "Thermostat mode=%s CMI=%.2f close=%.2f K=%.2f D=%.2f target=%s upper=%.2f lower=%.2f",
            mode,
            cmi_value,
            last_close,
            k_value,
            d_value,
            target_signal,
            upper,
            lower,
        )

        self._mode = mode
