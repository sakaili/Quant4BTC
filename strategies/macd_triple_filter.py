from __future__ import annotations

from datetime import datetime
from math import floor
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy

if TYPE_CHECKING:
    from config import Config
    from csv_logger import CsvLogger
    from data import DataFetcher
    from indicators import IndicatorEngine
    from order_executor import OrderExecutor
    from position_reader import PositionReader


def _compute_stop_loss(cfg: "Config", last_close: float, atr: float, signal: int) -> Optional[float]:
    mult = max(0.0, float(getattr(cfg, "macd_atr_stop_multiple", 0.0)))
    if signal == 0 or mult <= 0 or atr <= 0 or last_close <= 0:
        return None
    if signal > 0:
        return last_close - atr * mult
    if signal < 0:
        return last_close + atr * mult
    return None


def build_macd_triple_filter_timeseries(
    df: pd.DataFrame,
    cfg: "Config",
    indicator: "IndicatorEngine",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df_atr = indicator.compute_atr(df)
    fast = int(getattr(cfg, "macd_fast_length", 12))
    slow = int(getattr(cfg, "macd_slow_length", 26))
    signal_len = int(getattr(cfg, "macd_signal_length", 9))
    regime_ma = int(getattr(cfg, "macd_regime_ma_length", 200))
    confirm_bars = max(1, int(getattr(cfg, "macd_hist_confirm_bars", 2)))
    min_rows = max(slow + signal_len, regime_ma, confirm_bars + 1)
    if len(df_atr) < min_rows:
        return pd.DataFrame()

    macd_df = indicator.compute_macd(df_atr, fast, slow, signal_len)
    macd_df["RegimeEMA"] = indicator.compute_ema(macd_df["Close"], regime_ma)
    macd_df = macd_df.dropna(subset=["RegimeEMA", "atr", "DIF", "DEA", "Histogram"]).copy()
    if macd_df.empty:
        return macd_df

    close_series = macd_df["Close"]
    atr_series = macd_df["atr"]
    dif_series = macd_df["DIF"]
    dea_series = macd_df["DEA"]
    hist_series = macd_df["Histogram"]

    macd_df["atr_norm"] = atr_series / close_series.abs().clip(lower=1e-9)
    atr_norm_series = macd_df["atr_norm"]

    atr_min = max(0.0, float(getattr(cfg, "macd_atr_min", 0.0)))
    atr_max_raw = float(getattr(cfg, "macd_atr_max", 0.0))
    atr_max = atr_max_raw if atr_max_raw > 0 else float("inf")

    mode = getattr(cfg, "mode", "long_short").lower()

    signals = np.zeros(len(macd_df), dtype=int)
    stop_loss = np.full(len(macd_df), np.nan)
    vol_flags = np.zeros(len(macd_df), dtype=bool)
    regime_states = np.empty(len(macd_df), dtype=object)
    hist_positive_flags = np.zeros(len(macd_df), dtype=bool)
    hist_negative_flags = np.zeros(len(macd_df), dtype=bool)

    for i in range(len(macd_df)):
        close = float(close_series.iloc[i])
        atr_norm = float(atr_norm_series.iloc[i])
        vol_ok = atr_min <= atr_norm <= atr_max
        vol_flags[i] = vol_ok
        regime_state = "long" if close >= float(macd_df["RegimeEMA"].iloc[i]) else "short"
        regime_states[i] = regime_state

        if i == 0:
            continue

        dif_prev = float(dif_series.iloc[i - 1])
        dea_prev = float(dea_series.iloc[i - 1])
        dif_curr = float(dif_series.iloc[i])
        dea_curr = float(dea_series.iloc[i])

        if i >= confirm_bars - 1:
            tail = hist_series.iloc[i - confirm_bars + 1 : i + 1]
            hist_positive = bool((tail > 0).all())
            hist_negative = bool((tail < 0).all())
        else:
            hist_positive = False
            hist_negative = False
        hist_positive_flags[i] = hist_positive
        hist_negative_flags[i] = hist_negative

        cross_up = dif_prev <= dea_prev and dif_curr > dea_curr
        cross_down = dif_prev >= dea_prev and dif_curr < dea_curr

        candidate = 0
        if vol_ok:
            if cross_up and regime_state == "long" and hist_positive:
                candidate = 1
            elif mode != "long_flat" and cross_down and regime_state == "short" and hist_negative:
                candidate = -1

        prev_signal = signals[i - 1]
        if candidate != 0:
            current = candidate
        else:
            current = 0
            if prev_signal > 0:
                keep_long = vol_ok and dif_curr >= dea_curr and hist_positive and regime_state == "long"
                current = 1 if keep_long else 0
            elif prev_signal < 0 and mode != "long_flat":
                keep_short = vol_ok and dif_curr <= dea_curr and hist_negative and regime_state == "short"
                current = -1 if keep_short else 0

        if mode == "long_flat" and current < 0:
            current = 0

        signals[i] = current
        stop = _compute_stop_loss(cfg, close, float(atr_series.iloc[i]), current)
        stop_loss[i] = stop if stop is not None else np.nan

    macd_df["signal"] = signals
    macd_df["position"] = signals.astype(float)
    macd_df["stop_loss"] = stop_loss
    macd_df["vol_ok"] = vol_flags
    macd_df["regime_state"] = regime_states
    macd_df["hist_positive"] = hist_positive_flags
    macd_df["hist_negative"] = hist_negative_flags
    return macd_df


class TripleFilterMACDStrategy(Strategy):
    """MACD �������˲�����ͬѡ����Ϊ�ڲ���߲���."""

    def __init__(
        self,
        cfg: "Config",
        logger,
        data_fetcher: "DataFetcher",
        indicator_engine: "IndicatorEngine",
        signal_builder,  # noqa: ARG002 maintained for signature compatibility
        factor_selector,  # noqa: ARG002 maintained for signature compatibility
        order_executor: "OrderExecutor",
        position_reader: "PositionReader",
        csv_logger: "CsvLogger",
    ) -> None:
        super().__init__(cfg, logger, order_executor, position_reader, csv_logger)
        self.fetcher = data_fetcher
        self.ind = indicator_engine
        self._long_entry_price: float | None = None
        self._long_stop_loss: float | None = None
        self._long_take_profit: float | None = None
        self._long_qty: int = 0
        self._short_entry_price: float | None = None
        self._short_stop_loss: float | None = None
        self._short_take_profit: float | None = None
        self._short_qty: int = 0

    def _clear_long_state(self) -> None:
        self._long_entry_price = None
        self._long_stop_loss = None
        self._long_take_profit = None
        self._long_qty = 0

    def _clear_short_state(self) -> None:
        self._short_entry_price = None
        self._short_stop_loss = None
        self._short_take_profit = None
        self._short_qty = 0

    def _compute_position_size(
        self,
        signal: int,
        last_close: float,
        stop_loss: float | None,
        equity: float,
    ) -> int:
        if signal == 0:
            return 0
        if self.cfg.mode == "long_flat" and signal < 0:
            return 0
        if stop_loss is None or last_close <= 0 or equity <= 0:
            self.logger.warning("MACD: 缺少有效止损价，放弃做单")
            return 0

        market_info = self.exec.market_info()
        available_margin = self.exec.available_margin()
        contract_value = float(market_info.get("contractSize") or market_info.get("ctVal") or 1.0)
        stop_distance = last_close - stop_loss if signal > 0 else stop_loss - last_close
        if stop_distance <= 0:
            self.logger.warning("MACD: 止损距离无效，放弃做单")
            return 0

        risk_amount = equity * self.cfg.risk_per_trade
        loss_per_contract = stop_distance * contract_value
        if loss_per_contract <= 0:
            return 0
        contracts_by_risk = risk_amount / loss_per_contract

        leverage = max(float(self.cfg.leverage), 1.0)
        max_notional = equity * leverage
        per_contract_notional = max(last_close * contract_value, 1e-6)
        contracts_by_leverage = max_notional / per_contract_notional

        contracts_by_margin = available_margin / per_contract_notional if per_contract_notional > 0 else 0.0

        raw_contracts = min(contracts_by_risk, contracts_by_leverage, contracts_by_margin)
        contracts = floor(max(0.0, raw_contracts))

        min_contracts = int(max(1, self.exec.exch.min_contracts()))
        if contracts < min_contracts:
            if min_contracts > 0:
                self.logger.warning("MACD: 预估仓位不足，改用最小合约数量下单")
                return min_contracts
            return 0
        return contracts

    def run_once(self) -> None:
        df = self.fetcher.fetch_ohlcv_df()
        if df.empty:
            self.logger.warning("MACD: 未获取到真实 K 线，跳过")
            return

        df = self.fetcher.drop_unclosed_tail(df)
        ts = build_macd_triple_filter_timeseries(df, self.cfg, self.ind)
        if ts.empty:
            self.logger.warning("MACD: 数据不足以计算完整指标，跳过")
            return

        last = ts.iloc[-1]
        prev = ts.iloc[-2] if len(ts) >= 2 else last

        target_signal = int(last["signal"])
        last_close = float(last["Close"])
        atr_value = float(last["atr"])
        stop_loss = float(last["stop_loss"]) if not np.isnan(last["stop_loss"]) else None
        atr_norm = float(last["atr_norm"])
        hist_positive = bool(last.get("hist_positive", False))
        hist_negative = bool(last.get("hist_negative", False))
        dif_curr = float(last["DIF"])
        dea_curr = float(last["DEA"])
        regime_state = str(last.get("regime_state", "long"))
        vol_ok = bool(last.get("vol_ok", False))

        long_amt, short_amt = self.pos_reader._hedge_amounts()
        equity = self.exec.account_equity()
        drawdown_state = self._assess_drawdown(equity)
        if drawdown_state:
            msg = {
                "overall": "MACD: 触发总回撤上限，KillSwitch 清仓并停机",
                "daily_trigger": "MACD: 触发当日回撤上限，暂停 24 小时",
                "daily_active": "MACD: 回撤冷却中，继续空仓等待",
            }[drawdown_state]
            self.logger.error(msg)
            self._flatten_positions(long_amt, short_amt, last_close)
            self._clear_long_state()
            self._clear_short_state()
            return

        target_contracts = self._compute_position_size(target_signal, last_close, stop_loss, equity)

        desired_long = desired_short = 0
        if self.cfg.mode == "long_flat":
            desired_long = target_contracts if target_signal == 1 else 0
        else:
            if target_signal == 1:
                desired_long = target_contracts
            elif target_signal == -1:
                desired_short = target_contracts

        actions: list[str] = []
        prices: list[float] = []
        fees: list[float] = []
        order_ids: list[str] = []

        def record(resp, label):
            if resp and resp.get("status") == "ok":
                actions.append(label)
                if resp.get("price") is not None:
                    prices.append(resp["price"])
                if resp.get("fee") is not None:
                    fees.append(resp["fee"])
                if resp.get("order_id"):
                    order_ids.append(resp["order_id"])

        reduce_long = max(0, long_amt - desired_long)
        if reduce_long > 0:
            resp = self.exec.close_long(reduce_long, last_close)
            record(resp, f"close_long_{reduce_long}")
        reduce_short = max(0, short_amt - desired_short)
        if reduce_short > 0:
            resp = self.exec.close_short(reduce_short, last_close)
            record(resp, f"close_short_{reduce_short}")

        current_long = max(0, long_amt - reduce_long)
        current_short = max(0, short_amt - reduce_short)

        if current_long == 0:
            self._clear_long_state()
        if current_short == 0:
            self._clear_short_state()

        add_long = max(0, desired_long - current_long)
        if add_long > 0:
            base_long_qty = current_long
            resp = self.exec.open_long(add_long, last_close)
            record(resp, f"open_long_{add_long}")
            current_long += add_long
            fill_price: float | None = None
            if resp and resp.get("status") == "ok" and resp.get("price") is not None:
                try:
                    fill_price = float(resp["price"])
                except (TypeError, ValueError):
                    fill_price = None
            if fill_price is None:
                fill_price = last_close
            if current_long > 0:
                if base_long_qty <= 0 or self._long_entry_price is None:
                    self._long_entry_price = fill_price
                else:
                    self._long_entry_price = (
                        self._long_entry_price * base_long_qty + fill_price * add_long
                    ) / (base_long_qty + add_long)
            self._long_qty = current_long
        else:
            self._long_qty = current_long

        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            base_short_qty = current_short
            resp = self.exec.open_short(add_short, last_close)
            record(resp, f"open_short_{add_short}")
            current_short += add_short
            fill_price: float | None = None
            if resp and resp.get("status") == "ok" and resp.get("price") is not None:
                try:
                    fill_price = float(resp["price"])
                except (TypeError, ValueError):
                    fill_price = None
            if fill_price is None:
                fill_price = last_close
            if current_short > 0:
                if base_short_qty <= 0 or self._short_entry_price is None:
                    self._short_entry_price = fill_price
                else:
                    self._short_entry_price = (
                        self._short_entry_price * base_short_qty + fill_price * add_short
                    ) / (base_short_qty + add_short)
            self._short_qty = current_short
        else:
            self._short_qty = current_short

        self.exec.cancel_all_conditional()

        rr = max(0.0, float(getattr(self.cfg, "rr", 0.0)))
        long_tp = None
        short_tp = None

        if current_long > 0:
            if stop_loss is not None and target_signal == 1:
                self._long_stop_loss = stop_loss
            entry_price = self._long_entry_price
            stop_price = self._long_stop_loss
            if rr > 0 and entry_price is not None and stop_price is not None:
                risk = entry_price - stop_price
                if risk > 0:
                    long_tp = entry_price + rr * risk
            self._long_take_profit = long_tp
            self._long_qty = current_long
        else:
            self._clear_long_state()

        if current_short > 0:
            if stop_loss is not None and target_signal == -1:
                self._short_stop_loss = stop_loss
            entry_price = self._short_entry_price
            stop_price = self._short_stop_loss
            if rr > 0 and entry_price is not None and stop_price is not None:
                risk = stop_price - entry_price
                if risk > 0:
                    short_tp = entry_price - rr * risk
            self._short_take_profit = short_tp
            self._short_qty = current_short
        else:
            self._clear_short_state()

        if current_long > 0 and long_tp is not None and last_close >= long_tp:
            resp = self.exec.close_long(current_long, last_close)
            record(resp, f"take_profit_long_{current_long}")
            current_long = 0
            self._clear_long_state()
            self.exec.cancel_all_conditional()
        elif current_short > 0 and short_tp is not None and last_close <= short_tp:
            resp = self.exec.close_short(current_short, last_close)
            record(resp, f"take_profit_short_{current_short}")
            current_short = 0
            self._clear_short_state()
            self.exec.cancel_all_conditional()

        if current_long > 0 and self._long_stop_loss is not None:
            hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None
            self.exec.place_stop("sell", current_long, self._long_stop_loss, hedge_ps)
        elif current_short > 0 and self._short_stop_loss is not None:
            hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None
            self.exec.place_stop("buy", current_short, self._short_stop_loss, hedge_ps)

        stop_loss_value = self._long_stop_loss if self._long_stop_loss is not None else self._short_stop_loss
        if stop_loss_value is None:
            stop_loss_value = stop_loss

        take_profit_value = long_tp if long_tp is not None else short_tp

        action_str = "|".join(actions) if actions else None
        exec_price = prices[-1] if prices else None
        fee = sum(fees) if fees else None
        order_id = "|".join(order_ids) if order_ids else None

        mode_str = f"{'OKX-DEMO-SWAP' if self.cfg.use_demo else 'OKX-SWAP'}"
        self.csv.append(
            {
                "timestamp": datetime.now(),
                "signal": target_signal,
                "close": last_close,
                "position": self.pos_reader.net_sign(),
                "action": action_str,
                "exec_price": exec_price,
                "fee": fee,
                "order_id": order_id,
                "stop_loss": stop_loss_value,
                "take_profit": take_profit_value,
                "best_factor": None,
                "equity": equity,
                "mode": mode_str,
                "atr_norm": atr_norm,
            }
        )

        self.logger.info(
            "MACD: signal=%s regime=%s vol_ok=%s atr_norm=%.5f hist+?=%s hist-?=%s DIF=%.5f DEA=%.5f",
            target_signal,
            regime_state,
            vol_ok,
            atr_norm,
            hist_positive,
            hist_negative,
            dif_curr,
            dea_curr,
        )
