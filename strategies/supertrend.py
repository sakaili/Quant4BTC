from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

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


class SuperTrendStrategy(Strategy):
    """Concrete strategy implementing the existing SuperTrend logic."""

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

    def _risk_levels(self, last_close: float, st: dict, signal: int) -> float | None:
        pct = max(0.0, float(self.cfg.stop_loss_pct))
        if pct <= 0 or last_close <= 0:
            return None
        if self.cfg.mode == "long_short":
            if signal == 1:
                return last_close * (1.0 - pct)
            if signal == -1:
                return last_close * (1.0 + pct)
        else:
            if signal == 1:
                return last_close * (1.0 - pct)
        return None

    def _compute_position_size(
        self,
        signal: int,
        last_close: float,
        stop_loss: float | None,
        equity: float,
    ) -> float:
        if signal == 0:
            return 0.0
        if self.cfg.mode == "long_flat" and signal < 0:
            return 0.0

        fixed_size = float(getattr(self.cfg, "fixed_order_size", 0.0))
        if fixed_size > 0.0:
            return fixed_size

        if stop_loss is None or last_close <= 0 or equity <= 0:
            self.logger.warning("缺乏有效止损价格，放弃交易")
            return 0.0

        market_info = self.exec.market_info()
        available_margin = self.exec.available_margin()
        contract_value = float(market_info.get("contractSize") or market_info.get("ctVal") or 1.0)
        stop_distance = last_close - stop_loss if signal > 0 else stop_loss - last_close
        if stop_distance <= 0:
            self.logger.warning("止损距离无效，放弃交易")
            return 0.0

        risk_amount = equity * self.cfg.risk_per_trade
        loss_per_contract = stop_distance * contract_value
        if loss_per_contract <= 0:
            return 0.0
        contracts_by_risk = risk_amount / loss_per_contract

        leverage = max(float(self.cfg.leverage), 1.0)
        max_notional = equity * leverage
        per_contract_notional = max(last_close * contract_value, 1e-6)
        contracts_by_leverage = max_notional / per_contract_notional

        contracts_by_margin = available_margin / per_contract_notional if per_contract_notional > 0 else 0.0

        contracts = max(0.0, min(contracts_by_risk, contracts_by_leverage, contracts_by_margin))

        min_contracts = float(max(self.exec.exch.min_contracts(), 0.0))
        if contracts < min_contracts:
            if min_contracts > 0.0:
                self.logger.warning("预估仓位不足，改用最小合约数量下单")
                return min_contracts
            return 0.0
        return contracts

    def run_once(self) -> None:
        df = self.fetcher.fetch_ohlcv_df()
        if df.empty:
            self.logger.warning("未获取到数据，跳过")
            return

        df = self.fetcher.drop_unclosed_tail(df)
        df_atr = self.ind.compute_atr(df)
        if len(df_atr) < max(200, self.cfg.metric_lookback):
            self.logger.warning("数据不足以计算指标")
            return

        best_factor = self.selector.maybe_select(df_atr)
        st = self.ind.compute_supertrend(df_atr, best_factor)
        sig_arr = self.sbuilder.build(df_atr, st)
        current_signal = int(sig_arr[-1])
        last_close = float(df_atr["Close"].iloc[-1])

        if self.cfg.use_macd_filter:
            macd_df = self.ind.compute_macd(df_atr)
            macd_df = macd_df.dropna(subset=["DIF", "DEA"])
            if macd_df.empty:
                macd_allowed = False
                dif_val = dea_val = None
            else:
                dif_val = float(macd_df["DIF"].iloc[-1])
                dea_val = float(macd_df["DEA"].iloc[-1])
                if current_signal > 0:
                    macd_allowed = dif_val >= dea_val
                elif current_signal < 0:
                    macd_allowed = dif_val <= dea_val
                else:
                    macd_allowed = True
            if not macd_allowed:
                current_signal = 0
        else:
            macd_allowed = True
            dif_val = dea_val = None

        selection_info = {}
        if hasattr(self.selector, "last_selection_info"):
            selection_info = self.selector.last_selection_info() or {}
        factor_source = selection_info.get("method", "unknown")
        fallback_reason = selection_info.get("reason")
        if selection_info.get("fallback"):
            source_desc = f"{factor_source}|fallback"
            if fallback_reason:
                source_desc += f":{fallback_reason}"
        elif factor_source in {"cluster_kmeans", "regime_kmeans"}:
            source_desc = "kmeans"
        else:
            source_desc = factor_source
        if selection_info.get("reuse"):
            source_desc = f"{source_desc}|reuse"
        factor_display = float(selection_info.get("factor") or best_factor)

        self.logger.info(
            "信号:%s 因子:%.3f 来源:%s Close: %.2f MACD过滤:%s DIF:%s DEA:%s",
            current_signal,
            factor_display,
            source_desc,
            last_close,
            "启用" if self.cfg.use_macd_filter else "未启用",
            f"{dif_val:.6f}" if dif_val is not None else "NA",
            f"{dea_val:.6f}" if dea_val is not None else "NA",
        )

        long_amt, short_amt = self.pos_reader._hedge_amounts()
        equity = self.exec.account_equity()
        drawdown_state = self._assess_drawdown(equity)
        if drawdown_state:
            msg = {
                "overall": "触发总回撤 Kill Switch，强制清空并停止",
                "daily_trigger": "触发当日回撤上限，暂停 24 小时",
                "daily_active": "暂停冷却中，保持空仓",
            }[drawdown_state]
            self.logger.error(msg)
            self._flatten_positions(long_amt, short_amt, last_close)
            return

        stop_loss = self._risk_levels(last_close, st, current_signal)
        target_contracts = self._compute_position_size(current_signal, last_close, stop_loss, equity)

        desired_long = desired_short = 0
        if self.cfg.mode == "long_flat":
            desired_long = target_contracts if current_signal == 1 else 0
        else:
            if current_signal == 1:
                desired_long = target_contracts
            elif current_signal == -1:
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

        add_long = max(0, desired_long - current_long)
        if add_long > 0:
            resp = self.exec.open_long(add_long, last_close)
            record(resp, f"open_long_{add_long}")
            if resp and resp.get("status") == "ok":
                current_long += add_long
        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            resp = self.exec.open_short(add_short, last_close)
            record(resp, f"open_short_{add_short}")
            if resp and resp.get("status") == "ok":
                current_short += add_short

        self.exec.cancel_all_conditional()
        if current_long > 0 and stop_loss is not None:
            hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None
            self.exec.place_stop("sell", current_long, stop_loss, hedge_ps)
        elif current_short > 0 and stop_loss is not None:
            hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None
            self.exec.place_stop("buy", current_short, stop_loss, hedge_ps)

        action_str = "|".join(actions) if actions else None
        exec_price = prices[-1] if prices else None
        fee = sum(fees) if fees else None
        order_id = "|".join(order_ids) if order_ids else None

        mode_str = "BINANCE-USDM-TEST" if self.cfg.use_demo else "BINANCE-USDM"
        self.csv.append(
            {
                "timestamp": datetime.now(),
                "signal": current_signal,
                "close": last_close,
                "position": self.pos_reader.net_sign(),
                "action": action_str,
                "exec_price": exec_price,
                "fee": fee,
                "order_id": order_id,
                "stop_loss": stop_loss,
                "take_profit": None,
                "best_factor": best_factor,
                "equity": equity,
                "mode": mode_str,
            }
        )
