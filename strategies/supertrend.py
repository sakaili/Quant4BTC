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
        self._trade_anchor_equity: float | None = None
        self._position_sign: int = 0
        self._entry_price_long: float | None = None
        self._entry_price_short: float | None = None
        self._last_executed_signal: int | None = None
        self._invert_signal: bool = False


    def _risk_levels(self, last_close: float, st: dict, signal: int) -> float | None:
        """计算止损价格（考虑杠杆）"""
        # 使用scalping配置参数（已考虑杠杆）
        pct = max(0.0, float(getattr(self.cfg, "scalping_stop_loss_pct", self.cfg.stop_loss_pct)))
        if pct <= 0 or last_close <= 0:
            return None
        if self.cfg.mode == "long_short":
            if signal == 1:
                stop_price = last_close * (1.0 - pct / 100.0)
                self.logger.info(
                    "🛑 多头止损: 当前价=%.2f, 止损价=%.2f (%.3f%%)",
                    last_close, stop_price, pct
                )
                return stop_price
            if signal == -1:
                stop_price = last_close * (1.0 + pct / 100.0)
                self.logger.info(
                    "🛑 空头止损: 当前价=%.2f, 止损价=%.2f (%.3f%%)",
                    last_close, stop_price, pct
                )
                return stop_price
        else:
            if signal == 1:
                stop_price = last_close * (1.0 - pct / 100.0)
                self.logger.info(
                    "🛑 多头止损: 当前价=%.2f, 止损价=%.2f (%.3f%%)",
                    last_close, stop_price, pct
                )
                return stop_price
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

        # 支持百分比仓位模式
        if self.cfg.position_sizing_mode == "percentage":
            position_value = equity * self.cfg.position_size_pct
            contracts = position_value / last_close
            self.logger.info(
                "📊 仓位计算(百分比): 净值=%.2f USDC × %.1f%% = %.2f USDC → %.6f BTC",
                equity, self.cfg.position_size_pct * 100, position_value, contracts
            )
            return contracts

        # 固定仓位模式
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

    def run_once(self, equity: float | None = None) -> None:
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
        raw_signal = int(sig_arr[-1])
        trade_signal = raw_signal if not self._invert_signal else -raw_signal
        if self._last_executed_signal is None and trade_signal != 0:
            if self.cfg.allow_initial_position:
                self.logger.info(
                    "Initial signal %s detected, proceeding with initial position (allow_initial_position=True)",
                    trade_signal,
                )
            else:
                self.logger.info(
                    "Warmup phase: observing initial signal %s, awaiting reversal before first trade",
                    trade_signal,
                )
                self._last_executed_signal = trade_signal
                return
        if self._last_executed_signal is not None and trade_signal == self._last_executed_signal:
            self.logger.info("Signal %s unchanged, skip trade this cycle", trade_signal)
            return
        current_signal = trade_signal
        last_close = float(df_atr["Close"].iloc[-1])
        if hasattr(self.exec, "on_price_tick"):
            try:
                self.exec.on_price_tick(last_close)
            except Exception as hook_exc:  # pragma: no cover - defensive logging
                self.logger.warning("on_price_tick hook failed: %s", hook_exc)

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
            "[%s] 信号:%s 因子:%.3f 来源:%s 收盘价:%.4f MACD过滤:%s DIF:%s DEA:%s",
            self.cfg.symbol,
            current_signal,
            factor_display,
            source_desc,
            last_close,
            "启用" if self.cfg.use_macd_filter else "未启用",
            f"{dif_val:.6f}" if dif_val is not None else "NA",
            f"{dea_val:.6f}" if dea_val is not None else "NA",
        )

        long_amt, short_amt = self.pos_reader._hedge_amounts()

        # 使用传入的净值参数（多品种模式下共享快照），或自行读取（单品种模式/向后兼容）
        if equity is None:
            equity = self.exec.account_equity()

        cooldown_loss_amount = max(0.0, float(getattr(self.cfg, "cooldown_loss_amount", 0.0)))
        cooldown_loss_pct = max(0.0, float(getattr(self.cfg, "cooldown_loss_pct", 0.0)))
        net_sign = 1 if long_amt > 0 else -1 if short_amt > 0 else 0

        # 检测交易所止损单触发：之前有仓位，现在没了，且有亏损
        exchange_stop_triggered = False
        if net_sign == 0 and self._position_sign != 0 and self._trade_anchor_equity:
            loss_amount = self._trade_anchor_equity - equity
            if cooldown_loss_amount > 0 and loss_amount >= cooldown_loss_amount:
                exchange_stop_triggered = True
                prev_position_sign = self._position_sign
                self.logger.error(
                    "Exchange stop loss triggered! Loss: %.2f USDT, opening reverse position",
                    loss_amount,
                )
            elif cooldown_loss_amount <= 0 and cooldown_loss_pct > 0:
                loss_ratio = loss_amount / self._trade_anchor_equity
                if loss_ratio >= cooldown_loss_pct:
                    exchange_stop_triggered = True
                    prev_position_sign = self._position_sign
                    self.logger.error(
                        "Exchange stop loss triggered! Loss: %.2f%%, opening reverse position",
                        loss_ratio * 100,
                    )

        # 处理交易所止损单触发
        if exchange_stop_triggered:
            # 反转信号解释方式
            self._invert_signal = not self._invert_signal

            # 重置状态
            self._trade_anchor_equity = None
            self._position_sign = 0
            self._entry_price_long = None
            self._entry_price_short = None

            # 立即开反手仓位
            reverse_signal = -prev_position_sign
            target_size = float(getattr(self.cfg, "fixed_order_size", 0.0))

            if target_size > 0:
                if reverse_signal > 0:
                    resp = self.exec.open_long(target_size, last_close)
                    if resp and resp.get("status") == "ok":
                        fill_price = float(resp.get("price") or last_close)
                        self._entry_price_long = fill_price
                        self.logger.info("Opened LONG reverse position: size=%.4f price=%.4f", target_size, fill_price)
                elif reverse_signal < 0:
                    resp = self.exec.open_short(target_size, last_close)
                    if resp and resp.get("status") == "ok":
                        fill_price = float(resp.get("price") or last_close)
                        self._entry_price_short = fill_price
                        self.logger.info("Opened SHORT reverse position: size=%.4f price=%.4f", target_size, fill_price)

                # 下新的止损单和反向开仓条件单
                self.exec.cancel_all_conditional()
                market_info = self.exec.market_info()
                contract_value = float(market_info.get("contractSize") or market_info.get("ctVal") or 1.0)
                loss_amount_cfg = max(0.0, float(getattr(self.cfg, "cooldown_loss_amount", 0.0)))

                if loss_amount_cfg > 0 and contract_value > 0 and target_size > 0:
                    if reverse_signal > 0 and self._entry_price_long:
                        delta = loss_amount_cfg / (target_size * contract_value)
                        stop_price = max(0.0, self._entry_price_long - delta)
                        hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None

                        # 下止损单（平多仓）
                        self.exec.place_stop("sell", target_size, stop_price, hedge_ps, reduce_only=True)
                        self.logger.info("Placed LONG stop loss at %.4f", stop_price)

                        # 下反向开仓条件单（开空仓）
                        reverse_hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None
                        self.exec.place_stop("sell", target_size, stop_price, reverse_hedge_ps, reduce_only=False)
                        self.logger.info("Placed reverse SHORT open at %.4f", stop_price)

                    elif reverse_signal < 0 and self._entry_price_short:
                        delta = loss_amount_cfg / (target_size * contract_value)
                        stop_price = max(0.0, self._entry_price_short + delta)
                        hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None

                        # 下止损单（平空仓）
                        self.exec.place_stop("buy", target_size, stop_price, hedge_ps, reduce_only=True)
                        self.logger.info("Placed SHORT stop loss at %.4f", stop_price)

                        # 下反向开仓条件单（开多仓）
                        reverse_hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None
                        self.exec.place_stop("buy", target_size, stop_price, reverse_hedge_ps, reduce_only=False)
                        self.logger.info("Placed reverse LONG open at %.4f", stop_price)

            self._last_executed_signal = reverse_signal
            return

        # 正常的仓位追踪和止损检测（原有逻辑）
        if net_sign == 0:
            self._trade_anchor_equity = None
            self._position_sign = 0
        else:
            if self._position_sign != net_sign or self._trade_anchor_equity is None:
                self._trade_anchor_equity = equity if equity > 0 else None
                self._position_sign = net_sign
        if net_sign != 0 and self._trade_anchor_equity and self._trade_anchor_equity > 0:
            loss_amount = self._trade_anchor_equity - equity
            trigger = False
            if cooldown_loss_amount > 0 and loss_amount >= cooldown_loss_amount:
                trigger = True
            elif cooldown_loss_amount <= 0 and cooldown_loss_pct > 0:
                loss_ratio = loss_amount / self._trade_anchor_equity
                trigger = loss_ratio >= cooldown_loss_pct
            if trigger:
                self.logger.error(
                    "Single-trade loss %.2f USDT reached, flattening and opening reverse position",
                    loss_amount,
                )
                self._flatten_positions(long_amt, short_amt, last_close)

                # 反转信号解释方式，后续所有信号都会被反转
                self._invert_signal = not self._invert_signal

                # 重置状态，准备开新仓
                self._trade_anchor_equity = None
                self._position_sign = 0
                self._entry_price_long = None
                self._entry_price_short = None

                # 立即开反手仓位：如果是多头止损，开空仓；如果是空头止损，开多仓
                reverse_signal = -net_sign  # 反转信号：1变-1，-1变1
                target_size = float(getattr(self.cfg, "fixed_order_size", 0.0))

                if target_size > 0:
                    if reverse_signal > 0:
                        # 开多仓
                        resp = self.exec.open_long(target_size, last_close)
                        if resp and resp.get("status") == "ok":
                            fill_price = float(resp.get("price") or last_close)
                            self._entry_price_long = fill_price
                            self.logger.info("Opened LONG position: size=%.4f price=%.4f", target_size, fill_price)
                    elif reverse_signal < 0:
                        # 开空仓
                        resp = self.exec.open_short(target_size, last_close)
                        if resp and resp.get("status") == "ok":
                            fill_price = float(resp.get("price") or last_close)
                            self._entry_price_short = fill_price
                            self.logger.info("Opened SHORT position: size=%.4f price=%.4f", target_size, fill_price)

                    # 下止损单和反向开仓条件单
                    self.exec.cancel_all_conditional()
                    market_info = self.exec.market_info()
                    contract_value = float(market_info.get("contractSize") or market_info.get("ctVal") or 1.0)
                    loss_amount_cfg = max(0.0, float(getattr(self.cfg, "cooldown_loss_amount", 0.0)))

                    if loss_amount_cfg > 0 and contract_value > 0 and target_size > 0:
                        if reverse_signal > 0 and self._entry_price_long:
                            delta = loss_amount_cfg / (target_size * contract_value)
                            stop_price = max(0.0, self._entry_price_long - delta)
                            hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None

                            # 下止损单（平多仓）
                            self.exec.place_stop("sell", target_size, stop_price, hedge_ps, reduce_only=True)
                            self.logger.info("Placed LONG stop loss at %.4f", stop_price)

                            # 下反向开仓条件单（开空仓）
                            reverse_hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None
                            self.exec.place_stop("sell", target_size, stop_price, reverse_hedge_ps, reduce_only=False)
                            self.logger.info("Placed reverse SHORT open at %.4f", stop_price)

                        elif reverse_signal < 0 and self._entry_price_short:
                            delta = loss_amount_cfg / (target_size * contract_value)
                            stop_price = max(0.0, self._entry_price_short + delta)
                            hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None

                            # 下止损单（平空仓）
                            self.exec.place_stop("buy", target_size, stop_price, hedge_ps, reduce_only=True)
                            self.logger.info("Placed SHORT stop loss at %.4f", stop_price)

                            # 下反向开仓条件单（开多仓）
                            reverse_hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None
                            self.exec.place_stop("buy", target_size, stop_price, reverse_hedge_ps, reduce_only=False)
                            self.logger.info("Placed reverse LONG open at %.4f", stop_price)

                self._last_executed_signal = reverse_signal
                return  # 止损反手后直接返回，不继续执行后续逻辑

        anchor_equity = self._trade_anchor_equity or equity
        unrealized_pct = (
            ((equity - anchor_equity) / anchor_equity) * 100.0 if anchor_equity > 0 else 0.0
        )
        self.logger.info(
            "Position long=%.4f short=%.4f net=%d equity=%.2f unrealized=%.2f%%",
            long_amt,
            short_amt,
            net_sign,
            equity,
            unrealized_pct,
        )

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
        long_avg_base = current_long
        if add_long > 0:
            resp = self.exec.open_long(add_long, last_close)
            record(resp, f"open_long_{add_long}")
            if resp and resp.get("status") == "ok":
                fill_price = float(resp.get("price") or last_close)
                prev_amt = current_long
                current_long += add_long
                if prev_amt <= 0 or self._entry_price_long is None:
                    self._entry_price_long = fill_price
                else:
                    self._entry_price_long = (
                        (prev_amt * self._entry_price_long) + (add_long * fill_price)
                    ) / (prev_amt + add_long)
        if current_long == 0:
            self._entry_price_long = None

        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            resp = self.exec.open_short(add_short, last_close)
            record(resp, f"open_short_{add_short}")
            if resp and resp.get("status") == "ok":
                fill_price = float(resp.get("price") or last_close)
                prev_amt = current_short
                current_short += add_short
                if prev_amt <= 0 or self._entry_price_short is None:
                    self._entry_price_short = fill_price
                else:
                    self._entry_price_short = (
                        (prev_amt * self._entry_price_short) + (add_short * fill_price)
                    ) / (prev_amt + add_short)
        if current_short == 0:
            self._entry_price_short = None

        # 取消所有现有的条件单
        self.exec.cancel_all_conditional()

        # 使用百分比止盈止损（考虑杠杆）
        stop_loss_pct = self.cfg.scalping_stop_loss_pct  # 0.1% (已考虑杠杆)
        take_profit_pct = self.cfg.scalping_take_profit_pct  # 0.2% (已考虑杠杆)

        if current_long > 0 and self._entry_price_long:
            # 多头仓位的止损和止盈
            stop_price = self._entry_price_long * (1.0 - stop_loss_pct / 100.0)
            tp_price = self._entry_price_long * (1.0 + take_profit_pct / 100.0)
            hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None

            # 下止损单（平多仓）
            self.exec.place_stop("sell", current_long, stop_price, hedge_ps, reduce_only=True)
            self.logger.info(
                "🛑 多头止损单: 入场价=%.2f, 止损价=%.2f (%.3f%%)",
                self._entry_price_long, stop_price, stop_loss_pct
            )

            # 下止盈单（平多仓）
            if self.cfg.use_take_profit:
                self.exec.place_take_profit("sell", current_long, tp_price, hedge_ps, reduce_only=True)
                self.logger.info(
                    "🎯 多头止盈单: 入场价=%.2f, 止盈价=%.2f (%.3f%%)",
                    self._entry_price_long, tp_price, take_profit_pct
                )

        elif current_short > 0 and self._entry_price_short:
            # 空头仓位的止损和止盈
            stop_price = self._entry_price_short * (1.0 + stop_loss_pct / 100.0)
            tp_price = self._entry_price_short * (1.0 - take_profit_pct / 100.0)
            hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None

            # 下止损单（平空仓）
            self.exec.place_stop("buy", current_short, stop_price, hedge_ps, reduce_only=True)
            self.logger.info(
                "🛑 空头止损单: 入场价=%.2f, 止损价=%.2f (%.3f%%)",
                self._entry_price_short, stop_price, stop_loss_pct
            )

            # 下止盈单（平空仓）
            if self.cfg.use_take_profit:
                self.exec.place_take_profit("buy", current_short, tp_price, hedge_ps, reduce_only=True)
                self.logger.info(
                    "🎯 空头止盈单: 入场价=%.2f, 止盈价=%.2f (%.3f%%)",
                    self._entry_price_short, tp_price, take_profit_pct
                )

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
        self._last_executed_signal = current_signal
