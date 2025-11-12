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


class UltimateScalpingStrategy(Strategy):
    """
    Ultimate Scalping Strategy - Rebuilt from TradingView

    结合 EMA 趋势、RSI 动量和 SuperTrend 的多头空头剥头皮策略。

    信号逻辑:
    - 多头: 快速EMA > 慢速EMA, RSI > 55, 价格上穿快速EMA
    - 空头: 快速EMA < 慢速EMA, RSI < 45, 价格下穿快速EMA
    - 重入: 趋势中回调后重新进入
    - 反向平仓: 出现反向信号时自动平仓
    """

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
        self.selector = factor_selector  # 使用 KMeans 进行趋势识别
        self._last_executed_signal: int | None = None
        self._entry_price_long: float | None = None
        self._entry_price_short: float | None = None

    def run_once(self, equity: float | None = None) -> None:
        df = self.fetcher.fetch_ohlcv_df()
        if df.empty:
            self.logger.warning("未获取到数据,跳过")
            return

        df = self.fetcher.drop_unclosed_tail(df)
        df_atr = self.ind.compute_atr(df)

        # 需要足够的数据来计算所有指标
        min_bars = max(self.cfg.ema_slow_length, self.cfg.rsi_length,
                       self.cfg.metric_lookback, 200)
        if len(df_atr) < min_bars:
            self.logger.warning(f"数据不足以计算指标 (需要至少{min_bars}根K线)")
            return

        # 使用 KMeans 选择最佳 SuperTrend 因子
        best_factor = self.selector.maybe_select(df_atr)
        st = self.ind.compute_supertrend(df_atr, best_factor)

        # 计算EMA
        ema_fast = self.ind.compute_ema(df_atr['Close'], self.cfg.ema_fast_length)
        ema_slow = self.ind.compute_ema(df_atr['Close'], self.cfg.ema_slow_length)

        # 计算RSI
        rsi = self.ind.compute_rsi(df_atr['Close'], self.cfg.rsi_length)

        # 获取最新值
        last_close = float(df_atr['Close'].iloc[-1])
        last_ema_fast = float(ema_fast.iloc[-1])
        last_ema_slow = float(ema_slow.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        last_st_direction = int(st['trend'][-1])  # 1=上升趋势, 0=下降趋势

        # 需要前一根K线数据用于判断穿越
        if len(df_atr) < 2:
            return

        prev_close = float(df_atr['Close'].iloc[-2])
        prev_ema_fast = float(ema_fast.iloc[-2])

        # ================== 交易信号逻辑 ==================

        # 趋势识别
        trend_up = last_ema_fast > last_ema_slow and last_st_direction == 1
        trend_down = last_ema_fast < last_ema_slow and last_st_direction == 0

        # 动量判断
        rsi_bull = last_rsi > 55
        rsi_bear = last_rsi < 45

        # 主信号: 价格穿越快速EMA
        crossover_up = prev_close <= prev_ema_fast and last_close > last_ema_fast
        crossunder_down = prev_close >= prev_ema_fast and last_close < last_ema_fast

        long_condition = trend_up and rsi_bull and crossover_up
        short_condition = trend_down and rsi_bear and crossunder_down

        # 辅助信号: 回调重入
        long_reentry = (
            trend_up
            and last_close > last_ema_fast
            and 50 < last_rsi < 70
        )
        short_reentry = (
            trend_down
            and last_close < last_ema_fast
            and 30 < last_rsi < 50
        )

        # 最终信号合并
        final_long = long_condition or long_reentry
        final_short = short_condition or short_reentry

        # 确定当前信号
        if final_long and not final_short:
            current_signal = 1
        elif final_short and not final_long:
            current_signal = -1
        else:
            current_signal = 0

        # 获取因子选择信息 (在日志输出前获取)
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

        # ================== 详细日志输出 (每次都显示) ==================
        self.logger.info(
            "[%s] ========== 指标详情 ==========",
            self.cfg.symbol
        )
        self.logger.info("💰 价格: %.2f", last_close)
        self.logger.info("📊 EMA: Fast=%.2f Slow=%.2f → %s",
                        last_ema_fast, last_ema_slow,
                        "上升" if last_ema_fast > last_ema_slow else "下降")
        self.logger.info("📈 RSI: %.2f → %s",
                        last_rsi,
                        "强势" if last_rsi > 55 else ("弱势" if last_rsi < 45 else "中性"))
        self.logger.info("🔀 SuperTrend: %s (Factor=%.3f, %s)",
                        "上升" if last_st_direction == 1 else "下降",
                        factor_display, source_desc)
        self.logger.info("📍 综合趋势: %s",
                        "多头" if trend_up else ("空头" if trend_down else "震荡"))
        self.logger.info("🎯 主信号: %s | 重入: %s",
                        ("做多" if long_condition else ("做空" if short_condition else "无")),
                        ("做多" if long_reentry else ("做空" if short_reentry else "无")))
        self.logger.info("⚡ 最终信号: %s",
                        "🟢 做多" if current_signal == 1 else ("🔴 做空" if current_signal == -1 else "⚪ 空仓"))
        self.logger.info("===========================================")

        # 信号去重: 如果信号未变化则跳过交易
        if self._last_executed_signal is not None and current_signal == self._last_executed_signal:
            self.logger.info("⏭ 信号未变化 (signal=%d), 跳过交易", current_signal)
            return

        # 初始仓位处理
        if self._last_executed_signal is None and current_signal != 0:
            if not self.cfg.allow_initial_position:
                self.logger.info(
                    "🔄 预热阶段: 观察到初始信号 %s, 等待反转后才进行首次交易",
                    current_signal
                )
                self._last_executed_signal = current_signal
                return

        # ================== 仓位管理 ==================

        long_amt, short_amt = self.pos_reader._hedge_amounts()

        # 使用传入的净值参数（多品种模式下共享快照），或自行读取（单品种模式/向后兼容）
        if equity is None:
            equity = self.exec.account_equity()
            self.logger.debug("未传入净值参数，自行读取账户净值")
        else:
            self.logger.debug("使用周期开始时的净值快照: %.2f USDC", equity)

        # 风控检查
        drawdown_state = self._assess_drawdown(equity)
        if drawdown_state:
            msg = {
                "overall": "触发总回撤 Kill Switch, 强制清空并停止",
                "daily_trigger": "触发当日回撤上限, 暂停 24 小时",
                "daily_active": "暂停冷却中, 保持空仓",
            }[drawdown_state]
            self.logger.error(msg)
            self._flatten_positions(long_amt, short_amt, last_close)
            return

        # 计算止盈止损
        take_profit_pct = self.cfg.scalping_take_profit_pct
        stop_loss_pct = self.cfg.scalping_stop_loss_pct

        # 计算目标仓位 - 支持固定和百分比模式
        if self.cfg.position_sizing_mode == "percentage":
            # 百分比模式: 使用账户净值的固定百分比
            position_value = equity * self.cfg.position_size_pct
            target_contracts = position_value / last_close
            self.logger.info(
                "📊 仓位计算: 净值=%.2f USDC × %.1f%% = %.2f USDC → %.6f BTC",
                equity, self.cfg.position_size_pct * 100, position_value, target_contracts
            )
        else:
            # 固定模式: 使用固定数量
            target_contracts = float(self.cfg.fixed_order_size)
            self.logger.info(
                "📊 仓位计算: 固定模式 %.6f BTC (价值 %.2f USDC)",
                target_contracts, target_contracts * last_close
            )

        desired_long = desired_short = 0

        # 多头信号: 只在无空仓时开多仓
        if current_signal == 1 and short_amt == 0:
            desired_long = target_contracts
        # 空头信号: 只在无多仓时开空仓
        elif current_signal == -1 and long_amt == 0:
            desired_short = target_contracts

        # 反向平仓逻辑 (按Pine Script: reversal_exit)
        if self.cfg.scalping_reversal_exit:
            if current_signal == 1 and short_amt > 0:
                self.logger.info("检测到多头信号, 平掉空头仓位")
                self.exec.close_short(short_amt, last_close)
                short_amt = 0
            elif current_signal == -1 and long_amt > 0:
                self.logger.info("检测到空头信号, 平掉多头仓位")
                self.exec.close_long(long_amt, last_close)
                long_amt = 0

        # ================== 执行交易 ==================

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

        # 平仓逻辑
        reduce_long = max(0, long_amt - desired_long)
        if reduce_long > 0:
            resp = self.exec.close_long(reduce_long, last_close)
            record(resp, f"close_long_{reduce_long}")

        reduce_short = max(0, short_amt - desired_short)
        if reduce_short > 0:
            resp = self.exec.close_short(reduce_short, last_close)
            record(resp, f"close_short_{reduce_short}")

        # 更新当前仓位
        current_long = max(0, long_amt - reduce_long)
        current_short = max(0, short_amt - reduce_short)

        # 开仓逻辑
        add_long = max(0, desired_long - current_long)
        if add_long > 0:
            resp = self.exec.open_long(add_long, last_close)
            record(resp, f"open_long_{add_long}")
            if resp and resp.get("status") == "ok":
                fill_price = float(resp.get("price") or last_close)
                self._entry_price_long = fill_price
                current_long += add_long

        if current_long == 0:
            self._entry_price_long = None

        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            resp = self.exec.open_short(add_short, last_close)
            record(resp, f"open_short_{add_short}")
            if resp and resp.get("status") == "ok":
                fill_price = float(resp.get("price") or last_close)
                self._entry_price_short = fill_price
                current_short += add_short

        if current_short == 0:
            self._entry_price_short = None

        # ================== 设置止盈止损 ==================

        self.logger.info("[止盈止损] 取消所有挂单...")
        self.exec.cancel_all_conditional()

        # 初始化止盈止损价格变量（用于CSV记录）
        sl_price = None
        tp_price = None

        if current_long > 0 and self._entry_price_long:
            tp_price = self._entry_price_long * (1.0 + take_profit_pct / 100.0)
            sl_price = self._entry_price_long * (1.0 - stop_loss_pct / 100.0)

            hedge_ps = "long" if self.cfg.position_mode.lower() == "hedge" else None

            self.logger.info(
                "[止盈止损] 多头仓位: 入场价=%.2f 数量=%.4f",
                self._entry_price_long, current_long
            )
            self.logger.info(
                "[止盈止损] 止损价=%.2f (%.2f%%) | 止盈价=%.2f (%.2f%%)",
                sl_price, stop_loss_pct, tp_price, take_profit_pct
            )

            # 止损单
            sl_resp = self.exec.place_stop(
                "sell", current_long, sl_price, hedge_ps, reduce_only=True
            )
            self.logger.info("[止盈止损] 止损单状态: %s", sl_resp.get("status"))

            # 止盈单
            tp_resp = self.exec.place_take_profit(
                "sell", current_long, tp_price, hedge_ps, reduce_only=True
            )
            self.logger.info("[止盈止损] 止盈单状态: %s", tp_resp.get("status"))

        if current_short > 0 and self._entry_price_short:
            tp_price = self._entry_price_short * (1.0 - take_profit_pct / 100.0)
            sl_price = self._entry_price_short * (1.0 + stop_loss_pct / 100.0)

            hedge_ps = "short" if self.cfg.position_mode.lower() == "hedge" else None

            self.logger.info(
                "[止盈止损] 空头仓位: 入场价=%.2f 数量=%.4f",
                self._entry_price_short, current_short
            )
            self.logger.info(
                "[止盈止损] 止损价=%.2f (%.2f%%) | 止盈价=%.2f (%.2f%%)",
                sl_price, stop_loss_pct, tp_price, take_profit_pct
            )

            # 止损单
            sl_resp = self.exec.place_stop(
                "buy", current_short, sl_price, hedge_ps, reduce_only=True
            )
            self.logger.info("[止盈止损] 止损单状态: %s", sl_resp.get("status"))

            # 止盈单
            tp_resp = self.exec.place_take_profit(
                "buy", current_short, tp_price, hedge_ps, reduce_only=True
            )
            self.logger.info("[止盈止损] 止盈单状态: %s", tp_resp.get("status"))

        # ================== 记录日志 ==================

        action_str = "|".join(actions) if actions else None
        exec_price = prices[-1] if prices else None
        fee = sum(fees) if fees else None
        order_id = "|".join(order_ids) if order_ids else None

        mode_str = "BINANCE-USDM-TEST" if self.cfg.use_demo else "BINANCE-USDM"
        self.csv.append(
            {
                "timestamp": datetime.now(),
                "symbol": self.cfg.symbol,
                "signal": current_signal,
                "close": last_close,
                "position": self.pos_reader.net_sign(),
                "action": action_str,
                "exec_price": exec_price,
                "fee": fee,
                "order_id": order_id,
                "stop_loss": sl_price if current_long > 0 or current_short > 0 else None,
                "take_profit": tp_price if current_long > 0 or current_short > 0 else None,
                "best_factor": self.cfg.supertrend_mult,
                "equity": equity,
                "mode": mode_str,
            }
        )

        self._last_executed_signal = current_signal
