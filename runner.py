# runner.py
import time
from datetime import datetime, timedelta
from math import floor
from config import Config
from data import DataFetcher
from indicators import IndicatorEngine
from signals import SignalBuilder
from selector import FactorSelector
from order_executor import OrderExecutor
from csv_logger import CsvLogger
from position_reader import PositionReader


class StrategyRunner:
    """策略主控：拉取数据、生成信号并驱动执行和日志。"""

    def __init__(self, cfg: Config, exch, logger):
        """组装运行时所需的所有组件。"""
        self.cfg = cfg
        self.logger = logger
        self.fetcher = DataFetcher(cfg, exch, logger)
        self.ind = IndicatorEngine(cfg)
        self.sbuilder = SignalBuilder(cfg)
        self.selector = FactorSelector(cfg)
        self.exec = OrderExecutor(cfg, exch, logger)
        self.csv = CsvLogger(cfg.csv_log_file, logger)
        self.pos_reader = PositionReader(cfg, exch)
        self._daily_anchor_equity = None
        self._daily_anchor_date = None
        self._daily_suspend_until = None
        self._global_kill = False

    def _risk_levels(self, last_close: float, st: dict, signal: int):
        """根据当前信号计算止损/止盈价格。"""
        stop_loss = take_profit = None
        try:
            if self.cfg.mode == 'long_short':
                if signal == 1:
                    sl = float(st['lower'][-1])
                    if sl < last_close:
                        tp = last_close + self.cfg.rr * (last_close - sl)
                        stop_loss, take_profit = sl, tp
                elif signal == -1:
                    su = float(st['upper'][-1])
                    if su > last_close:
                        tp = last_close - self.cfg.rr * (su - last_close)
                        stop_loss, take_profit = su, tp
            else:
                if signal == 1:
                    sl = float(st['lower'][-1])
                    if sl < last_close:
                        tp = last_close + self.cfg.rr * (last_close - sl)
                        stop_loss, take_profit = sl, tp
        except Exception:
            pass
        return stop_loss, take_profit

    def _assess_drawdown(self, equity: float) -> str | None:
        """更新日内与总体回撤状态，返回 kill 状态。"""
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
            self._global_kill = True
            return 'overall'

        if self._daily_suspend_until and now < self._daily_suspend_until:
            return 'daily_active'

        anchor = self._daily_anchor_equity or equity
        daily_floor = anchor * (1.0 - self.cfg.daily_drawdown_limit)
        if equity <= daily_floor:
            self._daily_suspend_until = now + timedelta(hours=24)
            return 'daily_trigger'
        return None

    def _compute_position_size(self, signal: int, last_close: float, stop_loss: float | None, equity: float) -> int:
        """依据风险预算与杠杆限制计算目标张数。"""
        if signal == 0:
            return 0
        if self.cfg.mode == 'long_flat' and signal < 0:
            return 0
        if stop_loss is None or last_close <= 0 or equity <= 0:
            self.logger.warning("缺少有效止损或价格，放弃开仓")
            return 0

        market_info = self.exec.market_info()
        available_margin = self.exec.available_margin()
        contract_value = float(market_info.get('contractSize') or market_info.get('ctVal') or 1.0)
        stop_distance = last_close - stop_loss if signal > 0 else stop_loss - last_close
        if stop_distance <= 0:
            self.logger.warning("止损距离无效，跳过交易")
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
                self.logger.warning("风险预算不足，启用最小张数保护性下单")
                return min_contracts
            return 0
        return contracts

    def _flatten_positions(self, long_amt: int, short_amt: int, last_price: float):
        """强制清空仓位。"""
        if long_amt > 0:
            self.logger.info(f"KillSwitch: 平掉多头 {long_amt} 张")
            self.exec.close_long(long_amt, last_price)
        if short_amt > 0:
            self.logger.info(f"KillSwitch: 平掉空头 {short_amt} 张")
            self.exec.close_short(short_amt, last_price)

    def run_once(self):
        """执行一次完整的拉取-评估-下单流程。"""
        df = self.fetcher.fetch_ohlcv_df()
        if df.empty:
            self.logger.warning("未获取到数据，跳过")
            return

        df = self.fetcher.drop_unclosed_tail(df)
        df_atr = self.ind.compute_atr(df)
        if len(df_atr) < max(200, self.cfg.metric_lookback):
            self.logger.warning("数据不足以评估，跳过")
            return

        best_factor = self.selector.maybe_select(df_atr)
        st = self.ind.compute_supertrend(df_atr, best_factor)
        sig_arr = self.sbuilder.build(df_atr, st)
        current_signal = int(sig_arr[-1])
        last_close = float(df_atr['Close'].iloc[-1])

        self.logger.info(f"信号: {current_signal} 因子: {best_factor:.3f} Close: {last_close:.2f}")

        long_amt, short_amt = self.pos_reader._hedge_amounts()
        equity = self.exec.account_equity()
        drawdown_state = self._assess_drawdown(equity)
        if drawdown_state:
            msg = {
                'overall': "触发总回撤 Kill Switch，立即清仓并停止交易",
                'daily_trigger': "触发日内回撤限制，清仓并暂停 24 小时",
                'daily_active': "日内暂停中，保持空仓",
            }[drawdown_state]
            self.logger.error(msg)
            self._flatten_positions(long_amt, short_amt, last_close)
            return

        stop_loss, take_profit = self._risk_levels(last_close, st, current_signal)
        target_contracts = self._compute_position_size(current_signal, last_close, stop_loss, equity)

        desired_long = desired_short = 0
        if self.cfg.mode == 'long_flat':
            desired_long = target_contracts if current_signal == 1 else 0
        else:
            if current_signal == 1:
                desired_long = target_contracts
            elif current_signal == -1:
                desired_short = target_contracts

        actions = []
        prices = []
        fees = []
        order_ids = []

        def record(resp, label):
            if resp and resp.get('status') == 'ok':
                actions.append(label)
                if resp.get('price') is not None:
                    prices.append(resp['price'])
                if resp.get('fee') is not None:
                    fees.append(resp['fee'])
                if resp.get('order_id'):
                    order_ids.append(resp['order_id'])

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
            hedge_ps = 'long' if self.cfg.position_mode.lower() == 'hedge' else None
            if stop_loss is not None and desired_long > 0:
                self.exec.place_stop('sell', desired_long, stop_loss, hedge_ps)
            if take_profit is not None and desired_long > 0:
                self.exec.place_take_profit('sell', desired_long, take_profit, hedge_ps)
        add_short = max(0, desired_short - current_short)
        if add_short > 0:
            resp = self.exec.open_short(add_short, last_close)
            record(resp, f"open_short_{add_short}")
            hedge_ps = 'short' if self.cfg.position_mode.lower() == 'hedge' else None
            if stop_loss is not None and desired_short > 0:
                self.exec.place_stop('buy', desired_short, stop_loss, hedge_ps)
            if take_profit is not None and desired_short > 0:
                self.exec.place_take_profit('buy', desired_short, take_profit, hedge_ps)

        action_str = '|'.join(actions) if actions else None
        exec_price = prices[-1] if prices else None
        fee = sum(fees) if fees else None
        order_id = '|'.join(order_ids) if order_ids else None

        mode_str = f"{'OKX-DEMO-SWAP' if self.cfg.use_demo else 'OKX-SWAP'}"
        self.csv.append({
            'timestamp': datetime.now(),
            'signal': current_signal,
            'close': last_close,
            'position': self.pos_reader.net_sign(),
            'action': action_str,
            'exec_price': exec_price,
            'fee': fee,
            'order_id': order_id,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'best_factor': best_factor,
            'equity': equity,
            'mode': mode_str
        })

    def align_and_loop(self, align_fn):
        """先执行一次 run_once，再按周期对齐循环运行。"""
        self.logger.info(f"启动主循环 MODE={'OKX模拟盘' if self.cfg.use_demo else 'OKX实盘'}, SYMBOL={self.cfg.symbol}, TF={self.cfg.timeframe}, SELECTION={self.cfg.selection}, STRAT_MODE={self.cfg.mode}")
        try:
            self.run_once()
        except Exception as e:
            self.logger.exception(f"启动 run_once 失败: {e}")
        while True:
            align_fn()
            try:
                if self._global_kill:
                    self.logger.error("全局 Kill Switch 已触发，停止循环")
                    break
                self.run_once()
            except Exception as e:
                self.logger.exception(f"运行异常: {e}")
                time.sleep(5)
