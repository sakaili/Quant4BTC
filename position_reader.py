# position_reader.py
from config import Config
from exchange_client import ExchangeClient


class PositionReader:
    """屏蔽 OKX 多种仓位模式差异，对外输出净仓和明细。"""

    def __init__(self, cfg: Config, exch: ExchangeClient):
        """仅保存配置与交易所引用，便于随时查询持仓。"""
        self.cfg = cfg
        self.exch = exch

    def net_sign(self) -> int:
        """返回当前净仓方向：1 多，-1 空，0 无。"""
        long_c, short_c = self._hedge_amounts()
        if self.cfg.position_mode.lower() == 'hedge':
            net = long_c - short_c
        else:
            # net 模式：OKX 也会返回 side/amount；我们按 long-short 聚合
            net = long_c - short_c
        return 0 if net == 0 else (1 if net > 0 else -1)

    def _hedge_amounts(self) -> tuple[int, int]:
        """提取 Hedge 模式下的 long/short 张数。"""
        long_c = short_c = 0
        for p in self.exch.fetch_positions():
            if p.get('symbol') != self.cfg.symbol:
                continue
            pos_side = (p.get('positionSide') or p.get('posSide') or p.get('side') or '').lower()
            c = float(p.get('contracts', 0) or 0)
            if pos_side == 'long':
                long_c += c
            elif pos_side == 'short':
                short_c += c
            else:
                # 保险起见：若只有净暴露
                side = (p.get('side') or '').lower()
                if side == 'long':
                    long_c += c
                elif side == 'short':
                    short_c += c
        return int(long_c), int(short_c)
