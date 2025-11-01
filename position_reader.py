# position_reader.py
from config import Config
from exchange_client import ExchangeClient


class PositionReader:
    """Utility helpers to inspect current Binance futures positions."""

    def __init__(self, cfg: Config, exch: ExchangeClient):
        self.cfg = cfg
        self.exch = exch

    def net_sign(self) -> int:
        long_amt, short_amt = self._hedge_amounts()
        net = long_amt - short_amt
        if abs(net) < 1e-8:
            return 0
        return 1 if net > 0 else -1

    def _hedge_amounts(self) -> tuple[float, float]:
        long_amt = 0.0
        short_amt = 0.0
        for position in self.exch.fetch_positions():
            if position.get("symbol") != self.cfg.symbol:
                continue
            pos_side = (
                position.get("positionSide")
                or position.get("posSide")
                or position.get("side")
                or ""
            ).lower()
            contracts = float(position.get("contracts", 0) or 0)
            if pos_side == "long":
                long_amt += contracts
            elif pos_side == "short":
                short_amt += contracts
            else:
                side = (position.get("side") or "").lower()
                if side == "long":
                    long_amt += contracts
                elif side == "short":
                    short_amt += contracts
        return float(long_amt), float(short_amt)
