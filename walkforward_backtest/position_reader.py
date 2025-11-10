# position_reader.py
from .config import Config
from .exchange_client import ExchangeClient


class PositionReader:
    """Utility helpers to inspect current Binance futures positions."""

    def __init__(self, cfg: Config, exch: ExchangeClient):
        self.cfg = cfg
        self.exch = exch

    @staticmethod
    def _symbol_aliases(symbol: str | None) -> set[str]:
        if not symbol:
            return set()
        raw = str(symbol).strip().upper()
        aliases = {raw, raw.replace("/", "")}
        for delim in (":", "-", "_"):
            if delim in raw:
                left, _, right = raw.partition(delim)
                aliases.update({left, right})
                aliases.update({left.replace("/", ""), right.replace("/", "")})
        return {alias for alias in aliases if alias}

    def net_sign(self) -> int:
        long_amt, short_amt = self._hedge_amounts()
        net = long_amt - short_amt
        if abs(net) < 1e-8:
            return 0
        return 1 if net > 0 else -1

    def _hedge_amounts(self) -> tuple[float, float]:
        """
        Robustly fetch hedge mode position amounts.
        Checks unified ccxt keys and raw exchange payloads to ensure accuracy.
        """
        long_amt = 0.0
        short_amt = 0.0

        def to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        try:
            positions = self.exch.fetch_positions()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger = getattr(self.exch, "logger", None)
            if logger:
                logger.error(f"Failed to fetch positions: {exc}", exc_info=True)
            return 0.0, 0.0

        if not positions:
            return 0.0, 0.0

        target_aliases = self._symbol_aliases(self.cfg.symbol)

        for position in positions:
            info = position.get("info", {}) or {}
            symbol_candidates = [
                position.get("symbol"),
                position.get("instrument"),
                position.get("id"),
                info.get("symbol"),
                info.get("pair"),
            ]
            aliases = set()
            for candidate in symbol_candidates:
                aliases |= self._symbol_aliases(candidate)
            if not aliases & target_aliases:
                continue

            contracts_raw = position.get("contracts")
            contracts_val = to_float(contracts_raw)
            if contracts_val is None or abs(contracts_val) < 1e-12:
                contracts_raw = position.get("amount")
                contracts_val = to_float(contracts_raw)

            if contracts_val is None or abs(contracts_val) < 1e-12:
                for key in ("positionAmt", "posAmt", "position_amt", "open_position"):
                    raw = info.get(key)
                    val = to_float(raw)
                    if val is not None and abs(val) >= 1e-12:
                        contracts_val = val
                        break

            if contracts_val is None or abs(contracts_val) < 1e-9:
                continue

            pos_side = (
                position.get("positionSide")
                or position.get("posSide")
                or position.get("side")
                or ""
            ).lower()

            if pos_side not in {"long", "short"}:
                if contracts_val > 0:
                    pos_side = "long"
                elif contracts_val < 0:
                    pos_side = "short"
                else:
                    continue

            contracts = abs(contracts_val)
            if pos_side == "long":
                long_amt += contracts
            elif pos_side == "short":
                short_amt += contracts

        return float(long_amt), float(short_amt)
