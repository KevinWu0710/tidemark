from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.core.types import Side, TradeAction
from src.strategy.tmf_session import TmfSessionStrategy


@dataclass
class BacktestResult:
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    final_equity: float = 0.0
    n_steps: int = 0


class SecondBarBacktester:
    """
    Walk-forward second-by-second engine.

    Critical rule: at index i, strategy only sees rows[: i+1] (inclusive current),
    never rows after i. No vectorized peeking of future closes.
    """

    def __init__(
        self,
        strategy: TmfSessionStrategy,
        *,
        capital: float = 10000,
        point_value: float = 10,
        fee_per_side: float = 20,
        tax_rate: float = 0.00002,
        slip_ticks: int = 1,
    ):
        self.strategy = strategy
        self.capital = float(capital)
        self.point_value = float(point_value)
        self.fee_per_side = float(fee_per_side)
        self.tax_rate = float(tax_rate)
        self.slip_ticks = int(slip_ticks)

    def run(self, bars: pd.DataFrame, price_col: str = "close") -> BacktestResult:
        if bars is None or bars.empty:
            logger.warning("backtest received empty bars")
            return BacktestResult(final_equity=self.capital)

        df = bars.reset_index(drop=True)
        if price_col not in df.columns:
            raise ValueError(f"missing price column: {price_col}")

        equity = self.capital
        entry_price: Optional[float] = None
        result = BacktestResult()

        for i in range(len(df)):
            # ONLY past + current — never df.iloc[i+1:]
            history = df.iloc[: i + 1]
            row = df.iloc[i]
            now_ts = row["ts"] if "ts" in df.columns else i
            px = float(row[price_col])

            decision = self.strategy.on_bar(history, now_ts=now_ts)

            if decision.action in (TradeAction.BUY, TradeAction.SELL) and self.strategy.lots == 0:
                fill = self._fill_price(px, decision.action)
                cost = self._round_trip_cost_open(fill)
                equity -= cost
                entry_price = fill
                self.strategy.apply_fill(decision.action, lots=1, price=fill)
                result.trades.append(
                    {
                        "ts": now_ts,
                        "action": decision.action.value,
                        "price": fill,
                        "reason": decision.reason,
                    }
                )
            elif decision.action == TradeAction.CLOSE and self.strategy.lots > 0 and entry_price is not None:
                fill = self._fill_price(px, TradeAction.CLOSE, side=self.strategy.position)
                pnl = self._pnl(entry_price, fill, self.strategy.position)
                cost = self._close_cost(fill)
                net = pnl - cost
                equity += net
                result.trades.append(
                    {
                        "ts": now_ts,
                        "action": "CLOSE",
                        "price": fill,
                        "pnl": net,
                        "reason": decision.reason,
                    }
                )
                self.strategy.apply_fill(TradeAction.CLOSE, pnl=net)
                entry_price = None

            # mark-to-market
            mtm = equity
            if self.strategy.lots > 0 and entry_price is not None:
                mtm = equity + self._pnl(entry_price, px, self.strategy.position)

            result.equity_curve.append({"ts": now_ts, "equity": mtm, "price": px})
            result.n_steps += 1

        result.final_equity = (
            result.equity_curve[-1]["equity"] if result.equity_curve else equity
        )
        return result

    def _fill_price(self, mid: float, action: TradeAction, side: Side | None = None) -> float:
        slip = float(self.slip_ticks)
        if action == TradeAction.BUY:
            return mid + slip
        if action == TradeAction.SELL:
            return mid - slip
        # CLOSE: exit against position
        if side == Side.LONG:
            return mid - slip
        if side == Side.SHORT:
            return mid + slip
        return mid

    def _pnl(self, entry: float, exit_: float, side: Side) -> float:
        pts = (exit_ - entry) if side == Side.LONG else (entry - exit_)
        return pts * self.point_value

    def _round_trip_cost_open(self, price: float) -> float:
        notional = price * self.point_value
        return self.fee_per_side + notional * self.tax_rate

    def _close_cost(self, price: float) -> float:
        notional = price * self.point_value
        return self.fee_per_side + notional * self.tax_rate
