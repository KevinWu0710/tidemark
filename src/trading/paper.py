from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.core.config import notify_prefix
from src.core.types import Side, TradeAction


@dataclass
class PaperBroker:
    """
    紙上帳本：與回測同一套點值／手續費／滑價假設。
    負責成交與權益；不停損規則（停損在 strategy）。
    """

    capital: float = 10000
    max_lots: int = 1
    point_value: float = 10
    fee_per_side: float = 20
    tax_rate: float = 0.00002
    slip_ticks: float = 1
    instrument: str = "TMF"
    session_id: str = ""

    position: Side = Side.FLAT
    lots: int = 0
    entry_price: Optional[float] = None
    cash: float = 0.0  # settled equity (realized); init in __post_init__
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    fills: list[dict[str, Any]] = field(default_factory=list)
    round_trips: list[dict[str, Any]] = field(default_factory=list)
    equity_marks: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.capital = float(self.capital)
        self.max_lots = int(self.max_lots)
        self.point_value = float(self.point_value)
        self.fee_per_side = float(self.fee_per_side)
        self.tax_rate = float(self.tax_rate)
        self.slip_ticks = float(self.slip_ticks)
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cash == 0.0 and self.realized_pnl == 0.0 and not self.fills:
            self.cash = self.capital

    @classmethod
    def from_settings(cls, settings: dict[str, Any], **kwargs: Any) -> "PaperBroker":
        t = settings.get("trading", {})
        bt = settings.get("backtest", {})
        return cls(
            capital=float(t.get("capital", 10000)),
            max_lots=int(t.get("max_lots", 1)),
            point_value=float(t.get("point_value", 10)),
            fee_per_side=float(t.get("fee_per_side", 20)),
            tax_rate=float(t.get("tax_rate", 0.00002)),
            slip_ticks=float(bt.get("slip_ticks", 1)),
            instrument=str(t.get("instrument", "TMF")),
            **kwargs,
        )

    # --- pricing (match SecondBarBacktester) ---

    def _fill_price(self, mid: float, action: TradeAction, side: Side | None = None) -> float:
        slip = self.slip_ticks
        if action == TradeAction.BUY:
            return mid + slip
        if action == TradeAction.SELL:
            return mid - slip
        if side == Side.LONG:
            return mid - slip
        if side == Side.SHORT:
            return mid + slip
        return mid

    def _cost(self, price: float) -> float:
        return self.fee_per_side + price * self.point_value * self.tax_rate

    def _pnl_points(self, entry: float, exit_: float, side: Side) -> float:
        pts = (exit_ - entry) if side == Side.LONG else (entry - exit_)
        return pts * self.point_value

    def unrealized(self, mid: float) -> float:
        if self.lots <= 0 or self.entry_price is None or self.position == Side.FLAT:
            return 0.0
        return self._pnl_points(self.entry_price, float(mid), self.position)

    def equity(self, mid: float | None = None) -> float:
        u = self.unrealized(mid) if mid is not None else 0.0
        return self.cash + u

    def place(
        self,
        action: TradeAction,
        price: float,
        reason: str = "",
        *,
        ts: Any = None,
    ) -> bool:
        """
        price = mid（會套 slip）。成功寫入 fills；失敗寫 rejects。
        """
        mid = float(price)
        ts_s = str(ts) if ts is not None else None

        if action == TradeAction.HOLD:
            return False

        if action in (TradeAction.BUY, TradeAction.SELL):
            if self.lots >= self.max_lots or self.position != Side.FLAT:
                self._reject(action, mid, reason, ts_s, "max_lots_or_already_in")
                return False
            fill = self._fill_price(mid, action)
            cost = self._cost(fill)
            self.cash -= cost
            self.fees_paid += cost
            self.position = Side.LONG if action == TradeAction.BUY else Side.SHORT
            self.lots = 1
            self.entry_price = fill
            rec = {
                "ts": ts_s,
                "action": action.value,
                "side": self.position.value,
                "mid": mid,
                "fill": fill,
                "lots": 1,
                "cost": round(cost, 4),
                "realized_pnl": 0.0,
                "cash": round(self.cash, 2),
                "equity": round(self.equity(mid), 2),
                "reason": reason,
            }
            self.fills.append(rec)
            logger.info(
                f"{notify_prefix()} paper {action.value} @ {fill:.2f} | {reason}"
            )
            return True

        if action == TradeAction.CLOSE:
            if self.lots <= 0 or self.entry_price is None or self.position == Side.FLAT:
                self._reject(action, mid, reason, ts_s, "flat")
                return False
            side = self.position
            fill = self._fill_price(mid, TradeAction.CLOSE, side=side)
            gross = self._pnl_points(self.entry_price, fill, side)
            cost = self._cost(fill)
            net = gross - cost
            self.cash += net
            self.fees_paid += cost
            self.realized_pnl += net
            trip = {
                "ts_exit": ts_s,
                "side": side.value,
                "entry": self.entry_price,
                "exit": fill,
                "gross_pnl": round(gross, 2),
                "cost_close": round(cost, 4),
                "net_pnl": round(net, 2),
                "reason": reason,
            }
            self.round_trips.append(trip)
            rec = {
                "ts": ts_s,
                "action": "CLOSE",
                "side": side.value,
                "mid": mid,
                "fill": fill,
                "lots": 1,
                "cost": round(cost, 4),
                "realized_pnl": round(net, 2),
                "cash": round(self.cash, 2),
                "equity": round(self.cash, 2),
                "reason": reason,
            }
            self.fills.append(rec)
            logger.info(
                f"{notify_prefix()} paper CLOSE {side.value} @ {fill:.2f} "
                f"net={net:.2f} | {reason}"
            )
            self.position = Side.FLAT
            self.lots = 0
            self.entry_price = None
            return True

        return False

    def mark(self, mid: float, *, ts: Any = None) -> float:
        """記錄標記權益（覆盤用，可稀疏寫）。"""
        eq = self.equity(mid)
        self.equity_marks.append(
            {
                "ts": str(ts) if ts is not None else None,
                "mid": float(mid),
                "equity": round(eq, 2),
                "unrealized": round(self.unrealized(mid), 2),
                "position": self.position.value,
            }
        )
        return eq

    def _reject(
        self, action: TradeAction, mid: float, reason: str, ts: str | None, why: str
    ) -> None:
        self.rejects.append(
            {
                "ts": ts,
                "action": action.value,
                "mid": mid,
                "reason": reason,
                "why": why,
            }
        )
        logger.warning(
            f"{notify_prefix()} reject {action.value}: {why} | {reason}"
        )

    def summary(self, mid: float | None = None) -> dict[str, Any]:
        wins = [t for t in self.round_trips if t["net_pnl"] > 0]
        losses = [t for t in self.round_trips if t["net_pnl"] <= 0]
        eq = self.equity(mid) if mid is not None else self.cash
        return {
            "session_id": self.session_id,
            "instrument": self.instrument,
            "capital": self.capital,
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "net_pnl": round(eq - self.capital, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized": round(self.unrealized(mid), 2) if mid is not None else 0.0,
            "fees_paid": round(self.fees_paid, 2),
            "n_fills": len(self.fills),
            "n_round_trips": len(self.round_trips),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": round(len(wins) / len(self.round_trips), 4)
            if self.round_trips
            else 0.0,
            "position": self.position.value,
            "lots": self.lots,
            "n_rejects": len(self.rejects),
        }

    def to_session_dict(
        self,
        *,
        meta: dict[str, Any] | None = None,
        mid: float | None = None,
        include_marks: bool = False,
    ) -> dict[str, Any]:
        """一次覆盤／紙上 session 的完整結果（供落盤）。"""
        return {
            "status": "DONE",
            "brand": "tidemark",
            "kind": "paper_session",
            "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "meta": meta or {},
            "summary": self.summary(mid),
            "fills": self.fills,
            "round_trips": self.round_trips,
            "rejects": self.rejects,
            "equity_marks": self.equity_marks if include_marks else [],
        }

    def save_session(
        self,
        path: Path | str | None = None,
        *,
        meta: dict[str, Any] | None = None,
        mid: float | None = None,
        include_marks: bool = False,
        root: Path | None = None,
    ) -> Path:
        """
        寫入覆盤結果。預設：
        outputs/paper_sessions/{session_id}.json
        """
        if path is None:
            base = root or Path("outputs/paper_sessions")
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"{self.session_id}.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_session_dict(meta=meta, mid=mid, include_marks=include_marks)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"{notify_prefix()} paper session saved -> {path}")
        return path
