"""
日流程自動化（互補分流 → 一口執行）。

預設：PaperBroker 記帳 + Sinopac dry_run（不送券商真單）。
--place-sim：對永豐 SIM API 真下限價（仍 simulation；禁 LIVE）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
from loguru import logger

from src.backtest.complementary_fast import ComplementaryConfig
from src.core.config import notify_prefix
from src.core.types import TradeAction
from src.signals.complementary import ComplementarySignal
from src.strategy.complementary_risk import ComplementaryRiskStrategy
from src.trading.paper import PaperBroker
from src.trading.sinopac import SinopacFuturesTrader


def _ts_key(ts: Any) -> str:
    return str(pd.Timestamp(ts))


@dataclass
class SimDayConfig:
    date: str
    dry_run: bool = True  # True=Sinopac dry；False 需 place_sim
    place_sim: bool = False  # 對 SIM API 真 place_order
    connect_sim: bool = True  # 嘗試連線（即便 dry 也可連看報價）


@dataclass
class SimDayResult:
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)


class SimDayLoop:
    """
    單日自動化：
    1) 探針特徵分流 orb|rsi|flat
    2) 用互補引擎算出當日成交意圖
    3) Paper 執行；可選 mirror 到 Sinopac（dry 或 SIM 真單）
    """

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        cfg: SimDayConfig,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.settings = settings
        self.cfg = cfg
        self.on_event = on_event or (lambda *_: None)
        trading = settings["trading"]
        self.paper = PaperBroker.from_settings(
            settings, session_id=f"simday_{cfg.date.replace('-', '')}"
        )
        self.trader: SinopacFuturesTrader | None = None
        self.comp_cfg = ComplementaryConfig.from_locked(
            capital=float(trading["capital"]),
            point_value=float(trading["point_value"]),
            fee_per_side=float(trading.get("fee_per_side", 20)),
            tax_rate=float(trading.get("tax_rate", 0.00002)),
            slip_ticks=float(settings.get("backtest", {}).get("slip_ticks", 1)),
        )
        self.signal = ComplementarySignal(self.comp_cfg)
        self.risk = ComplementaryRiskStrategy(risk=self.comp_cfg.risk_dict())
        self._executed_keys: set[str] = set()

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        self.on_event(kind, data)
        logger.info(f"{notify_prefix(self.settings)} sim_day {kind} {data}")

    def _ensure_trader(self) -> SinopacFuturesTrader:
        if self.trader is None:
            self.trader = SinopacFuturesTrader(
                simulation=True, allow_live=False, max_lots=1
            )
            if self.cfg.connect_sim or self.cfg.place_sim:
                self.trader.connect()
        return self.trader

    def _mirror(self, action: TradeAction, mid: float, reason: str) -> dict[str, Any]:
        """同步意圖到 Sinopac：預設 dry_run；place_sim 才真下；--no-connect 略過。"""
        if not self.cfg.connect_sim and not self.cfg.place_sim:
            return {
                "ok": True,
                "skipped": True,
                "dry_run": True,
                "place_sim": False,
                "action": action.value,
                "price": float(mid),
                "reason": reason,
            }
        trader = self._ensure_trader()
        use_dry = not self.cfg.place_sim
        ok = trader.place_tmf(
            action, float(mid), reason=reason, dry_run=use_dry
        )
        return {
            "ok": ok,
            "dry_run": use_dry,
            "place_sim": self.cfg.place_sim,
            "action": action.value,
            "price": float(mid),
            "reason": reason,
            "trader": trader.summary(),
        }

    def run_once(self, bars: pd.DataFrame) -> SimDayResult:
        date = self.cfg.date
        as_of = bars.iloc[-1]["ts"] if bars is not None and not bars.empty else date
        intents, meta, choice_info = self.signal.intents(
            bars, date=date, as_of_ts=as_of
        )
        self._emit("choice", choice_info)
        if not choice_info.get("ok"):
            return SimDayResult(
                ok=False, payload={"error": "choice_failed", "choice": choice_info}
            )

        choice = choice_info.get("choice")
        events: list[dict[str, Any]] = []

        if choice == "flat":
            payload = self._build_payload(
                choice_info=choice_info,
                events=events,
                note="flat_no_trade",
                bars=bars,
            )
            return SimDayResult(ok=True, payload=payload)

        self._emit(
            "intents",
            {"date": date, "choice": choice, "n_intents": len(intents), "meta": meta},
        )

        slip = float(self.paper.slip_ticks)
        for intent in intents:
            if not self.risk.allow_bar_intent(
                intent, date=date, session_end=True
            ):
                continue
            key = intent.key()
            if key in self._executed_keys:
                continue
            mid = self.risk.intent_mid(intent, slip_ticks=slip)
            ok_paper = self.paper.place(
                intent.action, mid, intent.reason, ts=intent.ts
            )
            mirror = self._mirror(intent.action, mid, intent.reason)
            ev = {
                "ts": _ts_key(intent.ts),
                "action": intent.action.value,
                "mid": mid,
                "intent_price": float(intent.price),
                "reason": intent.reason,
                "track": intent.track,
                "paper_ok": ok_paper,
                "sinopac": mirror,
            }
            events.append(ev)
            self._executed_keys.add(key)
            self._emit("fill", ev)

        payload = self._build_payload(
            choice_info=choice_info,
            events=events,
            note="executed",
            bars=bars,
            complementary_meta=meta,
        )
        return SimDayResult(ok=True, payload=payload)

    def _build_payload(
        self,
        *,
        choice_info: dict[str, Any],
        events: list[dict[str, Any]],
        note: str,
        bars: pd.DataFrame,
        complementary_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_mid = None
        if bars is not None and not bars.empty and "close" in bars.columns:
            last_mid = float(bars.iloc[-1]["close"])
        summary = self.paper.summary(last_mid)
        # 若完全沒成交，仍給空 trips
        trips = list(self.paper.round_trips)
        return {
            "status": "DONE",
            "brand": "tidemark",
            "kind": "sim_day",
            "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "date": self.cfg.date,
            "dry_run": self.cfg.dry_run and not self.cfg.place_sim,
            "place_sim": self.cfg.place_sim,
            "choice": choice_info,
            "note": note,
            "summary": summary,
            "events": events,
            "round_trips": trips,
            "fills": self.paper.fills,
            "rejects": self.paper.rejects,
            "sinopac_summary": self.trader.summary() if self.trader else None,
            "complementary_meta": complementary_meta,
            "risk": self.comp_cfg.risk_dict(),
            "max_lots": 1,
        }

    def close(self) -> None:
        if self.trader is not None:
            self.trader.close()
            self.trader = None
