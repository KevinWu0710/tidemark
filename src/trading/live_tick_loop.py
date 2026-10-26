"""
盤中即時 tick 編排器（薄）：

  data(tick→1m) → ComplementarySignal → ComplementaryRiskStrategy → Paper/Sinopac

無偷看：策略／訊號只吃已關閉 1m；探針結束前不分流成交。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from src.backtest.complementary_fast import ComplementaryConfig
from src.core.config import notify_prefix
from src.core.types import TradeAction
from src.data.minute_bars import MinuteBarBuilder, bars_frame
from src.data.sessions import label_session
from src.signals.complementary import ComplementarySignal
from src.strategy.complementary_risk import (
    ComplementaryRiskStrategy,
    is_session_end,
    probe_ready,
)
from src.trading.paper import PaperBroker
from src.trading.sinopac import SinopacFuturesTrader


def _ts_key(ts: Any) -> str:
    return str(pd.Timestamp(ts))


@dataclass
class LiveTickConfig:
    date: str
    dry_run: bool = True
    place_sim: bool = False
    connect_sim: bool = False
    session_filter: str = "day"
    risk_check_seconds: int = 10


@dataclass
class LiveTickResult:
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)


class LiveTickLoop:
    """即時／重播：on_tick → 關棒 → signal → risk → place。"""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        cfg: LiveTickConfig,
        seed_bars: pd.DataFrame | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.settings = settings
        self.cfg = cfg
        self.on_event = on_event or (lambda *_: None)
        trading = settings["trading"]
        self.paper = PaperBroker.from_settings(
            settings, session_id=f"livetick_{cfg.date.replace('-', '')}"
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
        self.risk = ComplementaryRiskStrategy(
            risk=self.comp_cfg.risk_dict(),
            risk_check_seconds=int(cfg.risk_check_seconds or 10),
        )
        self.builder = MinuteBarBuilder()
        self._seed = (
            seed_bars.copy()
            if seed_bars is not None and not seed_bars.empty
            else bars_frame([])
        )
        if not self._seed.empty and "ts" in self._seed.columns:
            self._seed["ts"] = pd.to_datetime(self._seed["ts"])
            self._seed = self._seed.sort_values("ts").reset_index(drop=True)
        self._live_rows: list[dict[str, Any]] = []
        self._executed_keys: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._choice_info: dict[str, Any] | None = None
        self._last_meta: dict[str, Any] | None = None
        self._n_ticks = 0
        self._n_bars_closed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._subscribed = False

    # --- compat aliases for tests / callers ---
    @property
    def _completed_days(self) -> set[str]:
        return self.risk.completed_days

    @property
    def _active_entry_price(self) -> Optional[float]:
        return self.risk.entry_price

    @_active_entry_price.setter
    def _active_entry_price(self, v: Optional[float]) -> None:
        self.risk.entry_price = v

    @property
    def _active_entry_ts(self) -> Optional[pd.Timestamp]:
        return self.risk.entry_ts

    @_active_entry_ts.setter
    def _active_entry_ts(self, v: Optional[pd.Timestamp]) -> None:
        self.risk.entry_ts = pd.Timestamp(v) if v is not None else None

    @property
    def _active_max_price(self) -> Optional[float]:
        return self.risk.max_price

    @_active_max_price.setter
    def _active_max_price(self, v: Optional[float]) -> None:
        self.risk.max_price = v

    @property
    def _active_track(self) -> Optional[str]:
        return self.risk.track

    @_active_track.setter
    def _active_track(self, v: Optional[str]) -> None:
        self.risk.track = v

    @property
    def _last_risk_check_ts(self) -> Optional[pd.Timestamp]:
        return self.risk.last_risk_check_ts

    @_last_risk_check_ts.setter
    def _last_risk_check_ts(self, v: Optional[pd.Timestamp]) -> None:
        self.risk.last_risk_check_ts = pd.Timestamp(v) if v is not None else None

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        self.on_event(kind, data)
        logger.info(f"{notify_prefix(self.settings)} live_tick {kind} {data}")

    def _ensure_trader(self) -> SinopacFuturesTrader:
        if self.trader is None:
            self.trader = SinopacFuturesTrader(
                simulation=True, allow_live=False, max_lots=1
            )
            if self.cfg.connect_sim or self.cfg.place_sim:
                self.trader.connect()
        return self.trader

    def _mirror(self, action: TradeAction, mid: float, reason: str) -> dict[str, Any]:
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
        ok = trader.place_tmf(action, float(mid), reason=reason, dry_run=use_dry)
        return {
            "ok": ok,
            "dry_run": use_dry,
            "place_sim": self.cfg.place_sim,
            "action": action.value,
            "price": float(mid),
            "reason": reason,
            "trader": trader.summary(),
        }

    def _combined_bars(self) -> pd.DataFrame:
        live = bars_frame(self._live_rows)
        if self._seed.empty:
            return live
        if live.empty:
            return self._seed.copy()
        seed = self._seed[~self._seed["ts"].isin(live["ts"])]
        out = pd.concat([seed, live], ignore_index=True)
        return out.sort_values("ts").reset_index(drop=True)

    def _session_ok(self, ts: pd.Timestamp) -> bool:
        sess = label_session(ts)
        want = self.cfg.session_filter
        if want == "all":
            return sess in ("day", "night")
        return sess == want

    def _place_close_risk(self, exit_intent) -> list[dict[str, Any]]:
        slip = float(self.paper.slip_ticks)
        mid = self.risk.close_mid(exit_intent.mark_price, slip_ticks=slip)
        key = f"{_ts_key(exit_intent.ts)}|CLOSE|{exit_intent.reason}"
        if key in self._executed_keys:
            return []
        ok_paper = self.paper.place(
            TradeAction.CLOSE, mid, exit_intent.reason, ts=exit_intent.ts
        )
        if not ok_paper:
            self._executed_keys.add(key)
            return []
        mirror = self._mirror(TradeAction.CLOSE, mid, exit_intent.reason)
        self.risk.note_exit(date=str(pd.Timestamp(exit_intent.ts).date()))
        ev = {
            "ts": _ts_key(exit_intent.ts),
            "action": TradeAction.CLOSE.value,
            "mid": mid,
            "intent_price": float(exit_intent.mark_price),
            "reason": exit_intent.reason,
            "track": exit_intent.track,
            "paper_ok": ok_paper,
            "sinopac": mirror,
            "as_of": _ts_key(exit_intent.ts),
            "_choice": exit_intent.track,
        }
        self._events.append(ev)
        self._executed_keys.add(key)
        self._emit("fill", {k: v for k, v in ev.items() if k != "_choice"})
        return [ev]

    def _check_position_risk_unlocked(
        self, t: pd.Timestamp, price: float
    ) -> list[dict[str, Any]]:
        exit_intent = self.risk.check_tick(t, price, lots=int(self.paper.lots))
        if exit_intent is None:
            return []
        return self._place_close_risk(exit_intent)

    def on_tick(self, ts: Any, price: float, volume: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            self._n_ticks += 1
            t = pd.Timestamp(ts)
            events: list[dict[str, Any]] = []
            if self.paper.lots > 0 and self.risk.in_position():
                risk_ev = self._check_position_risk_unlocked(t, float(price))
                if risk_ev:
                    events.extend(risk_ev)
            if not self._session_ok(t):
                return events
            closed = self.builder.on_tick(t, float(price), float(volume or 0))
            if closed is not None:
                bar_events = self._on_bar_closed_unlocked(closed)
                if bar_events:
                    events.extend(bar_events)
            return events

    def on_bar(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            return self._on_bar_closed_unlocked(bar)

    def flush(self) -> list[dict[str, Any]]:
        with self._lock:
            closed = self.builder.flush()
            if closed is None:
                return []
            return self._on_bar_closed_unlocked(closed)

    def _on_bar_closed_unlocked(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        ts = pd.Timestamp(bar["ts"])
        if not self._session_ok(ts):
            return []
        if self._live_rows and pd.Timestamp(self._live_rows[-1]["ts"]) == ts:
            self._live_rows[-1] = bar
        else:
            self._live_rows.append(bar)
        self._n_bars_closed += 1
        if self._n_bars_closed == 1 or self._n_bars_closed % 30 == 0:
            self._emit(
                "bar",
                {
                    "ts": _ts_key(ts),
                    "close": float(bar["close"]),
                    "session": bar.get("session"),
                    "n_bars": self._n_bars_closed,
                },
            )
        return self._evaluate_unlocked(as_of_ts=ts)

    def _evaluate_unlocked(self, *, as_of_ts: pd.Timestamp) -> list[dict[str, Any]]:
        date = self.cfg.date
        if not probe_ready(
            as_of_ts, probe_end_minutes=self.signal.probe_end_minutes
        ):
            return []

        bars = self._combined_bars()
        bars = bars[pd.to_datetime(bars["ts"]) <= as_of_ts].copy()
        if bars.empty:
            return []

        intents, meta, choice_info = self.signal.intents(
            bars, date=date, as_of_ts=as_of_ts
        )
        if not choice_info.get("ok") or str(choice_info.get("date")) != date:
            return []

        prev_ch = (self._choice_info or {}).get("choice")
        self._choice_info = choice_info
        ch = choice_info.get("choice")
        if prev_ch != ch:
            self._emit(
                "choice",
                {k: v for k, v in choice_info.items() if k != "features"},
            )
        if ch == "flat":
            return []

        self._last_meta = meta
        session_end = is_session_end(as_of_ts)
        slip = float(self.paper.slip_ticks)
        new_events: list[dict[str, Any]] = []

        for intent in intents:
            if not self.risk.allow_bar_intent(
                intent, date=date, session_end=session_end
            ):
                continue
            key = intent.key()
            if key in self._executed_keys:
                continue
            mid = self.risk.intent_mid(intent, slip_ticks=slip)
            ok_paper = self.paper.place(
                intent.action, mid, intent.reason, ts=intent.ts
            )
            if not ok_paper:
                self._executed_keys.add(key)
                continue
            if intent.action == TradeAction.BUY:
                self.risk.note_entry(
                    fill_price=float(
                        self.paper.entry_price or (mid + slip)
                    ),
                    ts=intent.ts,
                    track=intent.track or ch,
                )
            elif intent.action == TradeAction.CLOSE:
                self.risk.note_exit(date=str(pd.Timestamp(intent.ts).date()))

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
                "as_of": _ts_key(as_of_ts),
                "_choice": ch,
            }
            self._events.append(ev)
            self._executed_keys.add(key)
            new_events.append(ev)
            self._emit("fill", {k: v for k, v in ev.items() if k != "_choice"})
        return new_events

    def replay_1m(self, day_bars: pd.DataFrame) -> LiveTickResult:
        if day_bars is None or day_bars.empty:
            return LiveTickResult(ok=False, payload={"error": "no_bars"})
        df = day_bars.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
        for _, row in df.iterrows():
            if self._stop.is_set():
                break
            bar = {
                "ts": row["ts"],
                "open": float(row.get("open", row["close"])),
                "high": float(row.get("high", row["close"])),
                "low": float(row.get("low", row["close"])),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0) or 0),
                "session": row.get("session") or label_session(row["ts"]),
            }
            self.on_bar(bar)
        self.flush()
        return LiveTickResult(ok=True, payload=self.snapshot(status="DONE"))

    def replay_ticks(self, ticks: pd.DataFrame) -> LiveTickResult:
        if ticks is None or ticks.empty:
            return LiveTickResult(ok=False, payload={"error": "no_ticks"})
        df = ticks.copy()
        if "ts" not in df.columns:
            if "date" in df.columns:
                df["ts"] = pd.to_datetime(df["date"])
            else:
                return LiveTickResult(ok=False, payload={"error": "tick_missing_ts"})
        if "price" not in df.columns and "close" in df.columns:
            df["price"] = df["close"]
        df = df.dropna(subset=["ts", "price"]).sort_values("ts")
        for _, row in df.iterrows():
            if self._stop.is_set():
                break
            self.on_tick(row["ts"], float(row["price"]), float(row.get("volume", 0) or 0))
        self.flush()
        return LiveTickResult(ok=True, payload=self.snapshot(status="DONE"))

    def subscribe_and_run(self, *, max_seconds: float = 0) -> LiveTickResult:
        self.cfg.connect_sim = True
        trader = self._ensure_trader()
        if trader.api is None:
            return LiveTickResult(ok=False, payload={"error": "not_connected"})
        try:
            import shioaji as sj

            contract = trader._contract()

            def _cb(_exchange, tick):
                if self._stop.is_set():
                    return
                try:
                    if bool(getattr(tick, "simtrade", 0)):
                        return
                    ts = getattr(tick, "datetime", None) or getattr(tick, "time", None)
                    px = float(
                        getattr(tick, "close", 0) or getattr(tick, "price", 0) or 0
                    )
                    vol = float(getattr(tick, "volume", 0) or 0)
                    if px <= 0:
                        return
                    self.on_tick(ts, px, vol)
                except Exception as e:
                    logger.warning(f"{notify_prefix(self.settings)} tick cb error: {e}")

            set_cb = getattr(trader.api.quote, "set_on_tick_fop_v1_callback", None)
            if set_cb is None:
                return LiveTickResult(
                    ok=False, payload={"error": "no_set_on_tick_fop_v1_callback"}
                )
            set_cb(_cb)
            trader.api.quote.subscribe(contract, quote_type=sj.QuoteType.Tick)
            self._subscribed = True
            self._emit(
                "subscribed",
                {"contract": str(getattr(contract, "code", "")), "date": self.cfg.date},
            )

            t0 = time.time()
            while not self._stop.is_set():
                if max_seconds and (time.time() - t0) >= max_seconds:
                    break
                now = datetime.now()
                if self.cfg.date == now.strftime("%Y-%m-%d"):
                    mins = now.hour * 60 + now.minute
                    if self.cfg.session_filter == "day" and mins > (13 * 60 + 45):
                        break
                time.sleep(1.0)
        except Exception as e:
            return LiveTickResult(ok=False, payload={"error": f"subscribe_failed:{e}"})
        finally:
            self.flush()
            try:
                if self._subscribed and trader.api is not None:
                    import shioaji as sj

                    contract = trader._contract()
                    trader.api.quote.unsubscribe(
                        contract, quote_type=sj.QuoteType.Tick
                    )
            except Exception:
                pass
            self._subscribed = False
        return LiveTickResult(ok=True, payload=self.snapshot(status="DONE"))

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self, *, status: str = "RUNNING") -> dict[str, Any]:
        bars = self._combined_bars()
        last_mid = None
        if not bars.empty and "close" in bars.columns:
            last_mid = float(bars.iloc[-1]["close"])
        summary = self.paper.summary(last_mid)
        return {
            "status": status,
            "brand": "tidemark",
            "kind": "live_tick",
            "saved_at": datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds"),
            "date": self.cfg.date,
            "dry_run": self.cfg.dry_run and not self.cfg.place_sim,
            "place_sim": self.cfg.place_sim,
            "connect_sim": self.cfg.connect_sim,
            "session_filter": self.cfg.session_filter,
            "n_ticks": self._n_ticks,
            "n_bars_closed": self._n_bars_closed,
            "choice": self._choice_info,
            "summary": summary,
            "events": [
                {k: v for k, v in e.items() if k != "_choice"} for e in self._events
            ],
            "round_trips": list(self.paper.round_trips),
            "fills": self.paper.fills,
            "rejects": self.paper.rejects,
            "sinopac_summary": self.trader.summary() if self.trader else None,
            "complementary_meta": self._last_meta,
            "risk": self.comp_cfg.risk_dict(),
            "max_lots": 1,
            "probe_end_minutes": self.signal.probe_end_minutes,
            "modules": {
                "signal": "ComplementarySignal",
                "strategy": "ComplementaryRiskStrategy",
                "trading": "PaperBroker+Sinopac",
            },
        }

    def close(self) -> None:
        self.stop()
        if self.trader is not None:
            self.trader.close()
            self.trader = None
