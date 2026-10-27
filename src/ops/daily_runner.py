"""
全自動日循環（Paper 模擬下單）：

  08:40 啟動盤中（subscribe 或 watch）
  13:50 停止盤中並封存
  18:00 晚間覆盤（1m 重播／sim_day）寫入 days/

禁止 LIVE；place_sim 僅當 ops.yaml / env 明確開啟。
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import yaml
from loguru import logger

from src.core.config import ROOT, load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.ops.ledger import OpsLedger, ledger_from_ops_cfg
from src.trading.live_tick_loop import LiveTickConfig, LiveTickLoop
from src.trading.sim_day_loop import SimDayConfig, SimDayLoop

TZ = ZoneInfo("Asia/Taipei")


def load_ops_config(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "config" / "ops.yaml")
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = str(s).strip().split(":")
    return int(hh), int(mm)


def _mins_now(now: datetime) -> int:
    return now.hour * 60 + now.minute


def _hhmm_mins(s: str) -> int:
    h, m = _parse_hhmm(s)
    return h * 60 + m


def _has_sim_keys() -> bool:
    return bool(
        os.environ.get("SHIOAJI_SIM_API_KEY", "").strip()
        and os.environ.get("SHIOAJI_SIM_SECRET_KEY", "").strip()
    )


class DailyOpsRunner:
    """單進程排程：盤中 thread + 主迴圈 phase。"""

    def __init__(self, ops_cfg: dict[str, Any] | None = None):
        self.ops_cfg = ops_cfg or load_ops_config()
        self.settings = load_settings()
        self.prefix = notify_prefix(self.settings)
        self.ledger: OpsLedger = ledger_from_ops_cfg(self.ops_cfg)
        sched = self.ops_cfg.get("schedule") or {}
        self.day_start = _hhmm_mins(str(sched.get("day_start", "08:40")))
        self.day_end = _hhmm_mins(str(sched.get("day_end", "13:50")))
        self.evening = _hhmm_mins(str(sched.get("evening_review", "18:00")))
        self.place_sim = bool(self.ops_cfg.get("place_sim", False)) or (
            os.environ.get("OPS_PLACE_SIM", "").strip() == "1"
        )
        self._session_thread: threading.Thread | None = None
        self._session_loop: LiveTickLoop | None = None
        self._session_stop = threading.Event()
        self._review_done_for: str | None = None
        self._phase = "boot"

    def _now(self) -> datetime:
        return datetime.now(TZ)

    def _date_str(self, now: datetime | None = None) -> str:
        return (now or self._now()).strftime("%Y-%m-%d")

    def _set_phase(self, phase: str, **extra: Any) -> None:
        self._phase = phase
        self.ledger.write_status(
            {
                "phase": phase,
                "date": self._date_str(),
                "place_sim": self.place_sim,
                "mode": "paper",
                "has_sim_keys": _has_sim_keys(),
                **extra,
            }
        )

    def _run_session_subscribe(self, date: str) -> None:
        """盤中訂閱（Paper；可選 place_sim）。"""
        cfg = LiveTickConfig(
            date=date,
            dry_run=not self.place_sim,
            place_sim=self.place_sim,
            connect_sim=True,
            session_filter="day",
            risk_check_seconds=10,
        )
        start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=14)).strftime(
            "%Y-%m-%d"
        )
        seed_all = load_1m_range(start, date)
        seed = None
        if seed_all is not None and not seed_all.empty:
            import pandas as pd

            seed_all = seed_all.copy()
            seed_all["ts"] = pd.to_datetime(seed_all["ts"])
            seed = seed_all[seed_all["ts"].dt.strftime("%Y-%m-%d") < date]

        loop = LiveTickLoop(self.settings, cfg=cfg, seed_bars=seed)
        self._session_loop = loop

        def _on_event(kind: str, data: dict[str, Any]) -> None:
            if kind in ("choice", "fill", "subscribed"):
                self.ledger.append_journal(f"live_{kind}", data)
                self._set_phase(
                    "session_day",
                    live={"kind": kind, "n_events": len(loop._events)},
                )

        loop.on_event = _on_event
        self.ledger.append_journal("session_start", {"date": date, "mode": "subscribe"})
        try:
            now = self._now()
            remain = max(60, (self.day_end - _mins_now(now)) * 60)
            # 另開 watchdog：day_end / stop 時停 loop；每 30 秒更新盤中心跳
            def _watch() -> None:
                last_hb = 0.0
                while not self._session_stop.is_set():
                    n = self._now()
                    if self._date_str(n) != date or _mins_now(n) >= self.day_end:
                        loop.stop()
                        break
                    now_t = time.time()
                    if now_t - last_hb >= 30.0:
                        last_hb = now_t
                        self._set_phase(
                            "session_day",
                            live={
                                "kind": "heartbeat",
                                "n_bars": getattr(loop, "_n_bars_closed", 0),
                                "n_events": len(loop._events),
                                "choice": (loop._choice_info or {}).get("choice"),
                            },
                        )
                    time.sleep(5)

            threading.Thread(target=_watch, name="session-watch", daemon=True).start()
            res = loop.subscribe_and_run(max_seconds=float(remain))
            if not res.ok and (res.payload or {}).get("error"):
                self.ledger.append_journal(
                    "session_error", {"error": res.payload.get("error")}
                )
        finally:
            try:
                snap = loop.snapshot(status="SESSION_END")
                path = self.ledger.write_day(
                    date,
                    {
                        "kind": "session",
                        "signal": snap.get("choice"),
                        "strategy": {"risk": snap.get("risk"), "modules": snap.get("modules")},
                        "orders": {
                            "events": snap.get("events"),
                            "summary": snap.get("summary"),
                            "fills": snap.get("fills"),
                        },
                        "live_tick": snap,
                    },
                )
                self.ledger.append_journal(
                    "session_end", {"date": date, "path": str(path)}
                )
            finally:
                loop.close()
                self._session_loop = None

    def _run_evening_review(self, date: str) -> Path:
        """晚間覆盤：1m 重播（有 cache）否則 sim_day。"""
        self._set_phase("evening_review", review_date=date)
        self.ledger.append_journal("evening_start", {"date": date})
        start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=14)).strftime(
            "%Y-%m-%d"
        )
        bars = load_1m_range(start, date)
        payload: dict[str, Any] = {"date": date, "ok": False}

        if bars is not None and not bars.empty:
            import pandas as pd

            bars = bars.copy()
            bars["ts"] = pd.to_datetime(bars["ts"])
            seed = bars[bars["ts"].dt.strftime("%Y-%m-%d") < date]
            day = bars[bars["ts"].dt.strftime("%Y-%m-%d") == date]
            if "session" in day.columns:
                day = day[day["session"] == "day"]
            if not day.empty:
                loop = LiveTickLoop(
                    self.settings,
                    cfg=LiveTickConfig(
                        date=date,
                        dry_run=True,
                        place_sim=False,
                        connect_sim=False,
                        session_filter="day",
                    ),
                    seed_bars=seed,
                )
                try:
                    res = loop.replay_1m(day)
                    snap = res.payload
                    payload = {
                        "ok": res.ok,
                        "method": "live_tick_replay_1m",
                        "signal": snap.get("choice"),
                        "strategy": {
                            "risk": snap.get("risk"),
                            "modules": snap.get("modules"),
                        },
                        "orders": {
                            "events": snap.get("events"),
                            "summary": snap.get("summary"),
                            "round_trips": snap.get("round_trips"),
                        },
                        "net_pnl": (snap.get("summary") or {}).get("net_pnl"),
                    }
                finally:
                    loop.close()

        if not payload.get("ok"):
            # fallback sim_day
            sim = SimDayLoop(
                self.settings,
                cfg=SimDayConfig(
                    date=date, dry_run=True, place_sim=False, connect_sim=False
                ),
            )
            try:
                if bars is None or bars.empty:
                    payload = {"ok": False, "error": "no_bars", "method": "sim_day"}
                else:
                    res = sim.run_once(bars)
                    snap = res.payload
                    payload = {
                        "ok": res.ok,
                        "method": "sim_day",
                        "signal": snap.get("choice"),
                        "strategy": {"risk": snap.get("risk")},
                        "orders": {
                            "events": snap.get("events"),
                            "summary": snap.get("summary"),
                            "round_trips": snap.get("round_trips"),
                        },
                        "net_pnl": (snap.get("summary") or {}).get("net_pnl"),
                    }
            finally:
                sim.close()

        prev = self.ledger.read_day(date)
        day_body = {
            **prev,
            "kind": prev.get("kind") or "evening_review",
            "review": payload,
            # 覆盤也同步三分欄，方便 UI 夜間查看
            "signal": payload.get("signal") or prev.get("signal"),
            "strategy": payload.get("strategy") or prev.get("strategy"),
            "orders": payload.get("orders") or prev.get("orders"),
        }
        path = self.ledger.write_day(date, day_body)
        latest = ROOT / "outputs" / "ops" / "evening_latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(
            path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.ledger.append_journal(
            "evening_done",
            {"date": date, "path": str(path), "net_pnl": payload.get("net_pnl")},
        )
        self._set_phase("idle_night", last_review=date, review_path=str(path))
        return path

    def _start_session_if_needed(self, date: str) -> None:
        if self._session_thread and self._session_thread.is_alive():
            return
        self._session_stop.clear()
        if not _has_sim_keys():
            self.ledger.append_journal(
                "session_skip",
                {"reason": "no_sim_keys", "hint": "fill SHIOAJI_SIM_* in .env"},
            )
            self._set_phase(
                "session_day",
                warning="no_sim_keys",
                note="盤中訂閱略過；晚間仍會用 cache 覆盤",
            )
            return

        def _target() -> None:
            try:
                self._run_session_subscribe(date)
            except Exception as e:
                logger.exception(f"{self.prefix} session crashed: {e}")
                self.ledger.append_journal("session_crash", {"error": str(e)})

        self._session_thread = threading.Thread(
            target=_target, name="tidemark-session", daemon=True
        )
        self._session_thread.start()
        self._set_phase("session_day", session="running")

    def _stop_session(self) -> None:
        self._session_stop.set()
        if self._session_loop is not None:
            try:
                self._session_loop.stop()
            except Exception:
                pass
        if self._session_thread and self._session_thread.is_alive():
            self._session_thread.join(timeout=30)
        self._session_thread = None

    def tick(self) -> None:
        """主迴圈呼叫一次（約每 15–30 秒）。"""
        now = self._now()
        date = self._date_str(now)
        mins = _mins_now(now)

        # 換日重置覆盤旗標
        if self._review_done_for and self._review_done_for != date:
            self._review_done_for = None

        if self.day_start <= mins < self.day_end:
            self._start_session_if_needed(date)
            return

        if mins >= self.day_end and self._session_thread and self._session_thread.is_alive():
            self._stop_session()
            self._set_phase("post_session", note="day session stopped")

        if mins >= self.evening and self._review_done_for != date:
            # 週末也可跑（有 cache 才有結果）
            try:
                self._run_evening_review(date)
            except Exception as e:
                logger.exception(f"{self.prefix} evening review failed: {e}")
                self.ledger.append_journal("evening_error", {"error": str(e)})
                self._set_phase("evening_error", error=str(e))
            self._review_done_for = date
            return

        if mins < self.day_start:
            self._set_phase("idle_pre", next="day_start")
        elif self.day_end <= mins < self.evening:
            self._set_phase("post_session", next="evening_review")
        else:
            self._set_phase("idle_night", last_review=self._review_done_for)

    def run_forever(self, *, poll_seconds: float = 20.0) -> None:
        self.ledger.append_journal("ops_boot", {"place_sim": self.place_sim})
        self._set_phase("boot")
        logger.info(
            f"{self.prefix} ops daemon start place_sim={self.place_sim} "
            f"day={self.day_start}-{self.day_end} evening={self.evening}"
        )
        try:
            while True:
                try:
                    self.tick()
                except Exception as e:
                    logger.exception(f"{self.prefix} ops tick error: {e}")
                    self.ledger.append_journal("ops_tick_error", {"error": str(e)})
                time.sleep(max(5.0, float(poll_seconds)))
        except KeyboardInterrupt:
            self.ledger.append_journal("ops_stop", {})
            self._stop_session()
            self._set_phase("stopped")
