#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.core.types import TradeAction
from src.data.bars import load_1m_range
from src.trading.live_tick_loop import LiveTickConfig, LiveTickLoop


class TestLiveTickLoop(unittest.TestCase):
    def test_replay_1m_causal(self):
        settings = load_settings()
        date = "2026-10-16"
        bars = load_1m_range("2026-10-01", date)
        if bars is None or bars.empty:
            self.skipTest("no 1m cache")
        bars = bars.copy()
        bars["ts"] = pd.to_datetime(bars["ts"])
        seed = bars[bars["ts"].dt.strftime("%Y-%m-%d") < date]
        day = bars[bars["ts"].dt.strftime("%Y-%m-%d") == date]
        if "session" in day.columns:
            day = day[day["session"] == "day"]
        if day.empty:
            self.skipTest("no day bars")
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date=date, dry_run=True, place_sim=False, connect_sim=False
            ),
            seed_bars=seed,
        )
        try:
            res = loop.replay_1m(day)
            self.assertTrue(res.ok)
            self.assertEqual(res.payload.get("kind"), "live_tick")
            self.assertTrue(res.payload.get("dry_run"))
            self.assertGreater(res.payload.get("n_bars_closed", 0), 0)
            choice = res.payload.get("choice") or {}
            if choice.get("ok"):
                self.assertEqual(choice.get("date"), date)
            # 不應在探針結束前成交；不應有提早 flatten_eod
            for ev in res.payload.get("events") or []:
                mins = pd.Timestamp(ev["ts"]).hour * 60 + pd.Timestamp(ev["ts"]).minute
                self.assertGreaterEqual(mins, 9 * 60)
                if ev.get("reason") == "flatten_eod":
                    self.assertGreaterEqual(mins, 13 * 60 + 45)
            # 與全日 batch 對齊：有成交日應有成功 fill
            ok_fills = [e for e in (res.payload.get("events") or []) if e.get("paper_ok")]
            self.assertGreaterEqual(len(ok_fills), 1)
        finally:
            loop.close()

    def test_tick_builder_feeds_loop(self):
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02", dry_run=True, place_sim=False, connect_sim=False
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            # 合成兩分鐘 tick；無完整特徵時 choice 可能 fail，但關棒應累積
            loop.on_tick("2026-01-02 09:01:10", 18000.0, 1)
            loop.on_tick("2026-01-02 09:01:50", 18010.0, 1)
            ev = loop.on_tick("2026-01-02 09:02:01", 18005.0, 1)
            self.assertIsInstance(ev, list)
            self.assertEqual(loop._n_bars_closed, 1)
        finally:
            loop.close()

    def test_position_risk_check_10s(self):
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            # 模擬建多單一口；風控基準用實際成交價
            t0 = pd.Timestamp("2026-01-02 09:10:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            # 5 秒後價格跌破 50 點（達 19940），但因未滿 10 秒檢查間隔，抗毛刺不平倉
            ev1 = loop.on_tick("2026-01-02 09:10:05", 19940.0, 1)
            self.assertEqual(len(ev1), 0)
            self.assertEqual(loop.paper.lots, 1)

            # 滿 10 秒後價格仍在 19940，風控確認跌破，果斷止血平倉
            ev2 = loop.on_tick("2026-01-02 09:10:10", 19940.0, 1)
            self.assertEqual(len(ev2), 1)
            self.assertEqual(ev2[0]["reason"], "stop_loss")
            self.assertEqual(loop.paper.lots, 0)
        finally:
            loop.close()

    def test_position_risk_eod_held_under_45m(self):
        """問題 1 覆核：ORB 未滿 45 分鐘時，到 13:45 收盤也必須正常觸發 flatten_eod。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            # 13:40 進場（僅持倉 5 分鐘，遠小於 45 分鐘 time_stop）
            t0 = pd.Timestamp("2026-01-02 13:40:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price or 20001.0)
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            # 13:45:01 收盤；價在停損／停利帶外 → reason 應為 flatten_eod
            mid_px = fill_entry + 10.0
            ev = loop.on_tick("2026-01-02 13:45:01", mid_px, 1)
            self.assertEqual(len(ev), 1)
            self.assertEqual(ev[0]["reason"], "flatten_eod")
            self.assertEqual(loop.paper.lots, 0)
            self.assertIn("2026-01-02", loop._completed_days)
        finally:
            loop.close()

    def test_session_end_stop_beats_flatten_eod(self):
        """收盤當下若同時觸停損，reason 應為 stop_loss 而非 flatten_eod。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            t0 = pd.Timestamp("2026-01-02 13:40:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            stop_px = fill_entry - 60.0  # 超過 ORB stop 50
            ev = loop.on_tick("2026-01-02 13:45:01", stop_px, 1)
            self.assertEqual(len(ev), 1)
            self.assertEqual(ev[0]["reason"], "stop_loss")
            self.assertEqual(loop.paper.lots, 0)
        finally:
            loop.close()

    def test_position_risk_take_profit_and_slip_direction(self):
        """問題 2 與 TP 覆核：停利平倉 mid = price + slip，實際成交 fill 等於 price。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            t0 = pd.Timestamp("2026-01-02 09:10:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            # 10 秒後價格達到 20205 (+204 點，超過 ORB TP 200 點)
            ev = loop.on_tick("2026-01-02 09:10:10", 20205.0, 1)
            self.assertEqual(len(ev), 1)
            self.assertEqual(ev[0]["reason"], "take_profit")
            self.assertEqual(loop.paper.lots, 0)

            # 驗證滑價方向：mid = 20205 + 1 = 20206；成交 fill = 20206 - 1 = 20205
            self.assertEqual(ev[0]["mid"], 20206.0)
            self.assertEqual(loop.paper.round_trips[-1]["exit"], 20205.0)
        finally:
            loop.close()

    def test_position_risk_breakeven(self):
        """保本煞車覆核：盤中獲利達 +50 點啟動保本，回落至 cost+5 點觸發 breakeven_stop。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            t0 = pd.Timestamp("2026-01-02 09:10:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)  # 20001.0
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            # 10 秒後盤中衝高至 20060 (+59 點，啟動保本)
            ev1 = loop.on_tick("2026-01-02 09:10:10", 20060.0, 1)
            self.assertEqual(len(ev1), 0)
            self.assertEqual(loop.paper.lots, 1)

            # 20 秒後回落至 20004 (<= entry 20001 + 5 = 20006)
            ev2 = loop.on_tick("2026-01-02 09:10:20", 20004.0, 1)
            self.assertEqual(len(ev2), 1)
            self.assertEqual(ev2[0]["reason"], "breakeven_stop")
            self.assertEqual(loop.paper.lots, 0)
        finally:
            loop.close()

    def test_position_risk_rsi_track(self):
        """RSI 腿覆核：TP 40 點，停損 300 點。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            t0 = pd.Timestamp("2026-01-02 09:10:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)  # 20001.0
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "rsi"
            loop._last_risk_check_ts = t0

            # 10 秒後價格達到 20045 (+44 點，超過 RSI TP 40 點)
            ev = loop.on_tick("2026-01-02 09:10:10", 20045.0, 1)
            self.assertEqual(len(ev), 1)
            self.assertEqual(ev[0]["reason"], "take_profit")
            self.assertEqual(loop.paper.lots, 0)
        finally:
            loop.close()

    def test_no_reentry_after_risk_close(self):
        """問題 4 雙通道覆核：風控平倉後標記 completed_days，後續 1m bar 不得重開。"""
        settings = load_settings()
        loop = LiveTickLoop(
            settings,
            cfg=LiveTickConfig(
                date="2026-01-02",
                dry_run=True,
                place_sim=False,
                connect_sim=False,
                risk_check_seconds=10,
            ),
            seed_bars=pd.DataFrame(),
        )
        try:
            t0 = pd.Timestamp("2026-01-02 09:10:00")
            loop.paper.place(TradeAction.BUY, 20000.0, "fake_entry", ts=t0)
            fill_entry = float(loop.paper.entry_price)
            loop._active_entry_price = fill_entry
            loop._active_entry_ts = t0
            loop._active_max_price = fill_entry
            loop._active_track = "orb"
            loop._last_risk_check_ts = t0

            # 10 秒後停損平倉
            ev = loop.on_tick("2026-01-02 09:10:10", 19940.0, 1)
            self.assertEqual(len(ev), 1)
            self.assertEqual(loop.paper.lots, 0)
            self.assertIn("2026-01-02", loop._completed_days)

            # 後續關閉 1m 棒呼叫 _evaluate_unlocked，確認不觸發新進場
            res_trades = loop._evaluate_unlocked(as_of_ts=pd.Timestamp("2026-01-02 09:15:00"))
            self.assertEqual(len(res_trades), 0)
            self.assertEqual(loop.paper.lots, 0)
        finally:
            loop.close()


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLiveTickLoop)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
