#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for OpenGapOrbSignal and time_stop in TmfSessionStrategy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.types import Side, SignalDecision, TradeAction
from src.signals.open_gap_orb import OpenGapOrbSignal
from src.strategy.tmf_session import TmfSessionStrategy


class TestOpenGapOrb(unittest.TestCase):
    def _create_mock_history(self, n_bars: int = 30, *, gap: float = 50.0, fill_gap: bool = False, break_high: bool = True) -> pd.DataFrame:
        # 建立前一日 1 根 (收在 100.0) + 今日日盤 08:45 開始的 K 棒
        rows = [
            {"ts": pd.Timestamp("2026-01-01 13:45:00"), "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "session": "day"}
        ]
        open_px = 100.0 + gap  # 跳空
        for i in range(n_bars):
            cur_ts = pd.Timestamp("2026-01-02 08:45:00") + pd.Timedelta(minutes=i)
            # 前 15 根 (08:45-09:00): probe
            if i < 15:
                # low 是否回補 100.0
                low_px = 99.0 if fill_gap else (open_px - 10.0)
                high_px = open_px + 20.0
                close_px = open_px + 5.0
            else:
                # 09:00 之後
                or_high = open_px + 20.0
                if i == 18 and break_high:
                    close_px = or_high + 5.0  # 突破
                    high_px = or_high + 10.0
                    low_px = or_high - 5.0
                else:
                    close_px = open_px + 10.0
                    high_px = open_px + 15.0
                    low_px = open_px + 5.0
            rows.append({
                "ts": cur_ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "session": "day",
            })
        return pd.DataFrame(rows)

    def test_probe_period_holds(self):
        sig = OpenGapOrbSignal({"probe_mins": 15, "gap_th": 40.0})
        # 只有 10 根今日 K 棒
        hist = self._create_mock_history(10)
        dec = sig.evaluate(hist, Side.FLAT, now_ts=hist.iloc[-1]["ts"])
        self.assertEqual(dec.action, TradeAction.HOLD)
        self.assertEqual(dec.reason, "probing")

    def test_gap_filled_does_not_buy(self):
        sig = OpenGapOrbSignal({"probe_mins": 15, "gap_th": 40.0})
        # 缺口回補 (fill_gap=True)
        hist = self._create_mock_history(20, fill_gap=True, break_high=True)
        dec = sig.evaluate(hist, Side.FLAT, now_ts=hist.iloc[-1]["ts"])
        self.assertEqual(dec.action, TradeAction.HOLD)

    def test_gap_unfilled_breakout_buys(self):
        sig = OpenGapOrbSignal({"probe_mins": 15, "gap_th": 40.0})
        # 缺口未補且在第 18 根突破 OR_High
        hist = self._create_mock_history(19, fill_gap=False, break_high=True)
        dec = sig.evaluate(hist, Side.FLAT, now_ts=hist.iloc[-1]["ts"])
        self.assertEqual(dec.action, TradeAction.BUY)
        self.assertIn("open_gap_orb long break", dec.reason)

    def test_time_stop_in_strategy(self):
        sig = OpenGapOrbSignal({"probe_mins": 15})
        strat = TmfSessionStrategy(
            signal=sig,
            time_stop_bars=5,
            stop_points=50,
            take_profit_points=150,
        )
        hist = self._create_mock_history(25)
        # 模擬成交進場
        strat.apply_fill(TradeAction.BUY, lots=1, price=150.0)
        # 持有 4 根
        for _ in range(4):
            dec = strat.on_bar(hist, now_ts=hist.iloc[-1]["ts"])
            self.assertEqual(dec.action, TradeAction.HOLD)
        # 第 5 根觸發時間停損
        dec = strat.on_bar(hist, now_ts=hist.iloc[-1]["ts"])
        self.assertEqual(dec.action, TradeAction.CLOSE)
        self.assertIn("time_stop_5bars", dec.reason)

    def test_breakeven_in_strategy(self):
        sig = OpenGapOrbSignal({"probe_mins": 15})
        strat = TmfSessionStrategy(
            signal=sig,
            stop_points=50,
            take_profit_points=200,
            breakeven_points=50,
        )
        hist = self._create_mock_history(25)
        # 100 進場
        strat.apply_fill(TradeAction.BUY, lots=1, price=100.0)
        # 最高價摸到 155 (gain=55 >= be_pts=50)
        hist_peak = hist.copy()
        hist_peak.iloc[-1, hist_peak.columns.get_loc("high")] = 155.0
        hist_peak.iloc[-1, hist_peak.columns.get_loc("close")] = 140.0
        strat.on_bar(hist_peak, now_ts=hist_peak.iloc[-1]["ts"])
        self.assertEqual(strat.highest_price, 155.0)

        # 隨後價格跌回 103 (小於成本+5=105，但未達原本 100-50=50 的大停損)
        hist_drop = hist.copy()
        hist_drop.iloc[-1, hist_drop.columns.get_loc("close")] = 103.0
        dec = strat.on_bar(hist_drop, now_ts=hist_drop.iloc[-1]["ts"])
        self.assertEqual(dec.action, TradeAction.CLOSE)
        self.assertIn("breakeven_50.0pt", dec.reason)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOpenGapOrb)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
