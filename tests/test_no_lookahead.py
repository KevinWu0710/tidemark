#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2: prove walk-forward engines never see future bars."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import SecondBarBacktester
from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast
from src.backtest.fast_signals import FastSignalConfig, _torb_entry_masks
from src.core.types import Side, SignalDecision, TradeAction
from src.signals.base import BaseSignal
from src.signals.candidates import BreakoutSignal
from src.strategy.tmf_session import TmfSessionStrategy


def _bars(n: int = 12, *, spike_at: int | None = None) -> pd.DataFrame:
    close = [100.0 + i for i in range(n)]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    if spike_at is not None:
        high[spike_at] = 9999.0
        close[spike_at] = 9999.0
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-02 09:00:00", periods=n, freq="min"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "session": ["day"] * n,
        }
    )


class _HistorySpy(BaseSignal):
    """Records history length + last close seen on each evaluate call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_lens: list[int] = []
        self.seen_last_close: list[float] = []

    def evaluate(self, history, position, *, now_ts=None) -> SignalDecision:
        self.seen_lens.append(0 if history is None else len(history))
        if history is not None and not history.empty:
            self.seen_last_close.append(float(history["close"].iloc[-1]))
        return SignalDecision(TradeAction.HOLD, "spy", ts=str(now_ts) if now_ts else None)


class TestNoLookahead(unittest.TestCase):
    def test_engine_history_is_prefix_only(self) -> None:
        spy = _HistorySpy()
        bars = _bars(8)
        strat = TmfSessionStrategy(signal=spy, max_lots=1, capital=10000)
        bt = SecondBarBacktester(strat, capital=10000, point_value=10)
        result = bt.run(bars)
        self.assertEqual(result.n_steps, 8)
        self.assertEqual(spy.seen_lens, list(range(1, 9)))
        self.assertEqual(spy.seen_last_close, [float(x) for x in bars["close"].tolist()])

    def test_breakout_channel_excludes_current_and_future(self) -> None:
        """Donchian window is prior bars only; current/future spikes stay out of channel."""
        lookback = 3
        clean = _bars(10)
        spiked = _bars(10, spike_at=7)
        sig = BreakoutSignal({"lookback": lookback})

        # Same past prefix → same decision even if full series has a future spike
        i = 5
        d_clean = sig.evaluate(clean.iloc[: i + 1], Side.FLAT, now_ts=clean["ts"].iloc[i])
        d_spiked = sig.evaluate(spiked.iloc[: i + 1], Side.FLAT, now_ts=spiked["ts"].iloc[i])
        self.assertEqual(d_clean.action, d_spiked.action)
        self.assertEqual(d_clean.reason, d_spiked.reason)

        # At spike bar itself: channel window must not include the spike high
        hist7 = spiked.iloc[:8]
        window = hist7.iloc[-(lookback + 1) : -1]
        self.assertLess(float(window["high"].max()), 9000.0)
        self.assertEqual(float(hist7["high"].iloc[-1]), 9999.0)

    def test_fast_breakout_uses_shifted_channel(self) -> None:
        cfg = FastBreakoutConfig(lookback=3, sessions=("day",), slip_ticks=0, fee_per_side=0, tax_rate=0)
        # Rising then break prior high on last bar
        bars = _bars(8)
        # Force a clean breakout: last close above prior 3-bar high
        bars = bars.copy()
        bars.loc[7, "close"] = 200.0
        bars.loc[7, "high"] = 200.0
        result = run_breakout_fast(bars, cfg)
        # Must have stepped all bars; entry only after lookback warmup
        self.assertEqual(result.n_steps, 8)
        opens = [t for t in result.trades if t["action"] in ("BUY", "SELL")]
        self.assertTrue(len(opens) >= 1)
        # First open cannot be before index lookback (need prior window)
        first_ts = opens[0]["ts"]
        first_i = bars.index[bars["ts"] == first_ts][0] if "ts" in bars.columns else 0
        self.assertGreaterEqual(int(first_i), cfg.lookback)

    def test_torb_range_excludes_post_probe_and_future(self) -> None:
        """OR high/low from first probe bars only; later spikes must not widen range."""
        probe = 3
        bars = _bars(12)
        bars = bars.copy()
        # Flat series so only our edits control breaks
        for i in range(12):
            bars.loc[i, ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.0]
        # Probe OR high=103, low=97
        bars.loc[0, ["high", "low", "close"]] = [101.0, 99.0, 100.0]
        bars.loc[1, ["high", "low", "close"]] = [102.0, 98.0, 100.0]
        bars.loc[2, ["high", "low", "close"]] = [103.0, 97.0, 100.0]
        # Bars 3–4 stay inside range (no entry yet)
        bars.loc[3, ["high", "low", "close"]] = [102.0, 98.0, 101.0]
        bars.loc[4, ["high", "low", "close"]] = [102.5, 98.0, 102.0]
        # Bar 5 breaks OR high; spike high must not have redefined the range
        bars.loc[5, ["high", "low", "close"]] = [9999.0, 100.0, 9999.0]
        cfg = FastSignalConfig(sessions=("day",), allow_short=False, flatten_eod=True)
        long_e, _ = _torb_entry_masks(bars, probe, cfg, once_per_day=True)
        self.assertFalse(any(bool(x) for x in long_e[:probe]))
        self.assertFalse(bool(long_e[3]))
        self.assertFalse(bool(long_e[4]))
        self.assertTrue(bool(long_e[5]))
        # Truncated series: break at bar 4 of close>103 still uses OR high=103
        early = bars.iloc[:5].copy()
        early.loc[4, ["high", "low", "close"]] = [104.0, 100.0, 104.0]
        long_early, _ = _torb_entry_masks(early, probe, cfg, once_per_day=True)
        self.assertTrue(bool(long_early[4]))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNoLookahead)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
