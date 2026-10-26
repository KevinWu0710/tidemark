#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析鎖策略虧損日，並搜尋進場過濾／風控，使 IS/OOS 交易日日勝率=100%。
結果 → outputs/no_loss_days_report.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import BacktestResult
from src.backtest.metrics import daily_pnl_stats, pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.core.types import Side
from src.data.bars import load_1m_range


@dataclass
class FilterSpec:
    # minutes from 08:45
    min_entry_mins: int = 0
    max_entry_mins: int = 999
    no_reentry_after_stop: bool = False
    # if True: do not hold loser to eod — close when last N mins of session if unrealized < 0
    cut_loser_before_eod_mins: int = 0
    # require close > ema at entry
    ema_period: int = 0
    # skip if open gap (day_open - prev_close) below this (points)
    min_gap: float | None = None
    max_gap: float | None = None
    # after reclaim, require next bar close > entry bar close (1-bar confirm; no peek beyond)
    confirm_next_bar: bool = False
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_entry_mins": self.min_entry_mins,
            "max_entry_mins": self.max_entry_mins,
            "no_reentry_after_stop": self.no_reentry_after_stop,
            "cut_loser_before_eod_mins": self.cut_loser_before_eod_mins,
            "ema_period": self.ema_period,
            "min_gap": self.min_gap,
            "max_gap": self.max_gap,
            "confirm_next_bar": self.confirm_next_bar,
            "tag": self.tag,
        }


def _run_filtered(
    bars: pd.DataFrame,
    *,
    period: int,
    low_th: float,
    take_profit: float,
    stop_points: float,
    capital: float,
    point_value: float,
    fee: float,
    tax: float,
    slip: float,
    filt: FilterSpec,
) -> BacktestResult:
    if bars is None or bars.empty:
        return BacktestResult(final_equity=capital)

    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0).values
    r0, r1 = rsi, np.roll(rsi, 1)
    r1[0] = np.nan
    raw_entry = (r1 < low_th) & (r0 >= low_th)
    raw_entry[: period + 2] = False

    if filt.ema_period and filt.ema_period > 1:
        ema = close.ewm(span=filt.ema_period, adjust=False).mean().values
    else:
        ema = None

    sess = (
        df["session"].astype(str).values
        if "session" in df.columns
        else np.array(["day"] * len(df))
    )
    in_sess = np.array([s == "day" for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])

    # minutes from session open 08:45
    ts = df["ts"]
    entry_mins = (ts.dt.hour * 60 + ts.dt.minute) - (8 * 60 + 45)

    # day open & gap vs prior day last close（開盤時已知）
    day_keys = ts.dt.strftime("%Y-%m-%d").values
    day_open: dict[str, float] = {}
    day_last_close: dict[str, float] = {}
    for i in range(len(df)):
        dk = day_keys[i]
        if in_sess[i] and dk not in day_open:
            day_open[dk] = float(df["open"].iloc[i])
        if in_sess[i]:
            day_last_close[dk] = float(close.iloc[i])
    ordered_days = sorted(day_open.keys())
    gap_by_day: dict[str, float | None] = {}
    prev_c: float | None = None
    for d in ordered_days:
        gap_by_day[d] = None if prev_c is None else day_open[d] - prev_c
        prev_c = day_last_close.get(d, prev_c)

    close_a = close.values
    ts_a = df["ts"].values
    n = len(df)
    equity = capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    day_realized = 0.0
    stopped_today = False
    pending_confirm = False  # confirm_next_bar: signal bar seen, wait one bar

    def cost(px: float) -> float:
        return fee + px * point_value * tax

    def fill_buy(mid: float) -> float:
        return mid + slip

    def fill_sell(mid: float) -> float:
        return mid - slip

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized, stopped_today
        f = fill_sell(px)
        net = (f - entry) * point_value - cost(f)
        equity += net
        day_realized += net
        trades.append(
            {"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason}
        )
        if reason.startswith("stop_loss"):
            stopped_today = True
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close_a[i])
        now = ts_a[i]
        dk = day_keys[i]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0
            stopped_today = False
            pending_confirm = False

        in_session = bool(in_sess[i])
        eod = bool(last_in_block[i])
        mins = int(entry_mins.iloc[i])

        if position == Side.LONG:
            if stop_points > 0 and (entry - px) >= stop_points:
                close_long(px, now, f"stop_loss_{stop_points}pt")
            elif take_profit > 0 and (px - entry) >= take_profit:
                close_long(px, now, f"take_profit_{take_profit}pt")
            elif (
                filt.cut_loser_before_eod_mins > 0
                and in_session
                and mins >= (300 - filt.cut_loser_before_eod_mins)
                and px < entry
            ):
                # 13:45 = 300 mins from 08:45; cut underwater before eod
                close_long(px, now, "cut_loser_before_eod")

        if in_session and eod and position == Side.LONG:
            close_long(px, now, "flatten_eod")

        # entry logic
        signal = bool(raw_entry[i]) and in_session and not eod
        allow_time = filt.min_entry_mins <= mins <= filt.max_entry_mins
        gap = gap_by_day.get(dk)
        allow_gap = True
        if gap is not None:
            if filt.min_gap is not None and gap < filt.min_gap:
                allow_gap = False
            if filt.max_gap is not None and gap > filt.max_gap:
                allow_gap = False
        allow_ema = True
        if ema is not None:
            allow_ema = px > float(ema[i])
        allow_re = not (filt.no_reentry_after_stop and stopped_today)

        if position == Side.FLAT and in_session and not eod:
            if filt.confirm_next_bar:
                if pending_confirm:
                    # confirm: this bar close > prior close
                    if px > float(close_a[i - 1]) and allow_time and allow_gap and allow_ema and allow_re:
                        f = fill_buy(px)
                        equity -= cost(f)
                        entry = f
                        position = Side.LONG
                        trades.append(
                            {
                                "ts": now,
                                "action": "BUY",
                                "price": f,
                                "reason": "rsi_tp_confirm",
                            }
                        )
                    pending_confirm = False
                elif signal and allow_time and allow_gap and allow_ema and allow_re:
                    pending_confirm = True
            elif signal and allow_time and allow_gap and allow_ema and allow_re:
                f = fill_buy(px)
                equity -= cost(f)
                entry = f
                position = Side.LONG
                trades.append(
                    {"ts": now, "action": "BUY", "price": f, "reason": "rsi_tp_long"}
                )

        mtm = equity + ((px - entry) * point_value if position == Side.LONG else 0.0)
        if i % 5 == 0 or i == n - 1:
            curve.append({"ts": now, "equity": mtm, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=n,
    )


def _loss_day_detail(result: BacktestResult) -> list[dict[str, Any]]:
    trips = pair_round_trips(result.trades)
    by: dict[str, list] = {}
    for t in trips:
        d = str(pd.Timestamp(t["exit_ts"]).date())
        by.setdefault(d, []).append(t)
    out = []
    for d, ts in sorted(by.items()):
        pnl = sum(x["pnl"] for x in ts)
        if pnl <= 0:
            out.append(
                {
                    "day": d,
                    "pnl": round(pnl, 2),
                    "n": len(ts),
                    "exits": [x["reason_exit"] for x in ts],
                    "entries": [str(x["entry_ts"]) for x in ts],
                }
            )
    return out


def _score(result: BacktestResult, capital: float) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    d = daily_pnl_stats(pair_round_trips(result.trades))
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s["win_rate"],
        "n_trade_days": d["n_trade_days"],
        "n_win_days": d["n_win_days"],
        "n_loss_days": d["n_loss_days"],
        "daily_win_rate": d["daily_win_rate"],
        "max_drawdown": s["max_drawdown"],
        "perfect_days": bool(
            d["n_trade_days"] >= 5 and d["n_loss_days"] == 0 and s["net_pnl"] > 0
        ),
    }


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    capital = float(trading["capital"])
    params = next(
        s["params"] for s in settings["strategies"] if s["id"] == "tmf_v1"
    )
    period = int(params.get("rsi_period", 28))
    low = float(params.get("rsi_low", 20))
    tp = float(params.get("take_profit_points", 40))
    stop = float(params.get("stop_points", 300))

    is_bars = load_1m_range("2024-08-01", "2026-03-31")
    oos_bars = load_1m_range("2026-04-01", "2026-10-22")

    baseline = FilterSpec(tag="baseline")
    base_is = _run_filtered(
        is_bars,
        period=period,
        low_th=low,
        take_profit=tp,
        stop_points=stop,
        capital=capital,
        point_value=10,
        fee=20,
        tax=0.00002,
        slip=1,
        filt=baseline,
    )
    base_oos = _run_filtered(
        oos_bars,
        period=period,
        low_th=low,
        take_profit=tp,
        stop_points=stop,
        capital=capital,
        point_value=10,
        fee=20,
        tax=0.00002,
        slip=1,
        filt=baseline,
    )

    # grid
    candidates: list[FilterSpec] = [baseline]
    for max_m in (60, 90, 120, 150, 180, 210, 240):
        candidates.append(
            FilterSpec(max_entry_mins=max_m, tag=f"max_entry_{max_m}")
        )
    for min_m in (5, 10, 15, 30):
        candidates.append(
            FilterSpec(min_entry_mins=min_m, tag=f"min_entry_{min_m}")
        )
    candidates.append(
        FilterSpec(no_reentry_after_stop=True, tag="no_reentry_stop")
    )
    for cut in (15, 30, 60):
        candidates.append(
            FilterSpec(cut_loser_before_eod_mins=cut, tag=f"cut_loser_{cut}")
        )
    for ema in (20, 60, 120):
        candidates.append(FilterSpec(ema_period=ema, tag=f"ema_{ema}"))
    candidates.append(FilterSpec(confirm_next_bar=True, tag="confirm_1bar"))
    for min_gap in (0, -50, -100, -200):
        candidates.append(FilterSpec(min_gap=float(min_gap), tag=f"min_gap_{min_gap}"))

    # combos targeting known loss modes
    for max_m in (90, 120, 180):
        for cut in (0, 30, 60):
            for nore in (True, False):
                candidates.append(
                    FilterSpec(
                        max_entry_mins=max_m,
                        cut_loser_before_eod_mins=cut,
                        no_reentry_after_stop=nore,
                        tag=f"combo_max{max_m}_cut{cut}_nore{int(nore)}",
                    )
                )
    for max_m in (120, 180):
        candidates.append(
            FilterSpec(
                max_entry_mins=max_m,
                confirm_next_bar=True,
                no_reentry_after_stop=True,
                cut_loser_before_eod_mins=30,
                tag=f"combo_confirm_max{max_m}",
            )
        )
    for ema in (20, 60):
        for max_m in (120, 180):
            candidates.append(
                FilterSpec(
                    ema_period=ema,
                    max_entry_mins=max_m,
                    no_reentry_after_stop=True,
                    cut_loser_before_eod_mins=30,
                    tag=f"combo_ema{ema}_max{max_m}",
                )
            )

    # dedupe by tag
    seen = set()
    uniq: list[FilterSpec] = []
    for c in candidates:
        if c.tag in seen:
            continue
        seen.add(c.tag)
        uniq.append(c)

    rows = []
    perfect = []
    report_path = ROOT / "outputs" / "no_loss_days_report.json"
    report_path.write_text(
        json.dumps({"status": "RUNNING", "n": len(uniq)}, indent=2), encoding="utf-8"
    )

    for i, filt in enumerate(uniq):
        is_r = _run_filtered(
            is_bars,
            period=period,
            low_th=low,
            take_profit=tp,
            stop_points=stop,
            capital=capital,
            point_value=10,
            fee=20,
            tax=0.00002,
            slip=1,
            filt=filt,
        )
        oos_r = _run_filtered(
            oos_bars,
            period=period,
            low_th=low,
            take_profit=tp,
            stop_points=stop,
            capital=capital,
            point_value=10,
            fee=20,
            tax=0.00002,
            slip=1,
            filt=filt,
        )
        is_s = _score(is_r, capital)
        oos_s = _score(oos_r, capital)
        row = {
            "filter": filt.to_dict(),
            "IS": is_s,
            "OOS": oos_s,
            "pass_perfect": bool(is_s["perfect_days"] and oos_s["perfect_days"]),
            "pass_no_loss_loose": bool(
                is_s["n_loss_days"] == 0
                and oos_s["n_loss_days"] == 0
                and is_s["net_pnl"] > 0
                and oos_s["net_pnl"] > 0
                and is_s["n_trade_days"] >= 3
                and oos_s["n_trade_days"] >= 3
            ),
        }
        rows.append(row)
        if row["pass_perfect"] or row["pass_no_loss_loose"]:
            perfect.append(row)
        if (i + 1) % 10 == 0:
            logger.info(f"{notify_prefix()} researched {i+1}/{len(uniq)}")

    rows.sort(
        key=lambda r: (
            r["pass_perfect"],
            r["pass_no_loss_loose"],
            (r["OOS"].get("daily_win_rate") or 0),
            (r["IS"].get("daily_win_rate") or 0),
            r["OOS"]["net_pnl"],
        ),
        reverse=True,
    )

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_loss_days": {
            "IS": _loss_day_detail(base_is),
            "OOS": _loss_day_detail(base_oos),
        },
        "baseline_score": {"IS": _score(base_is, capital), "OOS": _score(base_oos, capital)},
        "n_filters": len(uniq),
        "n_perfect": sum(1 for r in rows if r["pass_perfect"]),
        "n_no_loss_loose": sum(1 for r in rows if r["pass_no_loss_loose"]),
        "winners": [r for r in rows if r["pass_perfect"] or r["pass_no_loss_loose"]][:20],
        "top10": rows[:10],
        "note": (
            "perfect=IS&OOS 皆無虧損交易日、淨利>0、交易日>=5；"
            "loose 交易日>=3。過濾皆僅用進場當下可得資訊。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        f"{notify_prefix()} no-loss hunt done perfect={payload['n_perfect']} "
        f"loose={payload['n_no_loss_loose']} -> {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
