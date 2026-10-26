#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只做多：現役 RSI28 鎖 + 事前監控／過濾掃描。

過濾掛在進場 mask（strategy／director 層概念），不改價訊核心。
候選來自：虧損局式研究、台指夜盤／跳空實務、指數期貨開盤雜訊慣例。
（「美股大跌」以夜盤大跌／跳空低開代理，本輪不拉外盤 API。）

過關：IS>0 OOS>0 且日勝率皆 > --min-daily-win-rate（預設 0.8，對齊鎖）。
另報 delta vs baseline。結果 → outputs/long_prefilters_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import BacktestResult
from src.backtest.fast_signals import FastSignalConfig, _long_only_tp_loop, _rsi_series
from src.backtest.metrics import passes_session_daily_win_rate, summarize
from src.core.config import load_settings, notify_prefix
from src.core.types import Side
from src.data.bars import load_1m_range


DAY_OPEN_MIN = 8 * 60 + 45  # 08:45


def _cfg(trading: dict, slip: float) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=("day",),
        stop_points=300.0,
        max_daily_loss=0.0,
        flatten_eod=True,
        allow_short=False,
    )


def _score(result: BacktestResult, capital: float) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    day = (s.get("by_session") or {}).get("day") or {}
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s.get("day_win_rate"),
        "n_trade_days": day.get("n_trade_days") or 0,
        "n_win_days": day.get("n_win_days") or 0,
        "n_loss_days": day.get("n_loss_days") or 0,
        "daily_win_rate": s.get("day_daily_win_rate"),  # 日盤（本腳本只做日盤）
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "by_session": s.get("by_session"),
        "max_drawdown": s["max_drawdown"],
    }


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "session" not in df.columns:
        from src.data.sessions import label_session

        df["session"] = df["ts"].map(label_session)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["mins"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - DAY_OPEN_MIN
    close = df["close"].astype(float)
    df["rsi"] = _rsi_series(close, 28)
    r1 = np.roll(df["rsi"].values, 1)
    r1[0] = np.nan
    df["rsi_long_raw"] = (r1 < 20.0) & (df["rsi"].values >= 20.0)
    df.loc[: 28 + 1, "rsi_long_raw"] = False

    # per calendar date session stats (no future within day for open)
    day = df[df["session"] == "day"]
    night = df[df["session"] == "night"]

    day_open = day.groupby("date")["open"].first()
    day_close = day.groupby("date")["close"].last()
    day_high = day.groupby("date")["high"].max()
    day_low = day.groupby("date")["low"].min()

    # prior night = chronologically between prev day 13:45 and today 08:45
    night_ret: dict[str, float] = {}
    gap_pts: dict[str, float] = {}
    dates = sorted(day_open.index.tolist())
    for i, d in enumerate(dates):
        d_open_ts = day.loc[day["date"] == d, "ts"].iloc[0]
        if i == 0:
            # night bars before first day open same calendar
            nseg = night[night["ts"] < d_open_ts]
        else:
            prev = dates[i - 1]
            prev_day_last = day.loc[day["date"] == prev, "ts"].iloc[-1]
            nseg = night[(night["ts"] > prev_day_last) & (night["ts"] < d_open_ts)]
        if len(nseg) >= 2:
            night_ret[d] = float(nseg["close"].iloc[-1] - nseg["open"].iloc[0])
        else:
            night_ret[d] = 0.0
        if i == 0:
            gap_pts[d] = 0.0
        else:
            gap_pts[d] = float(day_open[d] - day_close[dates[i - 1]])

    # daily ATR (Wilder-ish simple TR) from completed day sessions; shift 1
    tr = (day_high - day_low).astype(float)
    # include gap component vs prior close
    atr_raw = {}
    for i, d in enumerate(dates):
        if i == 0:
            atr_raw[d] = float(tr[d])
        else:
            pc = float(day_close[dates[i - 1]])
            atr_raw[d] = float(
                max(tr[d], abs(day_high[d] - pc), abs(day_low[d] - pc))
            )
    atr_series = pd.Series(atr_raw).sort_index()
    atr14 = atr_series.rolling(14, min_periods=5).mean().shift(1)  # prior ATR only

    df["day_open"] = df["date"].map(day_open)
    df["night_ret"] = df["date"].map(night_ret).fillna(0.0)
    df["gap_pts"] = df["date"].map(gap_pts).fillna(0.0)
    df["atr14_prev"] = df["date"].map(atr14).fillna(np.nan)

    # session VWAP from day open only (causal)
    in_day = df["session"] == "day"
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float).clip(lower=0.0)
    pv = tp * vol
    df["vwap"] = np.nan
    for d, idx in df.loc[in_day].groupby("date").groups.items():
        ix = list(idx)
        c_pv = np.cumsum(pv.iloc[ix].values)
        c_v = np.cumsum(vol.iloc[ix].values)
        vwap = np.where(c_v > 0, c_pv / c_v, np.nan)
        df.loc[ix, "vwap"] = vwap

    return df


def _base_entry(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["session"].astype(str) == "day")
        & df["rsi_long_raw"].astype(bool)
    ).values


def _run_masked(df: pd.DataFrame, entry: np.ndarray, cfg: FastSignalConfig) -> BacktestResult:
    return _long_only_tp_loop(
        df,
        cfg,
        entry.astype(bool),
        take_profit_points=40.0,
        reason_entry="rsi_prefilter",
    )


def _run_no_reentry_after_stop(
    df: pd.DataFrame, entry: np.ndarray, cfg: FastSignalConfig
) -> BacktestResult:
    """當日若曾 stop_loss，禁止再進多。"""
    close_a = df["close"].astype(float).values
    ts = df["ts"].values
    sess = df["session"].astype(str).values
    dates = df["date"].values
    n = len(df)
    in_sess = sess == "day"
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])
    equity = cfg.capital
    position = Side.FLAT
    entry_px = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    stopped_today = False
    tp, stop = 40.0, float(cfg.stop_points)

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        dk = str(dates[i])
        if dk != day_key:
            day_key = dk
            stopped_today = False
        if position == Side.LONG:
            if stop > 0 and (entry_px - px) >= stop:
                f = px - cfg.slip_ticks
                net = (f - entry_px) * cfg.point_value - cost(f)
                equity += net
                trades.append(
                    {"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "stop_loss"}
                )
                position = Side.FLAT
                stopped_today = True
            elif (px - entry_px) >= tp:
                f = px - cfg.slip_ticks
                net = (f - entry_px) * cfg.point_value - cost(f)
                equity += net
                trades.append(
                    {
                        "ts": now,
                        "action": "CLOSE",
                        "price": f,
                        "pnl": net,
                        "reason": "take_profit",
                    }
                )
                position = Side.FLAT
        eod = bool(cfg.flatten_eod and last_in_block[i])
        if in_sess[i] and eod and position == Side.LONG:
            f = px - cfg.slip_ticks
            net = (f - entry_px) * cfg.point_value - cost(f)
            equity += net
            trades.append(
                {"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "flatten_eod"}
            )
            position = Side.FLAT
        if (
            in_sess[i]
            and not eod
            and position == Side.FLAT
            and not stopped_today
            and bool(entry[i])
        ):
            f = px + cfg.slip_ticks
            equity -= cost(f)
            entry_px = f
            position = Side.LONG
            trades.append({"ts": now, "action": "BUY", "price": f, "reason": "rsi_no_reentry"})
        mtm = equity + (
            (px - entry_px) * cfg.point_value if position == Side.LONG else 0.0
        )
        curve.append({"ts": now, "equity": mtm, "price": px})
    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=len(curve),
    )


FilterFn = Callable[[pd.DataFrame, np.ndarray], np.ndarray]


def _filter_catalog() -> list[dict[str, Any]]:
    """可回測、無偷看的事前過濾選單（含搜尋／實務常見項）。"""
    cats: list[dict[str, Any]] = []

    def add(fid: str, family: str, desc: str, fn: FilterFn, **params: Any) -> None:
        cats.append(
            {"id": fid, "family": family, "desc": desc, "params": params, "fn": fn}
        )

    add("baseline", "control", "無過濾（現役鎖）", lambda df, e: e)

    # --- 無條件開盤禁制（對照；先前 15m OOS 曾掛）---
    for m in (5, 15, 30, 45, 60):
        add(
            f"skip_open_{m}m",
            "clock_uncond",
            f"開盤後 {m} 分鐘內不進（無條件）",
            lambda df, e, mm=m: e & (df["mins"].values >= mm),
            skip_mins=m,
        )

    # --- 條件式：夜盤大跌 → 跳過早盤（你的例子的代理）---
    for thr in (-80.0, -120.0, -160.0, -200.0, -300.0):
        for m in (30, 60):
            add(
                f"night_down_{int(-thr)}_skip_{m}m",
                "overnight_proxy",
                f"前一夜盤跌超過 {-thr:.0f} 點 → 開盤 {m} 分不進",
                lambda df, e, t=thr, mm=m: e
                & ~((df["night_ret"].values <= t) & (df["mins"].values < mm)),
                night_ret_le=thr,
                skip_mins=m,
            )

    # --- 條件式：跳空低開 → 跳過早盤／整日否決 ---
    for thr in (-80.0, -120.0, -160.0, -200.0):
        for m in (30, 60):
            add(
                f"gap_down_{int(-thr)}_skip_{m}m",
                "gap",
                f"跳空低開超過 {-thr:.0f} 點 → 開盤 {m} 分不進",
                lambda df, e, t=thr, mm=m: e
                & ~((df["gap_pts"].values <= t) & (df["mins"].values < mm)),
                gap_le=thr,
                skip_mins=m,
            )
    for mult in (0.6, 0.8, 1.0):
        add(
            f"gap_down_atr{mult}_veto_day",
            "gap",
            f"向下跳空 ≥ {mult}×前日ATR14 → 全日不做多（台指常見硬否決）",
            lambda df, e, m=mult: e
            & ~(
                np.isfinite(df["atr14_prev"].values)
                & (df["gap_pts"].values <= -m * df["atr14_prev"].values)
            ),
            atr_mult=mult,
        )

    # --- 夜盤大跌 → 全日觀望 ---
    for thr in (-160.0, -200.0, -300.0):
        add(
            f"night_down_{int(-thr)}_flat_day",
            "overnight_proxy",
            f"前一夜盤跌超過 {-thr:.0f} 點 → 全日不做多",
            lambda df, e, t=thr: e & (df["night_ret"].values > t),
            night_ret_le=thr,
        )

    # --- 時段：禁尾盤（局式 B 類）---
    for cutoff in (150, 180, 210, 240):  # mins from 08:45 → 11:15, 11:45, 12:15, 12:45
        hh = 8 + (45 + cutoff) // 60
        mm = (45 + cutoff) % 60
        add(
            f"no_late_before_{hh:02d}{mm:02d}",
            "clock_late",
            f"僅 {hh:02d}:{mm:02d} 前允許進場",
            lambda df, e, c=cutoff: e & (df["mins"].values <= c) & (df["mins"].values >= 0),
            max_mins=cutoff,
        )

    # --- 路徑確認 ---
    def next_bar_confirm(df: pd.DataFrame, e: np.ndarray) -> np.ndarray:
        raw = e.astype(bool)
        close = df["close"].astype(float).values
        out = np.zeros(len(df), dtype=bool)
        pending = False
        for i in range(len(df)):
            if pending:
                if close[i] > close[i - 1]:
                    out[i] = True
                pending = False
            if raw[i]:
                pending = True
        return out

    add(
        "next_bar_confirm",
        "confirm",
        "RSI reclaim 後下一根收漲才進",
        next_bar_confirm,
    )

    add(
        "above_day_open",
        "structure",
        "收盤價須高於當日開盤才允許做多",
        lambda df, e: e
        & np.isfinite(df["day_open"].values)
        & (df["close"].astype(float).values > df["day_open"].values),
    )

    add(
        "above_day_vwap",
        "structure",
        "收盤價須高於當日盤中 VWAP（因果）",
        lambda df, e: e
        & np.isfinite(df["vwap"].values)
        & (df["close"].astype(float).values > df["vwap"].values),
    )

    # --- 組合：夜盤條件跳過 + 禁尾盤 ---
    add(
        "night120_skip60_and_no_late_1115",
        "combo",
        "夜盤≤-120 則跳過 60 分 + 11:15 前才進",
        lambda df, e: e
        & ~((df["night_ret"].values <= -120.0) & (df["mins"].values < 60))
        & (df["mins"].values <= 150)
        & (df["mins"].values >= 0),
    )
    add(
        "gap120_skip60_and_confirm",
        "combo",
        "跳空≤-120 則跳過 60 分 + 下一根確認",
        lambda df, e: next_bar_confirm(
            df,
            e
            & ~((df["gap_pts"].values <= -120.0) & (df["mins"].values < 60)),
        ),
    )

    # special runner flag
    add(
        "no_reentry_after_stop",
        "risk_state",
        "當日曾停損後不再進多",
        lambda df, e: e,  # mask 同 baseline；runner 不同
        special="no_reentry_after_stop",
    )

    return cats


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark long-only prefilter research")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-daily-win-rate", type=float, default=0.8)
    args = p.parse_args()
    min_dwr = float(args.min_daily_win_rate)

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    cfg = _cfg(trading, slip)

    is_raw = load_1m_range(args.is_start, args.is_end)
    oos_raw = load_1m_range(args.oos_start, args.oos_end)
    is_df = _prepare(is_raw)
    oos_df = _prepare(oos_raw)

    catalog = _filter_catalog()
    report_path = ROOT / "outputs" / "long_prefilters_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_filters": len(catalog),
                "min_daily_win_rate": min_dwr,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} long-prefilters n={len(catalog)} min_dwr={min_dwr}"
    )

    sources = {
        "loss_regimes": ["skip_open", "no_late", "next_bar_confirm"],
        "user_example_proxy": ["night_down_*_skip_*", "美股大跌→夜盤代理"],
        "tw_futures_practice": [
            "gap vs ATR veto",
            "night session → day open",
            "VWAP / day open structure",
        ],
        "index_futures_practice": [
            "skip first 5–15m uncond (noise)",
            "gap filters",
            "entry window / no late",
        ],
    }

    rows: list[dict[str, Any]] = []
    baseline_is = baseline_oos = None

    for i, spec in enumerate(catalog):
        fn: FilterFn = spec["fn"]
        is_e = fn(is_df, _base_entry(is_df))
        oos_e = fn(oos_df, _base_entry(oos_df))
        special = (spec.get("params") or {}).get("special")
        if special == "no_reentry_after_stop":
            is_r = _run_no_reentry_after_stop(is_df, is_e, cfg)
            oos_r = _run_no_reentry_after_stop(oos_df, oos_e, cfg)
        else:
            is_r = _run_masked(is_df, is_e, cfg)
            oos_r = _run_masked(oos_df, oos_e, cfg)
        is_s = _score(is_r, capital)
        oos_s = _score(oos_r, capital)
        if spec["id"] == "baseline":
            baseline_is, baseline_oos = is_s, oos_s
        passed = bool(
            is_s["net_pnl"] > 0
            and oos_s["net_pnl"] > 0
            and passes_session_daily_win_rate(
                is_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            )
            and passes_session_daily_win_rate(
                oos_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            )
            and is_s["n_round_trips"] >= 20
            and oos_s["n_round_trips"] >= 8
        )
        row = {
            "id": spec["id"],
            "family": spec["family"],
            "desc": spec["desc"],
            "params": {k: v for k, v in (spec.get("params") or {}).items() if k != "special"},
            "IS": is_s,
            "OOS": oos_s,
            "pass": passed,
        }
        rows.append(row)
        if (i + 1) % 15 == 0 or i + 1 == len(catalog):
            logger.info(f"{notify_prefix()} long-prefilters {i+1}/{len(catalog)}")

    assert baseline_is is not None and baseline_oos is not None
    for r in rows:
        r["delta_IS_pnl"] = round(r["IS"]["net_pnl"] - baseline_is["net_pnl"], 2)
        r["delta_OOS_pnl"] = round(r["OOS"]["net_pnl"] - baseline_oos["net_pnl"], 2)
        r["delta_IS_dwr"] = round(
            float(r["IS"]["daily_win_rate"] or 0) - float(baseline_is["daily_win_rate"] or 0),
            4,
        )
        r["delta_OOS_dwr"] = round(
            float(r["OOS"]["daily_win_rate"] or 0)
            - float(baseline_oos["daily_win_rate"] or 0),
            4,
        )
        r["delta_OOS_dd"] = round(
            float(baseline_oos["max_drawdown"] or 0) - float(r["OOS"]["max_drawdown"] or 0),
            2,
        )  # positive = smaller DD than baseline

    winners = [r for r in rows if r["pass"]]
    # rank: pass first, then OOS pnl lift, then dwr lift, then DD improvement
    rows.sort(
        key=lambda r: (
            r["pass"],
            r["delta_OOS_pnl"],
            r["delta_OOS_dwr"],
            r["delta_OOS_dd"],
            r["OOS"]["net_pnl"],
        ),
        reverse=True,
    )
    winners.sort(
        key=lambda r: (r["delta_OOS_pnl"], r["delta_OOS_dwr"], r["delta_OOS_dd"]),
        reverse=True,
    )

    by_family: dict[str, Any] = {}
    for r in rows:
        by_family.setdefault(r["family"], {"n": 0, "n_pass": 0, "best": None})
        by_family[r["family"]]["n"] += 1
        if r["pass"]:
            by_family[r["family"]]["n_pass"] += 1
            if by_family[r["family"]]["best"] is None:
                by_family[r["family"]]["best"] = r

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "goal": "long_only_pretrade_monitors_improve_wr_or_pnl",
        "baseline_signal": "rsi28_reclaim20_tp40_stop300_day",
        "min_daily_win_rate": min_dwr,
        "pass_rule": (
            f"IS>0 OOS>0 daily_win_rate>{min_dwr} "
            "IS_trades>=20 OOS_trades>=8"
        ),
        "n_filters": len(catalog),
        "n_pass": len(winners),
        "idea_sources": sources,
        "baseline": {"IS": baseline_is, "OOS": baseline_oos},
        "by_family": by_family,
        "winners_top15": winners[:15],
        "top15_all": rows[:15],
        "note": (
            "過濾屬事前監控；美股以夜盤／跳空代理。"
            "過關且明顯優於 baseline 才考慮寫進 strategy director；不改 tmf_v1 價訊。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} long-prefilters done pass={len(winners)}/{len(catalog)} "
        f"-> {report_path}"
    )
    if winners:
        b = winners[0]
        logger.info(
            f"  BEST {b['id']} ΔOOS={b['delta_OOS_pnl']} Δdwr={b['delta_OOS_dwr']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']}"
        )
    return 0 if winners else 2


if __name__ == "__main__":
    raise SystemExit(main())
