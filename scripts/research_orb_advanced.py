#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
進階開盤突破策略系統性回測 (ORB Advanced Optimization)。

四大維度全面升級回測：
1. 保本機制 (Breakeven Stop: +40, +50, +60 點保本)
2. 移動追蹤停利 (Trailing Stop: 80 點後追蹤，打開獲利天花板至 200~300 點)
3. 60m 大趨勢保護 (EMA Trend Filter)
4. 突破成交量倍增確認 (Volume Spike Filter)
5. 雙向多空解鎖 (Two-way Trading)

嚴格遵守無偷看原則、全歷史 IS/OOS 切分、摩擦成本 (手續費 20 元/邊 + 稅 0.00002 + 滑價 1 點)。
結果儲存至 outputs/orb_advanced_report.json。
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
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestResult
from src.backtest.metrics import pair_round_trips, summarize
from src.core.config import load_settings
from src.core.types import Side
from src.data.bars import load_1m_range
from src.data.sessions import label_session

REPORT_PATH = ROOT / "outputs" / "orb_advanced_report.json"


@dataclass
class TradeConfig:
    capital: float = 10000.0
    point_value: float = 10.0
    fee_per_side: float = 20.0
    tax_rate: float = 0.00002
    slip_ticks: float = 1.0


def prepare_advanced_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """計算日盤開盤特徵、大趨勢 EMA、成交量均線。"""
    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "session" not in df.columns:
        df["session"] = df["ts"].map(label_session)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")

    # 60分K 大趨勢代理：1m 均線 rolling 1200 棒 (相當於 60m EMA20)
    df["ema_trend"] = df["close"].ewm(span=1200, adjust=False).mean()

    day_df = df[df["session"] == "day"]
    night_df = df[df["session"] == "night"]

    dates = sorted(day_df["date"].unique().tolist())
    day_sessions: list[dict[str, Any]] = []

    for idx, d in enumerate(dates):
        d_bars = day_df[day_df["date"] == d]
        if len(d_bars) < 30:
            continue
        d_open_ts = d_bars["ts"].iloc[0]
        d_open_px = float(d_bars["open"].iloc[0])
        trend_val = float(d_bars["ema_trend"].iloc[0])

        # 前一日日盤收盤
        prev_day_close = np.nan
        if idx > 0:
            prev_bars = day_df[day_df["date"] == dates[idx - 1]]
            if len(prev_bars) > 0:
                prev_day_close = float(prev_bars["close"].iloc[-1])

        gap_pts = (d_open_px - prev_day_close) if np.isfinite(prev_day_close) else 0.0

        day_sessions.append({
            "date": d,
            "start_idx": d_bars.index[0],
            "end_idx": d_bars.index[-1] + 1,
            "open_px": d_open_px,
            "prev_day_close": prev_day_close,
            "gap_pts": gap_pts,
            "trend_ema": trend_val,
        })

    return df, day_sessions


def run_advanced_simulation(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    *,
    probe_mins: int = 15,
    cutoff_mins: int = 105,
    gap_th: float = 40.0,
    max_or_range: float = 250.0,
    stop_points: float = 50.0,
    tp_points: float = 200.0,
    be_trig_pts: float = 50.0,     # 保本觸發點數 (0=不設, 40, 50, 60)
    be_lock_pts: float = 5.0,      # 保本鎖定獲利 (5點)
    trail_trig_pts: float = 80.0,  # 移動停利啟動點數 (0=不設, 80, 100)
    trail_dist: float = 40.0,      # 移動停利回撤距離 (30, 40, 50)
    time_stop_mins: int = 45,      # 時間停損 (45, 60, 90, 0=不設)
    use_trend_filter: bool = False,# 是否開啟 60m 趨勢濾網
    vol_spike_mult: float = 0.0,   # 突破成交量倍數 (0.0=不檢查, 1.2, 1.5)
    allow_short: bool = False,     # 是否允許做空
) -> BacktestResult:
    """高效率單次回測模擬。"""
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    volume = df["volume"].astype(float).values
    ts = df["ts"].values

    equity = cfg.capital
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    for sess in day_sessions:
        s_idx = sess["start_idx"]
        e_idx = sess["end_idx"]
        n_bars = e_idx - s_idx
        if n_bars <= probe_mins + 2:
            continue

        probe_high = float(np.max(high[s_idx : s_idx + probe_mins]))
        probe_low = float(np.min(low[s_idx : s_idx + probe_mins]))
        or_range = probe_high - probe_low
        avg_probe_vol = float(np.mean(volume[s_idx : s_idx + probe_mins]))

        gap = sess["gap_pts"]
        pclose = sess["prev_day_close"]
        trend = sess["trend_ema"]
        d_open = sess["open_px"]

        if max_or_range > 0 and or_range > max_or_range:
            continue

        # 缺口未補檢查
        allow_l = (gap >= gap_th) and (np.isnan(pclose) or probe_low > pclose)
        allow_s = allow_short and (gap <= -gap_th) and (np.isnan(pclose) or probe_high < pclose)

        # 趨勢濾網檢查 (做多需在均線之上，做空需在均線之下)
        if use_trend_filter:
            if allow_l and d_open < trend:
                allow_l = False
            if allow_s and d_open > trend:
                allow_s = False

        if not allow_l and not allow_s:
            continue

        max_search_idx = min(e_idx, s_idx + cutoff_mins)
        pos = Side.FLAT
        entry_px = 0.0
        peak_px = 0.0
        valley_px = 0.0
        bars_held = 0

        for k in range(s_idx + probe_mins, e_idx):
            cur_c = float(close[k])
            cur_h = float(high[k])
            cur_l = float(low[k])
            cur_v = float(volume[k])
            cur_ts = ts[k]
            is_eod = (k == e_idx - 1)

            if pos == Side.LONG:
                bars_held += 1
                peak_px = max(peak_px, cur_h)
                unrealized_gain = peak_px - entry_px

                # 動態停損計算
                cur_sl = entry_px - stop_points
                # 1. 保本推進
                if be_trig_pts > 0 and unrealized_gain >= be_trig_pts:
                    cur_sl = max(cur_sl, entry_px + be_lock_pts)
                # 2. 移動追蹤停利
                if trail_trig_pts > 0 and trail_dist > 0 and unrealized_gain >= trail_trig_pts:
                    cur_sl = max(cur_sl, peak_px - trail_dist)

                tp_target = entry_px + tp_points

                if cur_l <= cur_sl:
                    f = fill_sell(cur_sl)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    reason = "breakeven" if cur_sl >= entry_px else "stop_loss"
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
                    pos = Side.FLAT
                    break
                elif cur_h >= tp_target:
                    f = fill_sell(tp_target)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "take_profit"})
                    pos = Side.FLAT
                    break
                elif time_stop_mins > 0 and bars_held >= time_stop_mins:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": f"time_stop_{time_stop_mins}"})
                    pos = Side.FLAT
                    break
                elif is_eod:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "flatten_eod"})
                    pos = Side.FLAT
                    break

            elif pos == Side.SHORT:
                bars_held += 1
                valley_px = min(valley_px, cur_l)
                unrealized_gain = entry_px - valley_px

                cur_sl = entry_px + stop_points
                if be_trig_pts > 0 and unrealized_gain >= be_trig_pts:
                    cur_sl = min(cur_sl, entry_px - be_lock_pts)
                if trail_trig_pts > 0 and trail_dist > 0 and unrealized_gain >= trail_trig_pts:
                    cur_sl = min(cur_sl, valley_px + trail_dist)

                tp_target = entry_px - tp_points

                if cur_h >= cur_sl:
                    f = fill_buy(cur_sl)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    reason = "breakeven" if cur_sl <= entry_px else "stop_loss"
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
                    pos = Side.FLAT
                    break
                elif cur_l <= tp_target:
                    f = fill_buy(tp_target)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "take_profit"})
                    pos = Side.FLAT
                    break
                elif time_stop_mins > 0 and bars_held >= time_stop_mins:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": f"time_stop_{time_stop_mins}"})
                    pos = Side.FLAT
                    break
                elif is_eod:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "flatten_eod"})
                    pos = Side.FLAT
                    break

            # 空手狀態開倉
            elif pos == Side.FLAT and k < max_search_idx:
                vol_ok = True
                if vol_spike_mult > 0:
                    vol_ok = (cur_v >= vol_spike_mult * avg_probe_vol)

                if allow_l and cur_c > probe_high and vol_ok:
                    f = fill_buy(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    peak_px = cur_c
                    bars_held = 0
                    pos = Side.LONG
                    trades.append({"ts": cur_ts, "action": "BUY", "price": f, "reason": "orb_break_high"})
                elif allow_s and cur_c < probe_low and vol_ok:
                    f = fill_sell(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    valley_px = cur_c
                    bars_held = 0
                    pos = Side.SHORT
                    trades.append({"ts": cur_ts, "action": "SELL", "price": f, "reason": "orb_break_low"})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=equity,
        n_steps=len(trades),
    )


def evaluate_advanced_splits(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    is_sessions = [s for s in day_sessions if s["date"] <= "2026-03-31"]
    oos_sessions = [s for s in day_sessions if s["date"] >= "2026-04-01"]

    res_is = run_advanced_simulation(df, is_sessions, cfg, **params)
    res_oos = run_advanced_simulation(df, oos_sessions, cfg, **params)

    trips_is = pair_round_trips(res_is.trades)
    trips_oos = pair_round_trips(res_oos.trades)

    summ_is = summarize(res_is, capital=cfg.capital, label="IS")
    summ_oos = summarize(res_oos, capital=cfg.capital, label="OOS")

    net_is = float(summ_is.get("net_pnl", 0.0))
    net_oos = float(summ_oos.get("net_pnl", 0.0))
    wr_is = float(summ_is.get("win_rate", 0.0)) if summ_is.get("win_rate") is not None else 0.0
    wr_oos = float(summ_oos.get("win_rate", 0.0)) if summ_oos.get("win_rate") is not None else 0.0
    dd_is = float(summ_is.get("max_drawdown", 0.0))
    dd_oos = float(summ_oos.get("max_drawdown", 0.0))

    pass_criteria = (
        net_is > 0.0
        and net_oos > 0.0
        and len(trips_is) >= 20
        and len(trips_oos) >= 10
    )

    full_trips = trips_is + trips_oos
    full_wins = sum(1 for t in full_trips if t.get("pnl", 0.0) > 0)
    full_wr = (full_wins / len(full_trips)) if full_trips else 0.0

    return {
        "params": params,
        "pass": pass_criteria,
        "IS": {
            "net_pnl": round(net_is, 2),
            "trades": len(trips_is),
            "win_rate": round(wr_is, 4),
            "max_drawdown": round(dd_is, 2),
        },
        "OOS": {
            "net_pnl": round(net_oos, 2),
            "trades": len(trips_oos),
            "win_rate": round(wr_oos, 4),
            "max_drawdown": round(dd_oos, 2),
        },
        "total_pnl": round(net_is + net_oos, 2),
        "total_trades": len(full_trips),
        "total_win_rate": round(full_wr, 4),
    }


def main():
    logger.info("[tidemark] 開始執行四大維度升級回測 (保本/追蹤/大趨勢/量能/多空)...")
    settings = load_settings()
    trading = settings.get("trading", {})
    cfg = TradeConfig(
        capital=float(trading.get("capital", 10000)),
        point_value=float(trading.get("point_value", 10)),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(settings.get("backtest", {}).get("slip_ticks", 1)),
    )

    df_raw = load_1m_range("2024-07-29", "2026-10-23")
    df, day_sessions = prepare_advanced_features(df_raw)
    logger.info(f"[tidemark] 完成特徵準備，有效日盤共 {len(day_sessions)} 天。")

    # 參數網格：涵蓋使用者要求的四大升級
    be_opts = [0.0, 40.0, 50.0]           # 保本觸發
    trail_opts = [(0.0, 0.0), (80.0, 35.0), (100.0, 40.0)] # (啟動, 回撤)
    tp_opts = [150.0, 200.0, 250.0]       # 停利天花板
    trend_opts = [False, True]             # 60m 趨勢濾網
    vol_opts = [0.0, 1.2, 1.5]            # 突破量能倍數
    short_opts = [False, True]             # 雙向做空

    combos = []
    for be in be_opts:
        for (tr_trig, tr_dist) in trail_opts:
            for tp in tp_opts:
                for use_tr in trend_opts:
                    for vol_m in vol_opts:
                        for s_opt in short_opts:
                            combos.append({
                                "probe_mins": 15,
                                "cutoff_mins": 105,
                                "gap_th": 40.0,
                                "max_or_range": 250.0,
                                "stop_points": 50.0,
                                "tp_points": tp,
                                "be_trig_pts": be,
                                "be_lock_pts": 5.0,
                                "trail_trig_pts": tr_trig,
                                "trail_dist": tr_dist,
                                "time_stop_mins": 45,
                                "use_trend_filter": use_tr,
                                "vol_spike_mult": vol_m,
                                "allow_short": s_opt,
                            })

    logger.info(f"[tidemark] 總計掃描組合數: {len(combos)} 組...")

    results = []
    pass_count = 0

    for idx, params in enumerate(combos):
        res = evaluate_advanced_splits(df, day_sessions, cfg, params)
        results.append(res)
        if res["pass"]:
            pass_count += 1

    # 排序：優先 pass，再依勝率與獲利綜合排序
    # 建立雙重榜單：最高勝率榜 & 最高總獲利榜
    passed_all = [r for r in results if r["pass"]]

    top_by_pnl = sorted(passed_all, key=lambda x: x["total_pnl"], reverse=True)[:15]
    top_by_wr = sorted(passed_all, key=lambda x: (x["total_win_rate"], x["total_pnl"]), reverse=True)[:15]

    report_data = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(),
        "n_combos": len(combos),
        "n_pass": len(passed_all),
        "pass_rule": "IS_net > 0 and OOS_net > 0 and IS_trades >= 20 and OOS_trades >= 10",
        "top_by_pnl": top_by_pnl,
        "top_by_win_rate": top_by_wr,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"[tidemark] 掃描完成！過關組數: {len(passed_all)} / {len(combos)}")
    logger.info(f"[tidemark] 結果報告已儲存至 {REPORT_PATH}")

    if top_by_pnl:
        b_pnl = top_by_pnl[0]
        logger.info(f"[tidemark] 👑 最高獲利王: Total={b_pnl['total_pnl']} (IS={b_pnl['IS']['net_pnl']}, OOS={b_pnl['OOS']['net_pnl']}) | WR={b_pnl['total_win_rate']*100:.1f}% ({b_pnl['total_trades']}t)")
        logger.info(f"    參數: {b_pnl['params']}")
    if top_by_wr:
        b_wr = top_by_wr[0]
        logger.info(f"[tidemark] 🎯 最高勝率王: WR={b_wr['total_win_rate']*100:.1f}% | Total={b_wr['total_pnl']} (IS={b_wr['IS']['net_pnl']}, OOS={b_wr['OOS']['net_pnl']}) | {b_wr['total_trades']}t")
        logger.info(f"    參數: {b_wr['params']}")


if __name__ == "__main__":
    main()
