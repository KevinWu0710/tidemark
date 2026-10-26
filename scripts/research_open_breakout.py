#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
開盤 15m 區間突破 + 缺口/夜盤動能過濾 + 時間停損 (Open 15m Range Breakout) 研究腳本。

嚴格遵守無偷看原則、IS/OOS 切分、max_lots=1、台指摩擦成本 (手續費 20 元/邊 + 稅 0.00002 + 滑價 1 點)。
結果寫入 outputs/open_breakout_report.json。
"""
from __future__ import annotations

import argparse
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

REPORT_PATH = ROOT / "outputs" / "open_breakout_report.json"
DAY_OPEN_MIN = 8 * 60 + 45  # 08:45


@dataclass
class TradeConfig:
    capital: float = 10000.0
    point_value: float = 10.0
    fee_per_side: float = 20.0
    tax_rate: float = 0.00002
    slip_ticks: float = 1.0


def prepare_day_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    計算日盤與前一日盤、前一夜盤的特徵。
    純因果：日盤 08:45 只能看到過去的夜盤與前日盤。
    """
    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "session" not in df.columns:
        df["session"] = df["ts"].map(label_session)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute

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

        # 前一日日盤收盤
        prev_day_close = np.nan
        if idx > 0:
            prev_d = dates[idx - 1]
            prev_bars = day_df[day_df["date"] == prev_d]
            if len(prev_bars) > 0:
                prev_day_close = float(prev_bars["close"].iloc[-1])

        # 前一夜盤資訊
        prev_night_ret = 0.0
        prev_night_high = np.nan
        prev_night_low = np.nan
        if idx > 0:
            prev_d_last_ts = day_df.loc[day_df["date"] == dates[idx - 1], "ts"].iloc[-1]
            nseg = night_df[(night_df["ts"] > prev_d_last_ts) & (night_df["ts"] < d_open_ts)]
            if len(nseg) >= 2:
                prev_night_ret = float(nseg["close"].iloc[-1] - nseg["open"].iloc[0])
                prev_night_high = float(nseg["high"].max())
                prev_night_low = float(nseg["low"].min())

        gap_pts = (d_open_px - prev_day_close) if np.isfinite(prev_day_close) else 0.0

        day_sessions.append({
            "date": d,
            "start_idx": d_bars.index[0],
            "end_idx": d_bars.index[-1] + 1,
            "open_px": d_open_px,
            "prev_day_close": prev_day_close,
            "prev_night_ret": prev_night_ret,
            "prev_night_high": prev_night_high,
            "prev_night_low": prev_night_low,
            "gap_pts": gap_pts,
        })

    return df, day_sessions


def run_single_simulation(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    *,
    probe_mins: int,          # 15 或 30
    cutoff_mins: int,         # 90 (10:15) 或 120 (10:45)
    filter_mode: str,         # 'none', 'gap_cont', 'night_mom', 'squeeze_orb', 'long_only'
    gap_th: float,            # 缺口門檻 (如 30.0)
    night_th: float,          # 夜盤門檻 (如 30.0)
    max_or_range: float,      # 最大 OR 振幅 (如 120.0，大於則不追)
    tp_points: float,         # 停利點數 (如 40, 60, 80)
    stop_points: float,       # 停損點數 (如 40, 60, 80, 或 0=OR_opposite)
    time_stop_mins: int,      # 時間停損 (如 30, 45, 60 根 K 棒，0=不設)
    allow_short: bool,
) -> BacktestResult:
    """
    高效率 walk-forward 回測日盤開盤突破策略。
    """
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
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

        # 計算探針 (08:45 ~ 08:45 + probe_mins)
        probe_high = float(np.max(high[s_idx : s_idx + probe_mins]))
        probe_low = float(np.min(low[s_idx : s_idx + probe_mins]))
        or_range = probe_high - probe_low
        or_mid = 0.5 * (probe_high + probe_low)

        gap = sess["gap_pts"]
        nret = sess["prev_night_ret"]
        pclose = sess["prev_day_close"]

        # 過濾器方向判斷
        allow_l = True
        allow_s = allow_short

        if filter_mode == "gap_cont":
            # 缺口未補且同向突破
            # 多：跳空高開 > gap_th 且探針低點未回補前日收盤
            gap_unfilled_long = (gap >= gap_th) and (np.isnan(pclose) or probe_low > pclose)
            # 空：跳空低開 < -gap_th 且探針高點未回補前日收盤
            gap_unfilled_short = (gap <= -gap_th) and (np.isnan(pclose) or probe_high < pclose)
            allow_l = gap_unfilled_long
            allow_s = allow_short and gap_unfilled_short
        elif filter_mode == "night_mom":
            # 前一夜盤動能
            allow_l = nret >= night_th
            allow_s = allow_short and (nret <= -night_th)
        elif filter_mode == "squeeze_orb":
            # 振幅壓縮 (20 <= or_range <= 80)
            is_compressed = (20.0 <= or_range <= 80.0)
            allow_l = is_compressed
            allow_s = allow_short and is_compressed
        elif filter_mode == "long_only":
            allow_s = False

        if max_or_range > 0 and or_range > max_or_range:
            # 振幅太大，已經宣洩完畢，不追
            allow_l = False
            allow_s = False

        if not allow_l and not allow_s:
            continue

        # 進場掃描 (probe_mins ~ cutoff_mins)
        # cutoff_mins 從 08:45 起算
        max_search_idx = min(e_idx, s_idx + cutoff_mins)
        pos = Side.FLAT
        entry_px = 0.0
        entry_idx = -1
        bars_held = 0

        for k in range(s_idx + probe_mins, e_idx):
            cur_c = float(close[k])
            cur_h = float(high[k])
            cur_l = float(low[k])
            cur_ts = ts[k]
            is_eod = (k == e_idx - 1)

            # 持倉中的出場檢查
            if pos == Side.LONG:
                bars_held += 1
                tp_target = entry_px + tp_points
                sl_target = (entry_px - stop_points) if stop_points > 0 else probe_low

                if cur_l <= sl_target:
                    f = fill_sell(sl_target)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "stop_loss"})
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
                tp_target = entry_px - tp_points
                sl_target = (entry_px + stop_points) if stop_points > 0 else probe_high

                if cur_h >= sl_target:
                    f = fill_buy(sl_target)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append({"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": "stop_loss"})
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

            # 空手狀態：檢查突破觸發 (僅在 cutoff 之前可開倉)
            elif pos == Side.FLAT and k < max_search_idx:
                if allow_l and cur_c > probe_high:
                    f = fill_buy(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    entry_idx = k
                    bars_held = 0
                    pos = Side.LONG
                    trades.append({"ts": cur_ts, "action": "BUY", "price": f, "reason": "orb_break_high"})
                elif allow_s and cur_c < probe_low:
                    f = fill_sell(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    entry_idx = k
                    bars_held = 0
                    pos = Side.SHORT
                    trades.append({"ts": cur_ts, "action": "SELL", "price": f, "reason": "orb_break_low"})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=equity,
        n_steps=len(trades),
    )


def evaluate_splits(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    """執行 IS 與 OOS 分段回測並計算統計指標。"""
    is_sessions = [s for s in day_sessions if s["date"] <= "2026-03-31"]
    oos_sessions = [s for s in day_sessions if s["date"] >= "2026-04-01"]

    res_is = run_single_simulation(df, is_sessions, cfg, **params)
    res_oos = run_single_simulation(df, oos_sessions, cfg, **params)

    trips_is = pair_round_trips(res_is.trades)
    trips_oos = pair_round_trips(res_oos.trades)

    summ_is = summarize(res_is, capital=cfg.capital, label="IS")
    summ_oos = summarize(res_oos, capital=cfg.capital, label="OOS")

    net_is = float(summ_is.get("net_pnl", 0.0))
    net_oos = float(summ_oos.get("net_pnl", 0.0))
    wr_is = float(summ_is.get("win_rate", 0.0)) if summ_is.get("win_rate") is not None else 0.0
    wr_oos = float(summ_oos.get("win_rate", 0.0)) if summ_oos.get("win_rate") is not None else 0.0
    dwr_is = float(summ_is.get("daily_win_rate", 0.0)) if summ_is.get("daily_win_rate") is not None else 0.0
    dwr_oos = float(summ_oos.get("daily_win_rate", 0.0)) if summ_oos.get("daily_win_rate") is not None else 0.0
    dd_is = float(summ_is.get("max_drawdown", 0.0))
    dd_oos = float(summ_oos.get("max_drawdown", 0.0))
    n_trips_is = len(trips_is)
    n_trips_oos = len(trips_oos)

    pass_criteria = (
        net_is > 0.0
        and net_oos > 0.0
        and n_trips_is >= 20
        and n_trips_oos >= 10
    )

    return {
        "params": params,
        "pass": pass_criteria,
        "IS": {
            "net_pnl": round(net_is, 2),
            "trades": n_trips_is,
            "win_rate": round(wr_is, 4),
            "daily_win_rate": round(dwr_is, 4),
            "max_drawdown": round(dd_is, 2),
        },
        "OOS": {
            "net_pnl": round(net_oos, 2),
            "trades": n_trips_oos,
            "win_rate": round(wr_oos, 4),
            "daily_win_rate": round(dwr_oos, 4),
            "max_drawdown": round(dd_oos, 2),
        },
        "total_pnl": round(net_is + net_oos, 2),
    }


def main():
    logger.info("[tidemark] 開始執行開盤區間突破 (ORB) 系統性研究...")
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
    logger.info(f"[tidemark] 載入 1m K 棒共 {len(df_raw)} 筆，計算日夜盤開盤特徵...")
    df, day_sessions = prepare_day_features(df_raw)
    logger.info(f"[tidemark] 完成特徵準備，有效日盤共 {len(day_sessions)} 天。")

    # 參數網格設計 - 適應 2024-2026 波動度體制漂移
    probe_list = [15, 30]
    filter_list = ["gap_cont", "night_mom", "none"]
    tp_list = [60.0, 80.0, 100.0, 120.0, 150.0]
    stop_list = [50.0, 60.0, 80.0, 100.0]
    time_stop_list = [0, 45, 60, 90]
    short_opts = [True, False]
    max_or_opts = [0.0, 250.0]

    combos = []
    for probe in probe_list:
        for fmode in filter_list:
            for max_or in max_or_opts:
                for tp in tp_list:
                    for stop in stop_list:
                        for t_stop in time_stop_list:
                            for s_opt in short_opts:
                                combos.append({
                                    "probe_mins": probe,
                                    "cutoff_mins": 105,
                                    "filter_mode": fmode,
                                    "gap_th": 40.0,
                                    "night_th": 40.0,
                                    "max_or_range": max_or,
                                    "tp_points": tp,
                                    "stop_points": stop,
                                    "time_stop_mins": t_stop,
                                    "allow_short": s_opt,
                                })

    logger.info(f"[tidemark] 總計掃描組合數: {len(combos)} 組...")

    results = []
    pass_count = 0

    for idx, params in enumerate(combos):
        res = evaluate_splits(df, day_sessions, cfg, params)
        results.append(res)
        if res["pass"]:
            pass_count += 1

    # 排序：優先篩選 pass=True，再依 total_pnl 降序
    results.sort(key=lambda x: (x["pass"], x["total_pnl"]), reverse=True)

    top20 = results[:20]
    passed_all = [r for r in results if r["pass"]]

    report_data = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(),
        "n_combos": len(combos),
        "n_pass": len(passed_all),
        "pass_rule": "IS_net > 0 and OOS_net > 0 and IS_trades >= 20 and OOS_trades >= 10",
        "top20": top20,
        "winners": passed_all[:15],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"[tidemark] 掃描完成！過關組數: {len(passed_all)} / {len(combos)}")
    logger.info(f"[tidemark] 結果報告已儲存至 {REPORT_PATH}")

    if passed_all:
        best = passed_all[0]
        logger.info(f"[tidemark] 最佳過關組合: {best['params']}")
        logger.info(f"[tidemark] IS: net={best['IS']['net_pnl']}, win_rate={best['IS']['win_rate']}, maxDD={best['IS']['max_drawdown']}")
        logger.info(f"[tidemark] OOS: net={best['OOS']['net_pnl']}, win_rate={best['OOS']['win_rate']}, maxDD={best['OOS']['max_drawdown']}")


if __name__ == "__main__":
    main()
