#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事前分流器：探針結束特徵 → 選 secure / trend / flat，IS 鎖規則、OOS 驗證勝率。

因果硬規則：分流只用探針結束前可得資訊；禁止用當日稍後突破量或收盤回填選軌。
結果：outputs/ensemble_selector_report.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.metrics import pair_round_trips  # noqa: E402
from src.core.config import load_settings, notify_prefix  # noqa: E402
from src.data.bars import load_1m_range  # noqa: E402
from src.strategy.ensemble_selector import (  # noqa: E402
    OpenProbeFeatures,
    SelectorRule,
    decide,
    rule_candidates,
)

REPORT_PATH = ROOT / "outputs" / "ensemble_selector_report.json"
IS_END = "2026-03-31"
OOS_START = "2026-04-01"
FULL_START = "2024-07-29"
FULL_END = "2026-10-23"
PROBE_MINS = 15


def _load_orb():
    path = ROOT / "scripts" / "research_orb_advanced.py"
    spec = importlib.util.spec_from_file_location("research_orb_advanced", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_orb = _load_orb()
TradeConfig = _orb.TradeConfig
prepare_advanced_features = _orb.prepare_advanced_features
run_advanced_simulation = _orb.run_advanced_simulation


SECURE_PARAMS = {
    "probe_mins": 15,
    "cutoff_mins": 105,
    "gap_th": 40.0,
    "max_or_range": 250.0,
    "stop_points": 50.0,
    "tp_points": 200.0,
    "be_trig_pts": 50.0,
    "be_lock_pts": 5.0,
    "trail_trig_pts": 0.0,
    "trail_dist": 0.0,
    "time_stop_mins": 45,
    "use_trend_filter": False,
    "vol_spike_mult": 1.2,
    "allow_short": False,
}
TREND_PARAMS = {
    **SECURE_PARAMS,
    "be_trig_pts": 0.0,
    "vol_spike_mult": 0.0,
}


def _day_pnl_map(trades: list[dict[str, Any]]) -> dict[str, float]:
    trips = pair_round_trips(trades)
    by: dict[str, float] = defaultdict(float)
    for t in trips:
        dk = str(pd.Timestamp(t["entry_ts"]).date())
        by[dk] += float(t.get("pnl") or 0.0)
    return dict(by)


def _build_features(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
) -> list[OpenProbeFeatures]:
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    volume = df["volume"].astype(float).values
    out: list[OpenProbeFeatures] = []
    prior_range = 0.0

    for sess in day_sessions:
        s_idx = int(sess["start_idx"])
        e_idx = int(sess["end_idx"])
        if e_idx - s_idx <= PROBE_MINS + 2:
            continue
        ph = float(np.max(high[s_idx : s_idx + PROBE_MINS]))
        pl = float(np.min(low[s_idx : s_idx + PROBE_MINS]))
        or_range = ph - pl
        pc = float(close[s_idx + PROBE_MINS - 1])
        pos = 0.5 if or_range <= 1e-9 else (pc - pl) / or_range
        pclose = sess["prev_day_close"]
        gap = float(sess["gap_pts"])
        unfilled = bool(
            gap >= 40.0 and (pclose is None or (isinstance(pclose, float) and np.isnan(pclose)) or pl > float(pclose))
        )
        # 前一根「已完成」日盤振幅：用上一筆 sess 的 prior_range 累積
        feat = OpenProbeFeatures(
            date=str(sess["date"]),
            gap=gap,
            or_range=float(or_range),
            gap_unfilled_long=unfilled,
            open_minus_ema=float(sess["open_px"]) - float(sess["trend_ema"]),
            probe_close_pos=float(pos),
            probe_vol_mean=float(np.mean(volume[s_idx : s_idx + PROBE_MINS])),
            prior_day_range=float(prior_range),
        )
        out.append(feat)
        # 更新 prior：當日全日振幅（分流當下還不知道，只供「下一日」用 → 因果 OK）
        prior_range = float(np.max(high[s_idx:e_idx]) - np.min(low[s_idx:e_idx]))
    return out


def _score_days(
    choices: dict[str, str],
    pnl_secure: dict[str, float],
    pnl_trend: dict[str, float],
    dates: list[str],
) -> dict[str, Any]:
    """依分流選擇取當日損益；flat → 不計入交易日。"""
    day_pnls: dict[str, float] = {}
    n_secure = n_trend = n_flat = 0
    for d in dates:
        ch = choices.get(d, "flat")
        if ch == "secure":
            n_secure += 1
            if d in pnl_secure:
                day_pnls[d] = pnl_secure[d]
        elif ch == "trend":
            n_trend += 1
            if d in pnl_trend:
                day_pnls[d] = pnl_trend[d]
        else:
            n_flat += 1

    n = len(day_pnls)
    wins = sum(1 for v in day_pnls.values() if v > 0)
    net = sum(day_pnls.values())
    return {
        "n_choice_secure": n_secure,
        "n_choice_trend": n_trend,
        "n_choice_flat": n_flat,
        "n_trade_days": n,
        "n_win_days": wins,
        "n_loss_days": n - wins,
        "daily_win_rate": round(wins / n, 4) if n else None,
        "net_pnl": round(net, 2),
        "avg_day_pnl": round(net / n, 2) if n else None,
    }


def _apply_rule(
    feats: list[OpenProbeFeatures], rule: SelectorRule
) -> dict[str, str]:
    return {f.date: decide(f, rule) for f in feats}


def _oracle_choices(
    dates: list[str],
    pnl_secure: dict[str, float],
    pnl_trend: dict[str, float],
) -> dict[str, str]:
    """上界（偷看當日結果）— 僅報告，不可部署。"""
    out: dict[str, str] = {}
    for d in dates:
        s = pnl_secure.get(d)
        t = pnl_trend.get(d)
        if s is None and t is None:
            out[d] = "flat"
        elif s is None:
            out[d] = "trend"
        elif t is None:
            out[d] = "secure"
        elif s >= t:
            out[d] = "secure"
        else:
            out[d] = "trend"
    return out


def _rank_key(score: dict[str, Any]) -> tuple:
    wr = score.get("daily_win_rate")
    wr_v = float(wr) if wr is not None else -1.0
    net = float(score.get("net_pnl") or 0.0)
    n = int(score.get("n_trade_days") or 0)
    return (wr_v, net, n)


def main() -> int:
    settings = load_settings()
    prefix = notify_prefix(settings)
    trading = settings.get("trading", {})
    cfg = TradeConfig(
        capital=float(trading.get("capital", 10000)),
        point_value=float(trading.get("point_value", 10)),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(settings.get("backtest", {}).get("slip_ticks", 1)),
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"status": "RUNNING", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"{prefix} selector: load 1m {FULL_START}..{FULL_END}")
    df_raw = load_1m_range(FULL_START, FULL_END)
    if df_raw.empty:
        REPORT_PATH.write_text(
            json.dumps({"status": "ERROR", "error": "no 1m bars"}, indent=2),
            encoding="utf-8",
        )
        return 1

    df, day_sessions = prepare_advanced_features(df_raw)
    feats = _build_features(df, day_sessions)
    feat_by_date = {f.date: f for f in feats}

    logger.info(f"{prefix} simulate secure / trend day PnL…")
    res_s = run_advanced_simulation(df, day_sessions, cfg, **SECURE_PARAMS)
    res_t = run_advanced_simulation(df, day_sessions, cfg, **TREND_PARAMS)
    pnl_s = _day_pnl_map(res_s.trades)
    pnl_t = _day_pnl_map(res_t.trades)

    is_dates = [f.date for f in feats if f.date <= IS_END]
    oos_dates = [f.date for f in feats if f.date >= OOS_START]
    is_feats = [feat_by_date[d] for d in is_dates]
    oos_feats = [feat_by_date[d] for d in oos_dates]

    baselines = {}
    for name, rule in (
        ("always_secure", SelectorRule(id="always_secure", default_choice="secure")),
        ("always_trend", SelectorRule(id="always_trend", default_choice="trend")),
    ):
        baselines[name] = {
            "IS": _score_days(_apply_rule(is_feats, rule), pnl_s, pnl_t, is_dates),
            "OOS": _score_days(_apply_rule(oos_feats, rule), pnl_s, pnl_t, oos_dates),
        }

    oracle_is = _oracle_choices(is_dates, pnl_s, pnl_t)
    oracle_oos = _oracle_choices(oos_dates, pnl_s, pnl_t)
    oracle = {
        "note": "偷看當日兩軌損益後選優 — 不可部署，僅上界",
        "IS": _score_days(oracle_is, pnl_s, pnl_t, is_dates),
        "OOS": _score_days(oracle_oos, pnl_s, pnl_t, oos_dates),
    }

    # IS 搜尋（舊雙軌：secure vs trend）
    candidates = rule_candidates(track_a="secure", track_b="trend")
    scored: list[dict[str, Any]] = []
    for rule in candidates:
        ch_is = _apply_rule(is_feats, rule)
        s_is = _score_days(ch_is, pnl_s, pnl_t, is_dates)
        # 過關門檻：IS 有足夠交易日、淨利>0
        ok = (
            int(s_is["n_trade_days"]) >= 20
            and float(s_is["net_pnl"]) > 0
            and s_is["daily_win_rate"] is not None
        )
        scored.append(
            {
                "rule": rule.to_dict(),
                "pass_is_gate": ok,
                "IS": s_is,
                "_rank": _rank_key(s_is) if ok else (-1.0, -1e18, 0),
            }
        )

    scored.sort(key=lambda x: x["_rank"], reverse=True)
    passed = [x for x in scored if x["pass_is_gate"]]
    best = passed[0] if passed else None

    locked = None
    if best is not None:
        rule = SelectorRule(**{k: v for k, v in best["rule"].items()})
        ch_oos = _apply_rule(oos_feats, rule)
        s_oos = _score_days(ch_oos, pnl_s, pnl_t, oos_dates)
        locked = {
            "rule": best["rule"],
            "IS": best["IS"],
            "OOS": s_oos,
            "oos_beats_always_secure_wr": (
                s_oos["daily_win_rate"] is not None
                and baselines["always_secure"]["OOS"]["daily_win_rate"] is not None
                and s_oos["daily_win_rate"]
                > baselines["always_secure"]["OOS"]["daily_win_rate"]
            ),
            "oos_beats_always_trend_wr": (
                s_oos["daily_win_rate"] is not None
                and baselines["always_trend"]["OOS"]["daily_win_rate"] is not None
                and s_oos["daily_win_rate"]
                > baselines["always_trend"]["OOS"]["daily_win_rate"]
            ),
            "oos_beats_always_trend_pnl": s_oos["net_pnl"]
            > baselines["always_trend"]["OOS"]["net_pnl"],
            "oos_net_positive": float(s_oos["net_pnl"]) > 0,
        }

    # 重疊診斷
    both = set(pnl_s) & set(pnl_t)
    only_s = set(pnl_s) - set(pnl_t)
    only_t = set(pnl_t) - set(pnl_s)

    top5 = []
    for x in passed[:5]:
        rule = SelectorRule(**{k: v for k, v in x["rule"].items()})
        s_oos = _score_days(_apply_rule(oos_feats, rule), pnl_s, pnl_t, oos_dates)
        top5.append({"rule_id": x["rule"]["id"], "IS": x["IS"], "OOS": s_oos})

    verdict = []
    if locked is None:
        verdict.append("IS 無規則過關（trades>=20 且 net>0）")
    else:
        verdict.append(f"鎖規則: {locked['rule']['id']}")
        verdict.append(
            f"OOS 日勝率={locked['OOS']['daily_win_rate']} "
            f"net={locked['OOS']['net_pnl']} "
            f"(trend基線 WR={baselines['always_trend']['OOS']['daily_win_rate']} "
            f"net={baselines['always_trend']['OOS']['net_pnl']})"
        )
        if locked["oos_net_positive"] and (
            locked["oos_beats_always_trend_wr"] or locked["oos_beats_always_secure_wr"]
        ):
            verdict.append("OOS：淨利>0 且日勝率勝過至少一條固定基線 → selector 有初步價值")
        elif locked["oos_net_positive"]:
            verdict.append("OOS：淨利>0，但日勝率未勝過固定基線 → 分流邊緣弱")
        else:
            verdict.append("OOS：淨利≤0 → 不建議部署此分流規則")
        verdict.append(
            f"oracle OOS WR={oracle['OOS']['daily_win_rate']} net={oracle['OOS']['net_pnl']}（上界）"
        )
    verdict.append(
        f"進場重疊：both={len(both)} only_secure={len(only_s)} only_trend={len(only_t)} "
        f"（secure⊆trend 時分流主要改出場／是否要求量能）"
    )

    report = {
        "status": "DONE",
        "brand": "tidemark",
        "updated_at": datetime.now().isoformat(),
        "range": {"full": [FULL_START, FULL_END], "IS_end": IS_END, "OOS_start": OOS_START},
        "engine": "research_orb_advanced HL-touch",
        "n_feature_days": len(feats),
        "n_rules_scanned": len(candidates),
        "n_rules_pass_is": len(passed),
        "is_gate": "n_trade_days>=20 and net_pnl>0；排序=日勝率→淨利→交易日數",
        "baselines": baselines,
        "oracle_look_ahead": oracle,
        "locked_selector": locked,
        "top5_by_is": top5,
        "overlap": {
            "both_trade_days": len(both),
            "only_secure": len(only_s),
            "only_trend": len(only_t),
        },
        "verdict": verdict,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{prefix} wrote {REPORT_PATH}")
    for line in verdict:
        logger.info(f"{prefix} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
