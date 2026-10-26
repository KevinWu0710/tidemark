#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真互補雙軌事前分流：ORB 動能（tmf_daily_trend）vs RSI28 回歸（tmf_v1 鎖）。

- 分流時機：探針結束（≈09:00），只用當下可得開盤特徵
- RSI 只計 09:00 後進場（與分流時刻對齊，避免探針內成交回填）
- IS 鎖規則、OOS 驗證；並與「舊 secure/trend 分流」結果並列比較

結果：outputs/complementary_selector_report.json
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

from src.backtest.fast_signals import FastSignalConfig, run_rsi_tp_fast  # noqa: E402
from src.backtest.metrics import pair_round_trips  # noqa: E402
from src.core.config import load_settings, notify_prefix  # noqa: E402
from src.data.bars import load_1m_range  # noqa: E402
from src.strategy.ensemble_selector import (  # noqa: E402
    OpenProbeFeatures,
    SelectorRule,
    decide,
    rule_candidates,
)

REPORT_PATH = ROOT / "outputs" / "complementary_selector_report.json"
OLD_REPORT = ROOT / "outputs" / "ensemble_selector_report.json"
IS_END = "2026-03-31"
OOS_START = "2026-04-01"
FULL_START = "2024-07-29"
FULL_END = "2026-10-23"
PROBE_MINS = 15
PROBE_END_MINUTES = 8 * 60 + 45 + PROBE_MINS  # 09:00


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

# 動能軌 = 驗證過的 trend ORB
ORB_PARAMS = {
    "probe_mins": 15,
    "cutoff_mins": 105,
    "gap_th": 40.0,
    "max_or_range": 250.0,
    "stop_points": 50.0,
    "tp_points": 200.0,
    "be_trig_pts": 0.0,
    "be_lock_pts": 5.0,
    "trail_trig_pts": 0.0,
    "trail_dist": 0.0,
    "time_stop_mins": 45,
    "use_trend_filter": False,
    "vol_spike_mult": 0.0,
    "allow_short": False,
}


def _minutes_of_day(ts) -> int:
    t = pd.Timestamp(ts)
    return int(t.hour * 60 + t.minute)


def _day_pnl_map(
    trades: list[dict[str, Any]],
    *,
    entry_after_probe: bool = False,
) -> dict[str, float]:
    trips = pair_round_trips(trades)
    by: dict[str, float] = defaultdict(float)
    for t in trips:
        ets = t.get("entry_ts")
        if entry_after_probe and ets is not None:
            if _minutes_of_day(ets) < PROBE_END_MINUTES:
                continue
        dk = str(pd.Timestamp(ets).date()) if ets is not None else None
        if dk is None:
            continue
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
            gap >= 40.0
            and (
                pclose is None
                or (isinstance(pclose, float) and np.isnan(pclose))
                or pl > float(pclose)
            )
        )
        out.append(
            OpenProbeFeatures(
                date=str(sess["date"]),
                gap=gap,
                or_range=float(or_range),
                gap_unfilled_long=unfilled,
                open_minus_ema=float(sess["open_px"]) - float(sess["trend_ema"]),
                probe_close_pos=float(pos),
                probe_vol_mean=float(np.mean(volume[s_idx : s_idx + PROBE_MINS])),
                prior_day_range=float(prior_range),
            )
        )
        prior_range = float(np.max(high[s_idx:e_idx]) - np.min(low[s_idx:e_idx]))
    return out


def _score_days(
    choices: dict[str, str],
    pnl_by_track: dict[str, dict[str, float]],
    dates: list[str],
) -> dict[str, Any]:
    day_pnls: dict[str, float] = {}
    counts: dict[str, int] = defaultdict(int)
    for d in dates:
        ch = choices.get(d, "flat")
        counts[ch] += 1
        if ch == "flat":
            continue
        track_pnl = pnl_by_track.get(ch, {})
        if d in track_pnl:
            day_pnls[d] = track_pnl[d]

    n = len(day_pnls)
    wins = sum(1 for v in day_pnls.values() if v > 0)
    net = sum(day_pnls.values())
    return {
        "n_choice": dict(counts),
        "n_trade_days": n,
        "n_win_days": wins,
        "n_loss_days": n - wins,
        "daily_win_rate": round(wins / n, 4) if n else None,
        "net_pnl": round(net, 2),
        "avg_day_pnl": round(net / n, 2) if n else None,
    }


def _apply_rule(feats: list[OpenProbeFeatures], rule: SelectorRule) -> dict[str, str]:
    return {f.date: decide(f, rule) for f in feats}


def _oracle_choices(
    dates: list[str],
    pnl_a: dict[str, float],
    pnl_b: dict[str, float],
    name_a: str,
    name_b: str,
) -> dict[str, str]:
    """上界：選當日較優軌；若最佳損益≤0 則空手（不強迫吃虧損日）。"""
    out: dict[str, str] = {}
    for d in dates:
        a = pnl_a.get(d)
        b = pnl_b.get(d)
        cand: list[tuple[float, str]] = []
        if a is not None:
            cand.append((a, name_a))
        if b is not None:
            cand.append((b, name_b))
        if not cand:
            out[d] = "flat"
            continue
        best_pnl, best_name = max(cand, key=lambda x: x[0])
        out[d] = best_name if best_pnl > 0 else "flat"
    return out


def _rank_key(score: dict[str, Any]) -> tuple:
    wr = score.get("daily_win_rate")
    wr_v = float(wr) if wr is not None else -1.0
    net = float(score.get("net_pnl") or 0.0)
    n = int(score.get("n_trade_days") or 0)
    return (wr_v, net, n)


def _load_old_summary() -> dict[str, Any] | None:
    if not OLD_REPORT.exists():
        return None
    try:
        data = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return None
    locked = data.get("locked_selector") or {}
    return {
        "source": str(OLD_REPORT),
        "pair": "secure_vs_trend（偽互補）",
        "locked_rule_id": (locked.get("rule") or {}).get("id"),
        "IS": locked.get("IS"),
        "OOS": locked.get("OOS"),
        "baselines": data.get("baselines"),
        "overlap": data.get("overlap"),
        "verdict": data.get("verdict"),
    }


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

    logger.info(f"{prefix} complementary: load 1m {FULL_START}..{FULL_END}")
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

    # --- ORB day PnL ---
    logger.info(f"{prefix} simulate ORB trend…")
    res_orb = run_advanced_simulation(df, day_sessions, cfg, **ORB_PARAMS)
    pnl_orb = _day_pnl_map(res_orb.trades)

    # --- RSI day PnL（日盤、探針後進場）---
    logger.info(f"{prefix} simulate RSI28 TP40…")
    rsi_cfg = FastSignalConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
        sessions=("day",),
        stop_points=300.0,
        max_daily_loss=0.0,
        flatten_eod=True,
        allow_short=False,
    )
    res_rsi = run_rsi_tp_fast(df_raw, 28, 20.0, rsi_cfg, take_profit_points=40.0)
    pnl_rsi_full = _day_pnl_map(res_rsi.trades, entry_after_probe=False)
    pnl_rsi = _day_pnl_map(res_rsi.trades, entry_after_probe=True)

    pnl_by = {"orb": pnl_orb, "rsi": pnl_rsi}

    is_dates = [f.date for f in feats if f.date <= IS_END]
    oos_dates = [f.date for f in feats if f.date >= OOS_START]
    is_feats = [feat_by_date[d] for d in is_dates]
    oos_feats = [feat_by_date[d] for d in oos_dates]

    baselines = {}
    for name, rule in (
        ("always_orb", SelectorRule(id="always_orb", default_choice="orb")),
        ("always_rsi", SelectorRule(id="always_rsi", default_choice="rsi")),
    ):
        baselines[name] = {
            "IS": _score_days(_apply_rule(is_feats, rule), pnl_by, is_dates),
            "OOS": _score_days(_apply_rule(oos_feats, rule), pnl_by, oos_dates),
        }
    # 參考：RSI 含探針內進場（部署 tmf_v1 全貌）
    baselines["always_rsi_incl_probe"] = {
        "IS": _score_days(
            {d: "rsi" for d in is_dates},
            {"rsi": pnl_rsi_full},
            is_dates,
        ),
        "OOS": _score_days(
            {d: "rsi" for d in oos_dates},
            {"rsi": pnl_rsi_full},
            oos_dates,
        ),
    }

    oracle = {
        "note": "偷看當日 orb/rsi 損益後選優 — 不可部署，僅上界",
        "IS": _score_days(
            _oracle_choices(is_dates, pnl_orb, pnl_rsi, "orb", "rsi"),
            pnl_by,
            is_dates,
        ),
        "OOS": _score_days(
            _oracle_choices(oos_dates, pnl_orb, pnl_rsi, "orb", "rsi"),
            pnl_by,
            oos_dates,
        ),
    }

    candidates = rule_candidates(track_a="orb", track_b="rsi", include_flat_else=True)
    scored: list[dict[str, Any]] = []
    for rule in candidates:
        s_is = _score_days(_apply_rule(is_feats, rule), pnl_by, is_dates)
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

    # 另鎖一條：IS 淨利最大（在過關集合內）— 方便與「勝率鎖」比較
    by_pnl = sorted(
        passed,
        key=lambda x: (float(x["IS"]["net_pnl"]), float(x["IS"]["daily_win_rate"] or 0)),
        reverse=True,
    )

    def _lock(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        rule = SelectorRule(**entry["rule"])
        s_oos = _score_days(_apply_rule(oos_feats, rule), pnl_by, oos_dates)
        b_orb = baselines["always_orb"]["OOS"]
        b_rsi = baselines["always_rsi"]["OOS"]
        return {
            "rule": entry["rule"],
            "IS": entry["IS"],
            "OOS": s_oos,
            "oos_beats_always_orb_wr": (
                s_oos["daily_win_rate"] is not None
                and b_orb["daily_win_rate"] is not None
                and s_oos["daily_win_rate"] > b_orb["daily_win_rate"]
            ),
            "oos_beats_always_rsi_wr": (
                s_oos["daily_win_rate"] is not None
                and b_rsi["daily_win_rate"] is not None
                and s_oos["daily_win_rate"] > b_rsi["daily_win_rate"]
            ),
            "oos_beats_always_orb_pnl": s_oos["net_pnl"] > b_orb["net_pnl"],
            "oos_beats_always_rsi_pnl": s_oos["net_pnl"] > b_rsi["net_pnl"],
            "oos_net_positive": float(s_oos["net_pnl"]) > 0,
        }

    locked_wr = _lock(passed[0] if passed else None)
    locked_pnl = _lock(by_pnl[0] if by_pnl else None)

    top5 = []
    for x in passed[:5]:
        rule = SelectorRule(**x["rule"])
        top5.append(
            {
                "rule_id": x["rule"]["id"],
                "IS": x["IS"],
                "OOS": _score_days(_apply_rule(oos_feats, rule), pnl_by, oos_dates),
            }
        )

    both = set(pnl_orb) & set(pnl_rsi)
    only_orb = set(pnl_orb) - set(pnl_rsi)
    only_rsi = set(pnl_rsi) - set(pnl_orb)
    overlap = {
        "both_trade_days": len(both),
        "only_orb": len(only_orb),
        "only_rsi": len(only_rsi),
        "orb_days": len(pnl_orb),
        "rsi_after_probe_days": len(pnl_rsi),
        "jaccard": round(len(both) / len(set(pnl_orb) | set(pnl_rsi)), 4)
        if (pnl_orb or pnl_rsi)
        else None,
        "note": "Jaccard 越低越互補；舊 secure/trend ≈ 1.0",
    }

    old = _load_old_summary()

    verdict: list[str] = []
    verdict.append(
        f"互補重疊 Jaccard={overlap['jaccard']} "
        f"(both={overlap['both_trade_days']} only_orb={overlap['only_orb']} "
        f"only_rsi={overlap['only_rsi']})"
    )
    if locked_wr:
        verdict.append(
            f"勝率鎖: {locked_wr['rule']['id']} | "
            f"OOS WR={locked_wr['OOS']['daily_win_rate']} net={locked_wr['OOS']['net_pnl']}"
        )
    if locked_pnl:
        verdict.append(
            f"淨利鎖: {locked_pnl['rule']['id']} | "
            f"OOS WR={locked_pnl['OOS']['daily_win_rate']} net={locked_pnl['OOS']['net_pnl']}"
        )
    verdict.append(
        f"基線 OOS: always_orb WR={baselines['always_orb']['OOS']['daily_win_rate']} "
        f"net={baselines['always_orb']['OOS']['net_pnl']} | "
        f"always_rsi WR={baselines['always_rsi']['OOS']['daily_win_rate']} "
        f"net={baselines['always_rsi']['OOS']['net_pnl']}"
    )
    verdict.append(
        f"oracle OOS WR={oracle['OOS']['daily_win_rate']} net={oracle['OOS']['net_pnl']}"
    )

    # 自動比較舊分流
    compare: dict[str, Any] = {"old_secure_trend": old}
    if locked_wr and old and old.get("OOS"):
        compare["oos_wr_delta_vs_old_selector"] = round(
            float(locked_wr["OOS"]["daily_win_rate"] or 0)
            - float((old["OOS"] or {}).get("daily_win_rate") or 0),
            4,
        )
        compare["oos_pnl_delta_vs_old_selector"] = round(
            float(locked_wr["OOS"]["net_pnl"]) - float((old["OOS"] or {}).get("net_pnl") or 0),
            2,
        )
    if locked_pnl and old and old.get("OOS"):
        compare["pnl_lock_oos_pnl_delta_vs_old"] = round(
            float(locked_pnl["OOS"]["net_pnl"]) - float((old["OOS"] or {}).get("net_pnl") or 0),
            2,
        )

    # 部署建議
    deploy = "none"
    if locked_pnl and locked_pnl["oos_net_positive"] and (
        locked_pnl["oos_beats_always_orb_pnl"] or locked_pnl["oos_beats_always_rsi_pnl"]
    ):
        deploy = "consider_pnl_lock"
        verdict.append("淨利鎖 OOS 勝過至少一條固定基線 → 可考慮接 paper 驗證")
    elif locked_wr and locked_wr["oos_net_positive"] and locked_wr["oos_beats_always_orb_wr"]:
        deploy = "wr_only_thin"
        verdict.append("勝率鎖有邊但可能犧牲淨利 → 僅作高勝率過濾參考")
    else:
        verdict.append("分流未穩定同時勝過雙基線 → 暫不取代單軌")

    report = {
        "status": "DONE",
        "brand": "tidemark",
        "updated_at": datetime.now().isoformat(),
        "pair": "orb(trend_ORB) vs rsi(tmf_v1 RSI28 TP40 stop300)",
        "range": {"full": [FULL_START, FULL_END], "IS_end": IS_END, "OOS_start": OOS_START},
        "causal": {
            "selector_at": "probe_end≈09:00",
            "rsi_entries": "only after 09:00",
            "orb_entries": "after probe (built-in)",
        },
        "n_feature_days": len(feats),
        "n_rules_scanned": len(candidates),
        "n_rules_pass_is": len(passed),
        "is_gate": "n_trade_days>=20 and net_pnl>0；勝率鎖排序=日勝率→淨利；另報淨利鎖",
        "overlap": overlap,
        "baselines": baselines,
        "oracle_look_ahead": oracle,
        "locked_by_win_rate": locked_wr,
        "locked_by_net_pnl": locked_pnl,
        "top5_by_is_wr": top5,
        "compare_vs_old_secure_trend": compare,
        "deploy_hint": deploy,
        "verdict": verdict,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{prefix} wrote {REPORT_PATH}")
    for line in verdict:
        logger.info(f"{prefix} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
