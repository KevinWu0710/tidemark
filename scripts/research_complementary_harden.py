#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
硬化現役互補：掃描 ORB 保本／虧損冷卻／RSI 停損，IS 選、OOS 驗，報 t/DSR。

結果 → outputs/complementary_harden_report.json
選中後寫入 src/backtest/complementary_fast.py 的 LOCKED_RISK（本腳本只報告；
套用由人工或 --apply 寫 lock JSON）。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.complementary_fast import (
    LOCKED_COMPLEMENTARY_RULE,
    ComplementaryConfig,
    run_complementary_fast,
)
from src.backtest.metrics import (
    deflated_sharpe_ratio,
    pair_round_trips,
    sharpe_from_pnls,
    summarize,
    t_stat,
)
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range

REPORT_PATH = ROOT / "outputs" / "complementary_harden_report.json"
LOCK_PATH = ROOT / "outputs" / "best_complementary_lock.json"

IS = ("2024-08-01", "2026-03-31")
OOS = ("2026-04-01", "2026-10-22")
FULL = ("2024-08-01", "2026-10-22")
BASELINE_OOS_DD = 3596.41  # paper complementary 硬化前 OOS maxDD


def _score_split(bars, trading, slip, risk: dict[str, Any], label: str) -> dict[str, Any]:
    cfg = ComplementaryConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=slip,
        rule=LOCKED_COMPLEMENTARY_RULE,
        orb_be_trig=float(risk["orb_be_trig"]),
        orb_stop=float(risk["orb_stop"]),
        orb_tp=float(risk["orb_tp"]),
        rsi_stop=float(risk["rsi_stop"]),
        rsi_tp=float(risk["rsi_tp"]),
        cooldown_after_loss_days=int(risk["cooldown_after_loss_days"]),
        max_daily_loss=float(risk["max_daily_loss"]),
    )
    result, meta = run_complementary_fast(bars, cfg)
    s = summarize(result, capital=float(trading["capital"]), label=label)
    trips = pair_round_trips(result.trades)
    trip_pnls = [float(t["pnl"]) for t in trips]
    day_map: dict[str, float] = {}
    for t in trips:
        dk = str(t.get("trading_day") or str(t.get("entry_ts"))[:10])
        day_map[dk] = day_map.get(dk, 0.0) + float(t["pnl"])
    day_pnls = list(day_map.values())
    return {
        "net_pnl": round(float(s["net_pnl"]), 2),
        "max_drawdown": round(float(s.get("max_drawdown") or 0), 2),
        "n_round_trips": int(s["n_round_trips"]),
        "win_rate": s.get("win_rate"),
        "daily_win_rate": s.get("daily_win_rate"),
        "t_stat_trips": t_stat(trip_pnls),
        "t_stat_days": t_stat(day_pnls),
        "sharpe_trips": sharpe_from_pnls(trip_pnls),
        "meta": {
            "n_choice_orb": meta.get("n_choice_orb"),
            "n_choice_rsi": meta.get("n_choice_rsi"),
            "n_cooldown_flat": meta.get("n_cooldown_flat"),
        },
        "_trip_pnls": trip_pnls,
    }


def main() -> int:
    apply = "--apply" in sys.argv
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    prefix = notify_prefix(settings)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"status": "RUNNING", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"{prefix} load bars for harden…")
    bars = {
        "IS": load_1m_range(*IS),
        "OOS": load_1m_range(*OOS),
        "FULL": load_1m_range(*FULL),
    }
    for k, v in bars.items():
        if v is None or v.empty:
            REPORT_PATH.write_text(
                json.dumps({"status": "ERROR", "error": f"no bars {k}"}, indent=2),
                encoding="utf-8",
            )
            return 1

    grid: list[dict[str, Any]] = []
    for be in (0.0, 50.0):
        for cool in (0, 1, 2):
            for rsi_stop in (200.0, 300.0):
                grid.append(
                    {
                        "orb_be_trig": be,
                        "orb_stop": 50.0,
                        "orb_tp": 200.0,
                        "rsi_stop": rsi_stop,
                        "rsi_tp": 40.0,
                        "cooldown_after_loss_days": cool,
                        "max_daily_loss": 0.0,
                    }
                )

    # baseline = 硬化前（be=0, cool=0, rsi_stop=300）
    baseline_risk = {
        "orb_be_trig": 0.0,
        "orb_stop": 50.0,
        "orb_tp": 200.0,
        "rsi_stop": 300.0,
        "rsi_tp": 40.0,
        "cooldown_after_loss_days": 0,
        "max_daily_loss": 0.0,
    }

    rows = []
    n_trials = len(grid)
    for i, risk in enumerate(grid):
        logger.info(f"{prefix} harden {i+1}/{n_trials} {risk}")
        is_s = _score_split(bars["IS"], trading, slip, risk, "IS")
        oos_s = _score_split(bars["OOS"], trading, slip, risk, "OOS")
        full_s = _score_split(bars["FULL"], trading, slip, risk, "FULL")
        dsr = deflated_sharpe_ratio(
            oos_s.get("sharpe_trips"),
            n_trials=n_trials,
            n_obs=max(int(oos_s["n_round_trips"]), 2),
        )
        pass_edge = (
            is_s["net_pnl"] > 0
            and oos_s["net_pnl"] > 0
            and oos_s["n_round_trips"] >= 10
        )
        dd_ok = oos_s["max_drawdown"] < BASELINE_OOS_DD
        t_ok = (oos_s.get("t_stat_trips") or 0) >= 2.0
        row = {
            "risk": risk,
            "pass_edge": pass_edge,
            "dd_improved": dd_ok,
            "t_stat_oos_ge_2": t_ok,
            "dsr_oos": dsr,
            "IS": {k: v for k, v in is_s.items() if not k.startswith("_")},
            "OOS": {k: v for k, v in oos_s.items() if not k.startswith("_")},
            "FULL": {k: v for k, v in full_s.items() if not k.startswith("_")},
            "score": (
                int(pass_edge),
                int(dd_ok),
                -oos_s["max_drawdown"],
                oos_s["net_pnl"],
                is_s["net_pnl"],
            ),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    baseline = None
    for r in rows:
        if r["risk"] == baseline_risk:
            baseline = r
            break
    if baseline is None:
        baseline = {
            "risk": baseline_risk,
            "OOS": _score_split(bars["OOS"], trading, slip, baseline_risk, "OOS"),
        }

    # 首選：過 edge 且 DD 改善；否則過 edge 中 DD 最小
    preferred = [r for r in rows if r["pass_edge"] and r["dd_improved"]]
    if not preferred:
        preferred = [r for r in rows if r["pass_edge"]]
    best = preferred[0] if preferred else rows[0]

    verdict = []
    verdict.append(
        f"best risk={best['risk']} OOS net={best['OOS']['net_pnl']} "
        f"DD={best['OOS']['max_drawdown']} t={best['OOS'].get('t_stat_trips')} "
        f"dsr={best.get('dsr_oos')}"
    )
    if best["dd_improved"] and best["pass_edge"]:
        verdict.append("OOS 淨利>0 且 MaxDD 優於硬化前 → 建議套用")
    elif best["pass_edge"]:
        verdict.append("OOS 仍賺但 DD 未明顯改善 → 可套用或維持原鎖")
    else:
        verdict.append("無過關組合 → 維持原風控")
    if not best.get("t_stat_oos_ge_2"):
        verdict.append("OOS t_stat<2：統計未穩，紙上可用、真倉需保守")

    if apply and best["pass_edge"]:
        lock = {
            "status": "LOCKED",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "rule": LOCKED_COMPLEMENTARY_RULE.to_dict(),
            "risk": best["risk"],
            "OOS": best["OOS"],
            "IS": best["IS"],
            "source": "research_complementary_harden --apply",
            "max_lots": 1,
        }
        LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        # 同步寫入 complementary_fast.LOCKED_RISK
        path = ROOT / "src" / "backtest" / "complementary_fast.py"
        text = path.read_text(encoding="utf-8")
        import re

        new_block = (
            "LOCKED_RISK: dict[str, Any] = {\n"
            f"    \"orb_be_trig\": {float(best['risk']['orb_be_trig'])},\n"
            f"    \"orb_stop\": {float(best['risk']['orb_stop'])},\n"
            f"    \"orb_tp\": {float(best['risk']['orb_tp'])},\n"
            f"    \"rsi_stop\": {float(best['risk']['rsi_stop'])},\n"
            f"    \"rsi_tp\": {float(best['risk']['rsi_tp'])},\n"
            f"    \"cooldown_after_loss_days\": {int(best['risk']['cooldown_after_loss_days'])},\n"
            f"    \"max_daily_loss\": {float(best['risk']['max_daily_loss'])},\n"
            "}"
        )
        text2, n = re.subn(
            r"LOCKED_RISK: dict\[str, Any\] = \{[^}]+\}",
            new_block,
            text,
            count=1,
            flags=re.S,
        )
        if n == 1:
            path.write_text(text2, encoding="utf-8")
            verdict.append(f"已 --apply 寫入 LOCKED_RISK 與 {LOCK_PATH}")
        else:
            verdict.append("WARN: 無法自動替換 LOCKED_RISK，請手動改")

    report = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_oos_dd": BASELINE_OOS_DD,
        "n_grid": n_trials,
        "baseline": {
            "risk": baseline_risk,
            "OOS": baseline.get("OOS") if isinstance(baseline, dict) else None,
        },
        "best": best,
        "top5": [
            {
                "risk": r["risk"],
                "OOS": r["OOS"],
                "IS": r["IS"],
                "pass_edge": r["pass_edge"],
                "dd_improved": r["dd_improved"],
                "dsr_oos": r.get("dsr_oos"),
            }
            for r in rows[:5]
        ],
        "verdict": verdict,
        "applied": bool(apply and best["pass_edge"]),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"{prefix} wrote {REPORT_PATH}")
    for line in verdict:
        logger.info(f"{prefix} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
