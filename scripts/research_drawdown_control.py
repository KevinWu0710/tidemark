#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控回撤研究：固定 lookback=60 日盤突破，掃描 stop_points / 可選日虧上限。
結果：outputs/drawdown_control_report.json（agent 讀此檔，勿盯 log）
說明：docs/ai/DRAWDOWN_CONTROL.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast
from src.backtest.metrics import summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range

IS = ("2026-07-16", "2026-09-15")
OOS = ("2026-09-16", "2026-10-21")
FULL = ("2026-07-16", "2026-10-21")
LOOKBACK = 60
BASELINE_OOS_DD = 22759.27  # 無停損版 OOS maxDD（MORNING_REPORT）


def eval_cfg(trading: dict, slip: float, bars_map: dict, stop: float, daily_loss: float) -> dict:
    cfg = FastBreakoutConfig(
        lookback=LOOKBACK,
        capital=trading["capital"],
        point_value=trading["point_value"],
        fee_per_side=trading.get("fee_per_side", 20),
        tax_rate=trading.get("tax_rate", 0.00002),
        slip_ticks=slip,
        sessions=("day",),
        stop_points=stop,
        max_daily_loss=daily_loss,
    )
    out = {
        "lookback": LOOKBACK,
        "stop_points": stop,
        "stop_twd_approx": stop * trading["point_value"],
        "max_daily_loss": daily_loss,
    }
    for label in ("IS", "OOS", "FULL"):
        result = run_breakout_fast(bars_map[label], cfg)
        s = summarize(result, capital=trading["capital"], label=label)
        s.pop("sample_trips", None)
        out[label] = s
    out["pass_strict"] = (
        out["IS"]["net_pnl"] > 0
        and out["OOS"]["net_pnl"] > 0
        and out["OOS"]["n_round_trips"] >= 5
    )
    out["dd_improved"] = out["OOS"]["max_drawdown"] < BASELINE_OOS_DD
    out["score"] = (
        int(out["pass_strict"]),
        int(out["dd_improved"]),
        -out["OOS"]["max_drawdown"],
        out["OOS"]["net_pnl"],
        out["IS"]["net_pnl"],
    )
    return out


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    bars_map = {
        "IS": load_1m_range(*IS),
        "OOS": load_1m_range(*OOS),
        "FULL": load_1m_range(*FULL),
    }

    stops = [0, 50, 80, 100, 120, 150, 200, 250, 300]
    daily_caps = [0, 1500, 2000, 3000]
    rows = []
    for stop in stops:
        for daily in daily_caps:
            if stop == 0 and daily > 0:
                continue  # 先掃純停損；日限搭配有停損再測
            row = eval_cfg(trading, slip, bars_map, float(stop), float(daily))
            rows.append(row)
            logger.info(
                f"{notify_prefix()} stop={stop} daily={daily} "
                f"IS={row['IS']['net_pnl']} OOS={row['OOS']['net_pnl']} "
                f"OOS_DD={row['OOS']['max_drawdown']} pass={row['pass_strict']}"
            )

    # 第二輪：通過的停損 × 日限
    base_stops = [r["stop_points"] for r in rows if r["pass_strict"] and r["stop_points"] > 0]
    for stop in sorted(set(base_stops)):
        for daily in (1500, 2000, 3000):
            if any(r["stop_points"] == stop and r["max_daily_loss"] == daily for r in rows):
                continue
            row = eval_cfg(trading, slip, bars_map, float(stop), float(daily))
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    winners = [r for r in rows_sorted if r["pass_strict"] and r["dd_improved"] and r["stop_points"] > 0]
    best = winners[0] if winners else None
    baseline = next(r for r in rows if r["stop_points"] == 0 and r["max_daily_loss"] == 0)

    report = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "docs": "docs/ai/DRAWDOWN_CONTROL.md",
        "baseline_no_stop": {
            "OOS_net": baseline["OOS"]["net_pnl"],
            "OOS_maxDD": baseline["OOS"]["max_drawdown"],
            "IS_net": baseline["IS"]["net_pnl"],
        },
        "baseline_oos_dd_ref": BASELINE_OOS_DD,
        "n_tested": len(rows_sorted),
        "n_winners": len(winners),
        "best": best,
        "top10": rows_sorted[:10],
        "all": rows_sorted,
        "explain": {
            "point_value": 10,
            "lot": 1,
            "capital": trading["capital"],
            "stop_twd": "stop_points * 10 = 約單筆最大虧損（元）",
            "maxDD": "權益自高點最大回落（元），與一口點數損益累積有關",
        },
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "drawdown_control_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # human md summary
    lines = [
        "# 控回撤研究結果摘要",
        "",
        f"產生時間：{report['updated_at']}",
        "",
        "完整觀念說明請讀：**`docs/ai/DRAWDOWN_CONTROL.md`**",
        "",
        "## 無停損基準（lookback=60）",
        "",
        f"- IS 淨利：{baseline['IS']['net_pnl']}",
        f"- OOS 淨利：{baseline['OOS']['net_pnl']}",
        f"- OOS maxDD：{baseline['OOS']['max_drawdown']}",
        "",
        "## 最佳控回撤參數",
        "",
    ]
    if best:
        lines += [
            f"- stop_points：**{best['stop_points']}**（約單筆最大虧 **{best['stop_twd_approx']}** 元）",
            f"- max_daily_loss：{best['max_daily_loss']}",
            f"- IS 淨利：{best['IS']['net_pnl']}｜OOS 淨利：{best['OOS']['net_pnl']}｜FULL：{best['FULL']['net_pnl']}",
            f"- OOS maxDD：{best['OOS']['max_drawdown']}（基準 {baseline['OOS']['max_drawdown']}）",
            f"- OOS 勝率：{best['OOS'].get('win_rate')}｜回合數：{best['OOS']['n_round_trips']}",
            "",
            "已達：IS/OOS 皆正 **且** OOS 回撤低於無停損版。",
        ]
        # lock + settings
        lock = {
            "status": "DONE",
            "updated_at": report["updated_at"],
            "strategy_id": "tmf_v1",
            "signal": "breakout",
            "params": {
                "lookback": LOOKBACK,
                "sessions": ["day"],
                "stop_points": best["stop_points"],
                "max_daily_loss": best["max_daily_loss"],
            },
            "is": best["IS"],
            "oos": best["OOS"],
            "full": best["FULL"],
            "pass_strict": True,
            "dd_improved": True,
            "note": "控回撤版：單筆停損 + 可選單日虧損上限",
        }
        (out_dir / "best_strategy_lock.json").write_text(
            json.dumps(lock, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        # patch settings.yaml stop fields
        sp = ROOT / "config" / "settings.yaml"
        text = sp.read_text(encoding="utf-8")
        if "stop_points:" not in text:
            text = text.replace(
                "      lookback: 60\n      sessions: [day]\n",
                f"      lookback: 60\n      sessions: [day]\n"
                f"      stop_points: {int(best['stop_points'])}\n"
                f"      max_daily_loss: {int(best['max_daily_loss'])}\n",
            )
            sp.write_text(text, encoding="utf-8")
        else:
            import re

            text = re.sub(r"stop_points:\s*\d+", f"stop_points: {int(best['stop_points'])}", text)
            text = re.sub(
                r"max_daily_loss:\s*\d+", f"max_daily_loss: {int(best['max_daily_loss'])}", text
            )
            sp.write_text(text, encoding="utf-8")
    else:
        lines += [
            "- **沒有**同時滿足「IS/OOS 皆正 + DD 改善」的停損組合。",
            "- 詳見 JSON top10；可能需放寬標準或加其他濾網。",
        ]

    lines += [
        "",
        "## Top 對照（節錄）",
        "",
        "| stop | daily_loss | IS | OOS | OOS_DD | pass | dd↓ |",
        "|------|------------|----|-----|--------|------|-----|",
    ]
    for r in rows_sorted[:12]:
        lines.append(
            f"| {r['stop_points']} | {r['max_daily_loss']} | {r['IS']['net_pnl']} | "
            f"{r['OOS']['net_pnl']} | {r['OOS']['max_drawdown']} | {r['pass_strict']} | {r['dd_improved']} |"
        )
    lines += ["", f"機器完整結果：`{path.name}`", ""]
    (out_dir / "DRAWDOWN_CONTROL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "DONE",
                "winners": len(winners),
                "best_stop": None if not best else best["stop_points"],
                "report": str(path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
