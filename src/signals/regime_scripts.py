"""
多局式訊號腳本（研究用配方）。

進場皆基於 RSI28 reclaim>20，再用時間／確認／趨勢過濾對應不同虧損局式。
實作細節與回測見 scripts/research_loss_regimes.py。
"""
from __future__ import annotations

from typing import Any

# 局式 → 腳本 id（與 research_loss_regimes.SCRIPTS 對齊）
REGIME_SCRIPTS: dict[str, dict[str, Any]] = {
    "default_mean_revert": {
        "id": "A_baseline_rsi",
        "summary": "RSI28 reclaim>20；TP40；stop300；日結 flatten",
    },
    "open_waterfall": {
        "id": "D_skip_open_15m",
        "summary": "開盤 15 分鐘內禁止進場（避開急殺停損）",
    },
    "late_entry_grind": {
        "id": "B_no_late_entry",
        "summary": "開盤後 150 分鐘內才允許進場（避開尾盤拖到收盤）",
    },
    "dead_cat_no_follow": {
        "id": "C_next_bar_confirm",
        "summary": "RSI reclaim 後下一根收盤續漲才進；或 F_time_stop_30",
    },
    "wrong_way_crash": {
        "id": "E_ema60_trend",
        "summary": "close > EMA60 才允許 RSI 做多",
    },
}
