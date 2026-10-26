"""把 raw JSON 轉成白話戰情（給簡單 UI）。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


PHASE_ZH = {
    "boot": "系統剛啟動",
    "idle_pre": "開盤前待機（約 08:40 開始）",
    "session_day": "盤中監控中",
    "post_session": "日盤已結束，等待晚間覆盤",
    "evening_review": "正在做晚間覆盤",
    "evening_error": "晚間覆盤失敗",
    "idle_night": "今日流程結束，夜間待機",
    "stopped": "已停止",
}

CHOICE_ZH = {
    "orb": "開盤突破（ORB）",
    "rsi": "RSI 回歸做多",
    "flat": "今日不做",
}

ACTION_ZH = {
    "BUY": "買進開倉",
    "SELL": "賣出開倉",
    "CLOSE": "平倉",
}

REASON_ZH = {
    "orb_break_high": "向上突破開盤區間",
    "orb_break_low": "向下突破開盤區間",
    "take_profit": "停利出場",
    "stop_loss": "停損出場",
    "breakeven_stop": "保本停損出場",
    "time_stop": "持倉過久時間停損",
    "flatten_eod": "收盤強制平倉",
    "fake_entry": "測試進場",
}


def _num(v: Any, digits: int = 0) -> str:
    try:
        x = float(v)
    except Exception:
        return "—"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def _choice_block(choice: Any) -> dict[str, Any]:
    if not isinstance(choice, dict) or not choice:
        return {
            "headline": "尚無訊號",
            "lines": ["等開盤後系統會依跳空大小決定今天走哪一軌。"],
            "badge": "等待",
        }
    if not choice.get("ok", True) and choice.get("error"):
        return {
            "headline": "訊號尚未就緒",
            "lines": [f"原因：{choice.get('error')}"],
            "badge": "未就緒",
        }
    ch = str(choice.get("choice") or "")
    gap = choice.get("gap")
    or_range = choice.get("or_range")
    headline = CHOICE_ZH.get(ch, ch or "未知")
    lines = [
        f"規則：跳空 ≥ 120 點 → 開盤突破；否則 → RSI。",
    ]
    if gap is not None:
        try:
            g = float(gap)
            direction = "向上跳空" if g >= 0 else "向下跳空"
            lines.append(f"今日跳空：{_num(abs(g))} 點（{direction}）")
        except Exception:
            pass
    if or_range is not None:
        lines.append(f"開盤探針區間寬度：{_num(or_range)} 點")
    if ch == "orb":
        lines.append("解讀：跳空夠大，今天跟「開盤後突破」做多。")
    elif ch == "rsi":
        lines.append("解讀：跳空不大，今天等 RSI 超賣回升再做多。")
    elif ch == "flat":
        lines.append("解讀：條件不符，今天不開新倉。")
    return {"headline": headline, "lines": lines, "badge": ch or "—", "raw_choice": ch}


def _strategy_block(risk: dict[str, Any], *, place_sim: bool) -> dict[str, Any]:
    risk = risk or {}
    lines = [
        "口數上限：固定 1 口（不會加碼）",
        "下單模式：紙上模擬（不是真倉 LIVE）",
        f"永豐 SIM 真下單：{'有開' if place_sim else '關閉（只記帳）'}",
        "持倉中每 10 秒檢查一次停損／停利（過濾毛刺）",
    ]
    if risk:
        lines.append(
            f"ORB 軌：停損 {_num(risk.get('orb_stop'))} 點、"
            f"停利 {_num(risk.get('orb_tp'))} 點、"
            f"賺 {_num(risk.get('orb_be_trig'))} 點後啟動保本"
        )
        lines.append(
            f"RSI 軌：停損 {_num(risk.get('rsi_stop'))} 點、"
            f"停利 {_num(risk.get('rsi_tp'))} 點"
        )
    return {
        "headline": "風控中（一口・模擬）",
        "lines": lines,
        "badge": "paper",
    }


def _reason_zh(reason: str) -> str:
    r = str(reason or "")
    if r in REASON_ZH:
        return REASON_ZH[r]
    if r.startswith("time_stop"):
        return "時間停損出場"
    return r or "—"


def _orders_block(summary: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summary or {}
    events = list(events or [])
    net = summary.get("net_pnl")
    pos = summary.get("position") or summary.get("side") or "FLAT"
    n = len(events)
    if n == 0:
        headline = "今天還沒有成交"
        lines = ["開盤後若出現進場訊號，這裡會列出買賣時間與原因。"]
    else:
        pnl_s = "—" if net is None else f"{float(net):+,.0f} 元"
        headline = f"已模擬成交 {n} 筆 · 淨損益 {pnl_s}"
        lines = [f"目前部位：{pos}（FLAT=空手）"]
    timeline = []
    for e in events[-20:]:
        act = str(e.get("action") or "")
        timeline.append(
            {
                "time": str(e.get("ts") or "")[-8:]
                if len(str(e.get("ts") or "")) >= 8
                else str(e.get("ts") or ""),
                "ts": str(e.get("ts") or ""),
                "what": ACTION_ZH.get(act, act),
                "why": _reason_zh(str(e.get("reason") or "")),
                "price": _num(e.get("intent_price") or e.get("mid"), 0),
                "ok": bool(e.get("paper_ok", True)),
            }
        )
    return {
        "headline": headline,
        "lines": lines,
        "badge": f"{n}筆",
        "timeline": timeline,
        "net_pnl": net,
    }


def _journal_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    kind_zh = {
        "ops_boot": "系統啟動",
        "ops_stop": "系統停止",
        "session_start": "開始盤中監控",
        "session_end": "結束盤中監控",
        "session_skip": "略過盤中（缺金鑰）",
        "session_error": "盤中錯誤",
        "session_crash": "盤中異常中止",
        "evening_start": "開始晚間覆盤",
        "evening_done": "晚間覆盤完成",
        "evening_error": "晚間覆盤失敗",
        "live_choice": "出現當日分流",
        "live_fill": "模擬成交",
        "live_subscribed": "已訂閱即時行情",
        "ops_tick_error": "排程心跳錯誤",
    }
    rows = []
    for j in reversed(items[-30:]):
        kind = str(j.get("kind") or "")
        data = j.get("data") or {}
        detail = ""
        if kind == "evening_done":
            detail = f"淨損益 {data.get('net_pnl', '—')}"
        elif kind == "live_fill":
            detail = f"{data.get('action')} @ {data.get('intent_price')}（{data.get('reason')}）"
        elif kind == "live_choice":
            detail = CHOICE_ZH.get(str(data.get("choice")), str(data.get("choice") or ""))
        elif kind == "session_skip":
            detail = str(data.get("hint") or data.get("reason") or "")
        elif "error" in data:
            detail = str(data.get("error"))
        rows.append(
            {
                "ts": str(j.get("ts") or "")[11:19],
                "what": kind_zh.get(kind, kind),
                "detail": detail or "—",
            }
        )
    return rows


def _parse_ts(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except Exception:
        return None


def _health_block(
    status: dict[str, Any], journal: list[dict[str, Any]]
) -> dict[str, Any]:
    """一眼判斷：正常 / 注意 / 異常。"""
    phase = str(status.get("phase") or "")
    updated = _parse_ts(status.get("updated_at"))
    now = datetime.now(timezone.utc)
    age_sec: float | None = None
    if updated is not None:
        age_sec = max(0.0, (now - updated.astimezone(timezone.utc)).total_seconds())

    updated_local = "—"
    if updated is not None:
        updated_local = updated.astimezone().strftime("%H:%M:%S")

    err_kinds = {
        "evening_error",
        "session_error",
        "session_crash",
        "ops_tick_error",
    }
    recent_err = None
    for j in reversed(journal[-40:]):
        kind = str(j.get("kind") or "")
        if kind in err_kinds:
            data = j.get("data") or {}
            recent_err = str(data.get("error") or kind)
            break
    if phase == "evening_error" and not recent_err:
        recent_err = str(status.get("error") or "evening_error")

    # 異常：覆盤失敗，或盤中／排程錯誤尚未離開錯誤态
    if phase == "evening_error" or (
        recent_err
        and phase
        in ("session_day", "evening_review", "boot", "stopped", "")
    ):
        return {
            "level": "bad",
            "label": "異常",
            "hint": f"出問題了：{recent_err or phase}",
            "updated_local": updated_local,
            "age_sec": age_sec,
        }

    if not status or not phase:
        return {
            "level": "warn",
            "label": "尚未連上",
            "hint": "看不到 ops 狀態。請確認 Docker 的 ops 容器有在跑。",
            "updated_local": updated_local,
            "age_sec": age_sec,
        }

    if phase == "stopped":
        return {
            "level": "warn",
            "label": "已停止",
            "hint": "排程器已停下，不會自動盤中／覆盤。",
            "updated_local": updated_local,
            "age_sec": age_sec,
        }

    # 盤中應常更新；其他 phase 允許較久
    stale_limit = 180 if phase == "session_day" else 900
    if age_sec is not None and age_sec > stale_limit and phase not in (
        "idle_night",
        "idle_pre",
        "post_session",
    ):
        mins = int(age_sec // 60)
        return {
            "level": "warn",
            "label": "可能卡住",
            "hint": f"狀態已 {mins} 分鐘沒更新（上次 {updated_local}）。請看下方「系統紀錄」或 Docker logs。",
            "updated_local": updated_local,
            "age_sec": age_sec,
        }

    if recent_err and phase in ("idle_night", "post_session"):
        # 錯誤已過、流程繼續 → 注意但不紅
        return {
            "level": "warn",
            "label": "注意",
            "hint": f"稍早有錯誤紀錄：{recent_err}（目前流程已繼續）",
            "updated_local": updated_local,
            "age_sec": age_sec,
        }

    hints = {
        "boot": "剛啟動，稍等進入待機或盤中。",
        "idle_pre": "正常待機，開盤前會自動開始監控。",
        "session_day": "盤中監控正常運作中。",
        "post_session": "日盤結束，等晚上覆盤（預設約 18:00）。",
        "evening_review": "正在覆盤，請稍候。",
        "idle_night": "今日流程正常結束。",
    }
    return {
        "level": "ok",
        "label": "正常",
        "hint": hints.get(phase, "系統狀態正常。"),
        "updated_local": updated_local,
        "age_sec": age_sec,
    }


def build_view(
    *,
    status: dict[str, Any],
    date: str,
    choice: Any,
    risk: dict[str, Any],
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    journal: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    health = _health_block(status or {}, journal or [])
    return {
        "date": date,
        "phase_zh": PHASE_ZH.get(phase, phase or "未知狀態"),
        "phase_raw": phase,
        "health": health,
        "place_sim": bool(status.get("place_sim")),
        "has_sim_keys": bool(status.get("has_sim_keys")),
        "signal": _choice_block(choice if isinstance(choice, dict) else {}),
        "strategy": _strategy_block(risk or {}, place_sim=bool(status.get("place_sim"))),
        "orders": _orders_block(summary or {}, events or []),
        "journal_rows": _journal_rows(journal or []),
        "prefix": prefix,
        "status": status,
    }
