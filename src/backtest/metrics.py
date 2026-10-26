from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import pandas as pd

from src.backtest.engine import BacktestResult
from src.data.sessions import label_session, trading_day_for


def t_stat(values: Sequence[float]) -> float | None:
    """One-sample t-stat vs 0 on trade/day pnls. None if n < 2 or zero variance."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var / n)


def sharpe_from_pnls(values: Sequence[float]) -> float | None:
    """Sample Sharpe (mean/std) on a pnl series; not annualized."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var)


_EULER_GAMMA = 0.5772156649015329


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf_approx(p: float) -> float:
    """Acklam rational approximation of Φ^{-1}."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577459334152e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    )
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe_ratio(
    observed_sr: float | None,
    *,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float | None:
    """
    Bailey & López de Prado Deflated Sharpe (simplified).
    Returns Prob(true SR > 0 after selection bias) ≈ Φ((SR − SR₀) / σ(SR)).
    """
    if observed_sr is None or n_obs < 2 or n_trials < 1:
        return None
    e_max = (1 - _EULER_GAMMA) * _norm_ppf_approx(
        1 - 1.0 / n_trials
    ) + _EULER_GAMMA * _norm_ppf_approx(1 - 1.0 / (n_trials * math.e))
    se = math.sqrt(
        (1 - skew * observed_sr + (kurtosis - 1) / 4.0 * observed_sr**2)
        / max(n_obs - 1, 1)
    )
    if se <= 0:
        return None
    return _phi((observed_sr - e_max) / se)


def pair_round_trips(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match BUY/SELL opens with following CLOSE; use engine's per-close pnl."""
    trips: list[dict[str, Any]] = []
    open_t: dict[str, Any] | None = None
    for t in trades:
        action = t.get("action")
        if action in ("BUY", "SELL"):
            open_t = t
        elif action == "CLOSE" and open_t is not None:
            entry_ts = open_t.get("ts")
            sess = open_t.get("session") or (
                label_session(entry_ts) if entry_ts is not None else "closed"
            )
            tday = open_t.get("trading_day") or (
                trading_day_for(entry_ts) if entry_ts is not None else None
            )
            trips.append(
                {
                    "entry_ts": entry_ts,
                    "exit_ts": t.get("ts"),
                    "side": open_t["action"],
                    "entry": open_t.get("price"),
                    "exit": t.get("price"),
                    "pnl": float(t.get("pnl", 0.0)),
                    "reason_entry": open_t.get("reason"),
                    "reason_exit": t.get("reason"),
                    "session": sess,
                    "trading_day": tday,
                }
            )
            open_t = None
    return trips


def _day_key(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return str(pd.Timestamp(ts).date())
    except Exception:
        s = str(ts)
        return s[:10] if len(s) >= 10 else s


def daily_pnl_stats(
    trips: list[dict[str, Any]],
    *,
    session: str | None = None,
) -> dict[str, Any]:
    """
    時段內「交易日勝率」（禁止日夜混算）。

    - session=None：僅供除錯的混算（daily_win_rate_mixed）；正式過關勿用
    - session='day'|'night'：只計該盤；交易日鍵用 trading_day（夜盤跨午夜已對齊）
    daily_win_rate = 該盤合計淨利>0 的交易日數 / 該盤有平倉的交易日數
    """
    by_day: dict[str, float] = defaultdict(float)
    for t in trips:
        if session is not None and str(t.get("session")) != session:
            continue
        if session is not None:
            dk = t.get("trading_day") or _day_key(t.get("entry_ts")) or _day_key(
                t.get("exit_ts")
            )
        else:
            dk = _day_key(t.get("exit_ts"))
        if dk is None:
            continue
        by_day[dk] += float(t.get("pnl") or 0.0)
    if not by_day:
        return {
            "n_trade_days": 0,
            "n_win_days": 0,
            "n_loss_days": 0,
            "daily_win_rate": None,
            "daily_pnls": {},
        }
    win_days = sum(1 for v in by_day.values() if v > 0)
    loss_days = sum(1 for v in by_day.values() if v <= 0)
    n = len(by_day)
    return {
        "n_trade_days": n,
        "n_win_days": win_days,
        "n_loss_days": loss_days,
        "daily_win_rate": round(win_days / n, 4) if n else None,
        "daily_pnls": {k: round(v, 2) for k, v in sorted(by_day.items())},
    }


def trade_win_rate(trips: list[dict[str, Any]], *, session: str | None = None) -> float | None:
    xs = [
        t
        for t in trips
        if session is None or str(t.get("session")) == session
    ]
    if not xs:
        return None
    wins = sum(1 for t in xs if float(t.get("pnl") or 0) > 0)
    return round(wins / len(xs), 4)


def session_win_stats(trips: list[dict[str, Any]]) -> dict[str, Any]:
    """日盤／夜盤勝率分開（筆數勝率 + 交易日勝率）。"""
    out: dict[str, Any] = {}
    for sess in ("day", "night"):
        sub = [t for t in trips if str(t.get("session")) == sess]
        dstat = daily_pnl_stats(trips, session=sess)
        pnls = [float(t.get("pnl") or 0) for t in sub]
        out[sess] = {
            "n_round_trips": len(sub),
            "n_wins": sum(1 for p in pnls if p > 0),
            "n_losses": sum(1 for p in pnls if p <= 0),
            "win_rate": trade_win_rate(trips, session=sess),
            "net_pnl": round(sum(pnls), 2) if sub else 0.0,
            "n_trade_days": dstat["n_trade_days"],
            "n_win_days": dstat["n_win_days"],
            "n_loss_days": dstat["n_loss_days"],
            "daily_win_rate": dstat["daily_win_rate"],
        }
    return out


def passes_session_daily_win_rate(
    by_session: dict[str, Any],
    min_dwr: float,
    *,
    require_sessions: Sequence[str] | None = None,
    min_trade_days: int = 1,
) -> bool:
    """
    過關用：有成交的 session 各自 daily_win_rate 都須 > min_dwr。
    日夜禁止用混算值。
    """
    want = list(require_sessions) if require_sessions else ["day", "night"]
    checked = 0
    for sess in want:
        s = by_session.get(sess) or {}
        n_td = int(s.get("n_trade_days") or 0)
        if n_td < min_trade_days:
            continue
        checked += 1
        dwr = s.get("daily_win_rate")
        if dwr is None or float(dwr) <= float(min_dwr):
            return False
    return checked > 0


def summarize(
    result: BacktestResult,
    *,
    capital: float,
    label: str,
) -> dict[str, Any]:
    trips = pair_round_trips(result.trades)
    pnls = [t["pnl"] for t in trips]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    net = result.final_equity - capital
    by_session = session_win_stats(trips)
    # 向後相容：頂層 daily_win_rate / win_rate 改為「日盤」；夜盤見 by_session.night
    # 混算僅放 daily_win_rate_mixed（正式過關禁止使用）
    mixed = daily_pnl_stats(trips, session=None)
    day_s = by_session["day"]
    night_s = by_session["night"]

    # max drawdown on equity curve
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    for pt in result.equity_curve:
        eq = float(pt["equity"])
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak) if peak else 0.0

    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    long_trips = [t for t in trips if t["side"] == "BUY"]
    short_trips = [t for t in trips if t["side"] == "SELL"]

    return {
        "label": label,
        "capital": capital,
        "final_equity": round(result.final_equity, 2),
        "net_pnl": round(net, 2),
        "n_steps": result.n_steps,
        "n_round_trips": len(trips),
        "n_wins": len(wins),
        "n_losses": len(losses),
        # 頂層 = 日盤（不與夜盤混算）
        "win_rate": day_s["win_rate"],
        "n_trade_days": day_s["n_trade_days"],
        "n_win_days": day_s["n_win_days"],
        "n_loss_days": day_s["n_loss_days"],
        "daily_win_rate": day_s["daily_win_rate"],
        "day_win_rate": day_s["win_rate"],
        "day_daily_win_rate": day_s["daily_win_rate"],
        "night_win_rate": night_s["win_rate"],
        "night_daily_win_rate": night_s["daily_win_rate"],
        "daily_win_rate_mixed": mixed["daily_win_rate"],
        "by_session": by_session,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "best_trade": round(max(pnls), 2) if pnls else None,
        "worst_trade": round(min(pnls), 2) if pnls else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "long_trips": len(long_trips),
        "short_trips": len(short_trips),
        "long_pnl": round(sum(t["pnl"] for t in long_trips), 2),
        "short_pnl": round(sum(t["pnl"] for t in short_trips), 2),
        "sample_trips": trips[:5] + (trips[-5:] if len(trips) > 10 else []),
    }
