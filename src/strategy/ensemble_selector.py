"""
日盤雙軌事前分流（探針結束後、進場前）。

只用 ≤ 探針末日可得特徵，輸出任一軌道名或 flat。
規則由 research_*_selector 在 IS 上學、OOS 驗證。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OpenProbeFeatures:
    """探針結束瞬間可得（禁止用當日稍後突破量／收盤）。"""

    date: str
    gap: float
    or_range: float
    gap_unfilled_long: bool
    open_minus_ema: float
    probe_close_pos: float  # 0=探針低、1=探針高；or_range=0 → 0.5
    probe_vol_mean: float
    prior_day_range: float


@dataclass(frozen=True)
class SelectorRule:
    """
    可解釋門檻：條件全成立 → then_choice，否則 → else_choice。
    無任何條件 → default_choice（always_*）。
    skip_or_range_gt：探針過寬直接 flat（優先於其他條件）。
    """

    id: str
    gap_ge: float | None = None
    gap_le: float | None = None
    abs_gap_le: float | None = None
    or_range_le: float | None = None
    or_range_ge: float | None = None
    open_above_ema: bool | None = None
    gap_unfilled_required: bool | None = None
    probe_close_pos_ge: float | None = None
    probe_close_pos_lt: float | None = None
    skip_or_range_gt: float | None = None
    then_choice: str = "orb"
    else_choice: str = "rsi"
    default_choice: str = "orb"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(feat: OpenProbeFeatures, rule: SelectorRule) -> str:
    if rule.skip_or_range_gt is not None and feat.or_range > rule.skip_or_range_gt:
        return "flat"

    has_cond = any(
        x is not None
        for x in (
            rule.gap_ge,
            rule.gap_le,
            rule.abs_gap_le,
            rule.or_range_le,
            rule.or_range_ge,
            rule.open_above_ema,
            rule.gap_unfilled_required,
            rule.probe_close_pos_ge,
            rule.probe_close_pos_lt,
        )
    )
    if not has_cond:
        return rule.default_choice

    ok = True
    if rule.gap_ge is not None and not (feat.gap >= rule.gap_ge):
        ok = False
    if rule.gap_le is not None and not (feat.gap <= rule.gap_le):
        ok = False
    if rule.abs_gap_le is not None and not (abs(feat.gap) <= rule.abs_gap_le):
        ok = False
    if rule.or_range_le is not None and not (feat.or_range <= rule.or_range_le):
        ok = False
    if rule.or_range_ge is not None and not (feat.or_range >= rule.or_range_ge):
        ok = False
    if rule.open_above_ema is True and not (feat.open_minus_ema > 0):
        ok = False
    if rule.open_above_ema is False and not (feat.open_minus_ema <= 0):
        ok = False
    if rule.gap_unfilled_required is True and not feat.gap_unfilled_long:
        ok = False
    if rule.probe_close_pos_ge is not None and not (
        feat.probe_close_pos >= rule.probe_close_pos_ge
    ):
        ok = False
    if rule.probe_close_pos_lt is not None and not (
        feat.probe_close_pos < rule.probe_close_pos_lt
    ):
        ok = False

    return rule.then_choice if ok else rule.else_choice


def rule_candidates(
    *,
    track_a: str = "orb",
    track_b: str = "rsi",
    include_flat_else: bool = True,
) -> list[SelectorRule]:
    """
    小網格可解釋規則（只在 IS 上選）。
    直覺：大跳空／寬探針 → track_a（動能）；小跳空／窄探針 → track_b（回歸）。
    """
    a, b = track_a, track_b
    rules: list[SelectorRule] = [
        SelectorRule(id=f"always_{a}", default_choice=a),
        SelectorRule(id=f"always_{b}", default_choice=b),
    ]

    for g in (40.0, 60.0, 80.0, 100.0, 120.0):
        rules.append(
            SelectorRule(
                id=f"gap≥{int(g)}→{a}_else_{b}",
                gap_ge=g,
                then_choice=a,
                else_choice=b,
            )
        )
        if include_flat_else:
            rules.append(
                SelectorRule(
                    id=f"gap≥{int(g)}→{a}_else_flat",
                    gap_ge=g,
                    then_choice=a,
                    else_choice="flat",
                )
            )
        rules.append(
            SelectorRule(
                id=f"gap≥{int(g)}→{b}_else_{a}",
                gap_ge=g,
                then_choice=b,
                else_choice=a,
            )
        )

    for ag in (20.0, 30.0, 40.0, 50.0):
        rules.append(
            SelectorRule(
                id=f"abs_gap≤{int(ag)}→{b}_else_{a}",
                abs_gap_le=ag,
                then_choice=b,
                else_choice=a,
            )
        )
        if include_flat_else:
            rules.append(
                SelectorRule(
                    id=f"abs_gap≤{int(ag)}→{b}_else_flat",
                    abs_gap_le=ag,
                    then_choice=b,
                    else_choice="flat",
                )
            )

    for r in (40.0, 60.0, 80.0, 100.0, 120.0, 150.0):
        rules.append(
            SelectorRule(
                id=f"or≤{int(r)}→{b}_else_{a}",
                or_range_le=r,
                then_choice=b,
                else_choice=a,
            )
        )
        rules.append(
            SelectorRule(
                id=f"or≥{int(r)}→{a}_else_{b}",
                or_range_ge=r,
                then_choice=a,
                else_choice=b,
            )
        )

    for g in (40.0, 60.0, 80.0, 100.0):
        for r in (60.0, 80.0, 100.0, 120.0):
            rules.append(
                SelectorRule(
                    id=f"gap≥{int(g)}_or≤{int(r)}→{a}_else_{b}",
                    gap_ge=g,
                    or_range_le=r,
                    then_choice=a,
                    else_choice=b,
                )
            )
            if include_flat_else:
                rules.append(
                    SelectorRule(
                        id=f"gap≥{int(g)}_or≤{int(r)}→{a}_else_flat",
                        gap_ge=g,
                        or_range_le=r,
                        then_choice=a,
                        else_choice="flat",
                    )
                )

    for ag in (30.0, 40.0, 50.0):
        for r in (60.0, 80.0, 100.0):
            rules.append(
                SelectorRule(
                    id=f"abs_gap≤{int(ag)}_or≤{int(r)}→{b}_else_{a}",
                    abs_gap_le=ag,
                    or_range_le=r,
                    then_choice=b,
                    else_choice=a,
                )
            )

    rules.append(
        SelectorRule(
            id=f"above_ema→{a}_else_{b}",
            open_above_ema=True,
            then_choice=a,
            else_choice=b,
        )
    )
    rules.append(
        SelectorRule(
            id=f"below_ema→{b}_else_{a}",
            open_above_ema=False,
            then_choice=b,
            else_choice=a,
        )
    )

    for g in (60.0, 80.0, 100.0):
        rules.append(
            SelectorRule(
                id=f"unfilled_gap≥{int(g)}→{a}_else_{b}",
                gap_ge=g,
                gap_unfilled_required=True,
                then_choice=a,
                else_choice=b,
            )
        )

    for thr in (0.55, 0.65, 0.75):
        rules.append(
            SelectorRule(
                id=f"probe_pos≥{thr}→{a}_else_{b}",
                probe_close_pos_ge=thr,
                then_choice=a,
                else_choice=b,
            )
        )
        rules.append(
            SelectorRule(
                id=f"probe_pos<{thr}→{b}_else_{a}",
                probe_close_pos_lt=thr,
                then_choice=b,
                else_choice=a,
            )
        )

    for skip in (120.0, 150.0, 180.0):
        rules.append(
            SelectorRule(
                id=f"skip_or>{int(skip)}_else_always_{a}",
                skip_or_range_gt=skip,
                default_choice=a,
            )
        )
        rules.append(
            SelectorRule(
                id=f"skip_or>{int(skip)}_gap≥80→{a}_else_{b}",
                skip_or_range_gt=skip,
                gap_ge=80.0,
                then_choice=a,
                else_choice=b,
            )
        )

    return rules
