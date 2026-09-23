"""Healing-trajectory analysis.

Why the old version said "healing" for a scar that was getting worse, and
what this module does differently:

1. Direction per dimension. "Increasing" is only good for smoothness; for
   redness, size, colour difference, height and tightness it is bad. Every
   trend is converted to better/worse with DIMENSION_DIRECTION.
2. Photos are ordered by the date they were TAKEN, not upload order, and
   photos uploaded on the same day are spaced by index so the regression does
   not collapse to a zero-width x axis.
3. Worsening is checked first and uses several independent signals: fitted
   severity change, the most recent step, size growth, and what the patient
   reports. A long improving history cannot hide a recent turn for the worse.
4. Thresholds are both absolute and percentage-based, so small scars and
   large scars are judged fairly and photo noise is not read as change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .schemas import DIMENSION_DIRECTION, DIMENSION_LABELS, DIMENSIONS, ScarStateVector, SelfReport

# Minimum fitted change (absolute) for a dimension to count as a real change.
DIM_ABS_THRESHOLD = {
    "vascularity": 0.25,
    "pigmentation": 0.25,
    "pliability": 0.5,
    "height": 0.5,
    "surface_area": 0.004,
    "texture_regularity": 0.06,
    "color_delta_e": 1.5,
}
# Minimum fitted change relative to the starting value.
DIM_REL_THRESHOLD = {
    "vascularity": 0.10,
    "pigmentation": 0.12,
    "pliability": 0.10,
    "height": 0.10,
    "surface_area": 0.30,   # apparent size varies ~±25% with camera distance
    "texture_regularity": 0.06,
    "color_delta_e": 0.10,
}


@dataclass
class ObservationPoint:
    taken_at: datetime
    vector: ScarStateVector
    severity: float
    report: SelfReport
    reliable: bool = True
    observation_id: Optional[str] = None


@dataclass
class DimensionStat:
    dimension: str
    label: str
    trend: str            # increasing / decreasing / stable
    effect: str           # worse / better / no_change
    first: float
    latest: float
    total_change: float
    percent_change: float
    velocity: float       # per week
    acceleration: float
    confidence: float


@dataclass
class Alert:
    level: str            # warning / info / positive
    code: str
    message: str


@dataclass
class TrajectoryResult:
    status: str                      # baseline / improving / stable / worsening / stagnating
    n_observations: int
    severity_change: float = 0.0
    recent_change: float = 0.0
    severity_velocity: float = 0.0   # points per week
    area_growth: float = 0.0
    area_accelerating: bool = False
    dimension_stats: List[DimensionStat] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    worsening_reasons: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def _x_axis(times: List[datetime]) -> np.ndarray:
    days = np.array([(t - times[0]).total_seconds() / 86400.0 for t in times])
    if days[-1] - days[0] < 3.0:
        # Photos uploaded in one sitting: treat as equally spaced weekly steps.
        return np.arange(len(times), dtype=float) * 7.0
    # keep strict ordering even if two photos share a timestamp
    for i in range(1, len(days)):
        if days[i] <= days[i - 1]:
            days[i] = days[i - 1] + 0.5
    return days


def _fit(x: np.ndarray, y: np.ndarray):
    """Return (fitted total change, slope/day, acceleration, r2)."""
    n = len(x)
    if n < 2 or np.ptp(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    acc = 0.0
    if n >= 4:
        acc = float(np.polyfit(x, y, 2)[0])
    elif n == 3:
        d1 = (y[1] - y[0]) / max(x[1] - x[0], 1e-6)
        d2 = (y[2] - y[1]) / max(x[2] - x[1], 1e-6)
        acc = float((d2 - d1) / max((x[2] - x[0]) / 2.0, 1e-6))
    return float(slope * (x[-1] - x[0])), float(slope), acc, max(0.0, min(1.0, r2))


def _dimension_stats(x: np.ndarray, vecs: List[ScarStateVector]) -> List[DimensionStat]:
    out = []
    n = len(vecs)
    for d in DIMENSIONS:
        y = np.array([getattr(v, d) for v in vecs], dtype=float)
        fitted, slope, acc, r2 = _fit(x, y)
        # With only two points the "fit" is the raw difference.
        base = abs(y[0]) if abs(y[0]) > 1e-9 else max(abs(y).max(), 1e-9)
        thr = max(DIM_ABS_THRESHOLD[d], DIM_REL_THRESHOLD[d] * base)
        if d == "surface_area":
            thr = max(DIM_ABS_THRESHOLD[d] * 0.25, DIM_REL_THRESHOLD[d] * base)
        if fitted > thr:
            trend = "increasing"
        elif fitted < -thr:
            trend = "decreasing"
        else:
            trend = "stable"
        if trend == "stable":
            effect = "no_change"
        else:
            sign = 1 if trend == "increasing" else -1
            effect = "worse" if sign * DIMENSION_DIRECTION[d] > 0 else "better"
        pct = 100.0 * (y[-1] - y[0]) / base if base > 1e-9 else 0.0
        out.append(DimensionStat(
            dimension=d, label=DIMENSION_LABELS[d], trend=trend, effect=effect,
            first=round(float(y[0]), 4), latest=round(float(y[-1]), 4),
            total_change=round(float(y[-1] - y[0]), 4), percent_change=round(float(pct), 1),
            velocity=round(slope * 7.0, 5), acceleration=round(acc, 6),
            confidence=round(float(r2 * min(1.0, n / 5.0)), 3),
        ))
    return out


def _framing_comparable(pts: List[ObservationPoint]) -> bool:
    """False when photos were clearly taken at different zoom levels.

    Even fast-growing keloids don't more than double in visible area within a
    couple of months, so a bigger jump between two photos means the camera
    distance changed. A 4x change in the size of the box the patient drew
    around the scar is treated the same way."""
    for a, b in zip(pts, pts[1:]):
        days = abs((b.taken_at - a.taken_at).total_seconds()) / 86400.0
        aa, ab = a.vector.surface_area, b.vector.surface_area
        faded = a.vector.raw.get("faded") or b.vector.raw.get("faded")
        if not faded and aa > 0 and ab > 0 and days < 60 and max(aa, ab) / min(aa, ab) > 2.0:
            return False
        fa, fb = a.vector.raw.get("roi_frac", 0.0), b.vector.raw.get("roi_frac", 0.0)
        if fa > 0 and fb > 0 and max(fa, fb) / min(fa, fb) > 4.0:
            return False
    return True


# --------------------------------------------------------------------------- #
def analyze(points: List[ObservationPoint], scar_age_days: Optional[float] = None) -> TrajectoryResult:
    pts = sorted([p for p in points if p.reliable], key=lambda p: p.taken_at)
    n = len(pts)
    res = TrajectoryResult(status="baseline", n_observations=n)
    if n == 0:
        return res

    latest = pts[-1]
    if latest.report.growing_beyond_boundary:
        res.alerts.append(Alert("warning", "patient_reported_growth",
                                "You reported that the scar is spreading beyond the original wound. "
                                "Scars that grow past their original edge should be checked by a dermatologist."))
    if n == 1:
        if latest.severity >= 60:
            res.alerts.append(Alert("warning", "high_baseline",
                                    "This scar already looks quite active. Consider showing it to a doctor."))
        res.alerts.append(Alert("info", "need_more_photos",
                                "Add another photo in 1–2 weeks to start tracking how this scar is changing."))
        return res

    x = _x_axis([p.taken_at for p in pts])
    sev = np.array([p.severity for p in pts], dtype=float)
    sev_fit, sev_slope, _, sev_r2 = _fit(x, sev)
    recent = float(sev[-1] - sev[-2])
    res.severity_change = round(float(sev[-1] - sev[0]), 1)
    res.recent_change = round(recent, 1)
    res.severity_velocity = round(sev_slope * 7.0, 2)
    res.dimension_stats = _dimension_stats(x, [p.vector for p in pts])
    stats: Dict[str, DimensionStat] = {s.dimension: s for s in res.dimension_stats}

    # --- size -------------------------------------------------------------- #
    # Apparent size depends on how close the camera was. When the patient marks
    # the scar, the size of the marked box tells us roughly how zoomed-in each
    # photo is; if that changes a lot (close-up vs whole face), size can't be
    # compared, but a scar that has faded to nothing still counts.
    size_comparable = _framing_comparable(pts)
    if not size_comparable:
        for st in res.dimension_stats:
            if st.dimension == "surface_area" and not pts[-1].vector.raw.get("faded"):
                st.trend, st.effect = "stable", "no_change"
        stats = {s.dimension: s for s in res.dimension_stats}
    area = np.array([p.vector.surface_area for p in pts], dtype=float)
    a0 = max(area[0], 1e-6)
    area_fit, _, area_acc, area_r2 = _fit(x, area)
    area_growth = float(area_fit / a0) if n >= 3 else float((area[-1] - area[0]) / a0)
    if not size_comparable:
        area_growth, area_acc, area_r2 = 0.0, 0.0, 0.0
    res.area_growth = round(area_growth, 3)
    res.area_accelerating = bool(n >= 3 and area_acc > 0 and area_growth > 0.15
                                 and (area[-1] - area[-2]) > (area[1] - area[0]))

    thr = max(4.0, 0.10 * max(sev[0], sev[-1]))
    vasc = stats["vascularity"]
    colour_worse = vasc.effect == "worse" or stats["color_delta_e"].effect == "worse"
    colour_better = vasc.effect == "better" or stats["color_delta_e"].effect == "better"

    # --- 1. worsening (checked first) ---------------------------------------- #
    reasons: List[str] = []
    if sev_fit >= thr:
        reasons.append("overall_severity_rising")
    if recent >= max(5.0, 0.8 * thr):
        reasons.append("recent_jump")
    # Apparent size depends on camera distance, so size growth on its own needs
    # stronger and more consistent evidence than size growth + colour worsening.
    if colour_worse:
        size_confirmed = area_growth >= (0.20 if n >= 3 else 0.30)
    else:
        size_confirmed = (area_growth >= 0.55) or (n >= 3 and area_growth >= 0.35 and area_r2 >= 0.75)
    if size_confirmed and vasc.effect != "better":
        reasons.append("size_growing")
    elif area_growth >= 0.25 and not reasons:
        res.alerts.append(Alert("info", "possible_growth",
                                "The scar may look a little bigger, but this can be caused by taking the photo "
                                "from a different distance. Keep the same distance next time so we can confirm."))
    if latest.report.growing_beyond_boundary:
        reasons.append("patient_reported_growth")
    worse_dims = [s for s in res.dimension_stats if s.effect == "worse"
                  and s.dimension in ("vascularity", "color_delta_e", "surface_area", "texture_regularity")]
    better_dims = [s for s in res.dimension_stats if s.effect == "better"
                   and s.dimension in ("vascularity", "color_delta_e", "surface_area", "texture_regularity")]
    if len(worse_dims) >= 2 and len(better_dims) == 0 and sev_fit > 0:
        reasons.append("multiple_features_worse")
    # new symptoms
    first = pts[0].report
    if (latest.report.painful and not first.painful) or (latest.report.feels_tight and not first.feels_tight):
        if sev_fit > -thr:
            reasons.append("new_symptoms")
    res.worsening_reasons = reasons

    age = scar_age_days
    if reasons:
        res.status = "worsening"
    elif sev_fit <= -thr and recent < 3.0 and area_growth < 0.30:
        res.status = "improving"
    elif colour_better and not colour_worse and sev_fit < 0 and recent < 3.0 and area_growth < 0.30:
        res.status = "improving"
    else:
        res.status = "stable"
        if age is not None and age > 180 and sev[-1] >= 40:
            res.status = "stagnating"

    # --- alerts ---------------------------------------------------------------- #
    if res.status == "worsening":
        if "recent_jump" in reasons and sev_fit < 0:
            res.alerts.append(Alert("warning", "reversal",
                                    "The scar was improving but has got worse since your last photo."))
        elif "recent_jump" in reasons and recent >= 10:
            res.alerts.append(Alert("warning", "rapid_worsening",
                                    "The scar looks noticeably worse than in your last photo."))
        if "size_growing" in reasons:
            res.alerts.append(Alert("warning", "size_growing",
                                    f"The scar looks about {round(area_growth * 100)}% larger than in your first photo."))
        if vasc.effect == "worse":
            res.alerts.append(Alert("warning", "redness_increasing",
                                    "Redness is increasing, which can mean the scar is still very active."))
        if "new_symptoms" in reasons:
            res.alerts.append(Alert("warning", "new_symptoms",
                                    "You reported new pain or tightness since you started tracking."))
    if res.area_accelerating:
        res.alerts.append(Alert("warning", "accelerating_growth",
                                "The scar's growth is speeding up. Growth that speeds up rather than slows down "
                                "is an early warning sign of a keloid."))
    if age is not None and age > 180 and pts[-1].vector.vascularity >= 1.8:
        res.alerts.append(Alert("warning", "sustained_redness",
                                "The scar is still very red more than 6 months after the injury. "
                                "Most scars fade by this stage."))
    if age is not None and age > 90 and area_growth > 0.15 and res.status != "worsening":
        res.alerts.append(Alert("warning", "late_growth",
                                "The scar is still getting bigger more than 3 months after the injury."))
    if not size_comparable:
        res.alerts.append(Alert("info", "framing_changed",
                                "Your photos were taken from different distances, so we couldn't compare the scar's "
                                "size. Try to match the framing of your earlier photos next time."))
    if res.status == "stagnating":
        res.alerts.append(Alert("info", "stagnating",
                                "The scar hasn't changed much recently. At this stage, treatment can still help."))
    if res.status == "improving":
        res.alerts.append(Alert("positive", "improving",
                                "The scar is getting less noticeable over time. Keep up your current care."))
    return res
