"""Orchestrates features -> classification -> severity -> trajectory -> suggestions.

The report is always rebuilt from ALL stored observations, so it can never go
stale (e.g. after deleting a photo or uploading an older one).
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import List, Optional

from . import classifier, severity, suggestions, trajectory
from .schemas import ScarStateVector, SelfReport

DISCLAIMER = ("ScarAnalyzer gives general information to help you track your scar. "
              "It is not a diagnosis. Talk to a doctor about any concerns.")

STATUS_HEADLINE = {
    "baseline": "First photo saved",
    "improving": "Healing well",
    "stable": "No major change",
    "stagnating": "Healing has slowed",
    "worsening": "Getting worse",
}


def _parse_date(s) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    if isinstance(s, date):
        return datetime(s.year, s.month, s.day)
    try:
        return datetime.fromisoformat(str(s)[:19])
    except ValueError:
        return None


def build_report(scar: dict, observations: List[dict]) -> dict:
    """scar: stored scar row. observations: stored rows with vector/report/quality dicts."""
    obs = sorted(observations, key=lambda o: o["taken_at"])
    base = {
        "scar_id": scar["id"], "nickname": scar.get("nickname"), "body_region": scar.get("body_region"),
        "cause": scar.get("cause"), "date_of_injury": scar.get("date_of_injury"),
        "observation_count": len(obs), "disclaimer": DISCLAIMER,
    }
    if not obs:
        return {**base, "status": "empty", "headline": "No photos yet", "alerts": [], "changes": [],
                "severity_timeline": [], "suggestions": [], "summary": "Add a photo to start tracking this scar.",
                "needs_professional_review": False, "current_severity": None, "clinical_grade": None,
                "scar_type": None, "classification_confidence": None, "trend": "baseline",
                "dimension_stats": [], "photo_tips": [], "scar_age_days": None}

    injury = _parse_date(scar.get("date_of_injury"))
    first_t = _parse_date(obs[0]["taken_at"])
    last_t = _parse_date(obs[-1]["taken_at"])
    start = injury or first_t
    age_days = max(0.0, (last_t - start).total_seconds() / 86400.0) if (start and last_t) else None

    points = []
    for o in obs:
        v = ScarStateVector.from_json(o["vector"])
        r = SelfReport.from_dict(o.get("report"))
        points.append(trajectory.ObservationPoint(
            taken_at=_parse_date(o["taken_at"]), vector=v, report=r,
            severity=severity.base_severity(v, r),
            reliable=bool((o.get("quality") or {}).get("scar_found", True)),
            observation_id=o["id"]))

    traj = trajectory.analyze(points, scar_age_days=age_days)

    reliable = [p for p in points if p.reliable] or points
    latest = reliable[-1]
    vasc_stat = next((s for s in traj.dimension_stats if s.dimension == "vascularity"), None)
    t = classifier.TemporalSignals(
        scar_age_days=age_days, area_growth=traj.area_growth if traj.n_observations >= 2 else 0.0,
        area_accelerating=traj.area_accelerating,
        vascularity_trend=vasc_stat.total_change if vasc_stat else 0.0,
        n_observations=traj.n_observations)
    cls = classifier.classify(latest.vector, latest.report, scar.get("body_region") or "", t)
    modifier = severity.TYPE_MODIFIER.get(cls.scar_type, 1.0)
    current = round(min(100.0, latest.severity * modifier), 1)
    grade = severity.grade(current)

    needs_review = bool(
        cls.needs_review
        or (traj.status == "worsening" and current >= 40)
        or any(a.code in ("accelerating_growth", "patient_reported_growth", "sustained_redness") for a in traj.alerts)
        or current >= 70
    )

    sugg = suggestions.suggest(latest.vector, latest.report, cls.scar_type, traj.status, current,
                               age_days, scar.get("cause") or "", needs_review)

    changes = []
    for s in traj.dimension_stats:
        if s.dimension in ("vascularity", "color_delta_e", "surface_area", "texture_regularity", "pigmentation"):
            changes.append({"key": s.dimension, "label": s.label, "effect": s.effect,
                            "percent_change": s.percent_change, "trend": s.trend})

    photo_tips = []
    last_q = obs[-1].get("quality") or {}
    photo_tips.extend(last_q.get("warnings", []))

    timeline = [{
        "observation_id": p.observation_id,
        "timestamp": p.taken_at.isoformat(),
        "composite_score": round(min(100.0, p.severity * modifier), 1),
        "reliable": p.reliable,
    } for p in points]

    report = {
        **base,
        "scar_age_days": None if age_days is None else round(age_days),
        "status": traj.status,
        "trend": traj.status,
        "headline": STATUS_HEADLINE.get(traj.status, traj.status),
        "current_severity": current,
        "clinical_grade": grade,
        "scar_type": cls.scar_type,
        "classification_confidence": cls.confidence,
        "severity_change": traj.severity_change,
        "recent_change": traj.recent_change,
        "severity_timeline": timeline,
        "changes": changes if traj.n_observations >= 2 else [],
        "dimension_stats": [asdict(s) for s in traj.dimension_stats],
        "alerts": [asdict(a) for a in traj.alerts],
        "worsening_reasons": traj.worsening_reasons,
        "needs_professional_review": needs_review,
        "suggestions": [s.to_dict() for s in sugg],
        "photo_tips": photo_tips,
    }
    report["summary"] = summarize(report)
    return report


def summarize(r: dict) -> str:
    n = r["observation_count"]
    status = r["status"]
    parts = []
    if status == "baseline":
        parts.append("We've saved your first photo as a starting point.")
        parts.append("Take the next photo in 1–2 weeks, in similar light and from a similar distance, "
                     "so we can tell you how the scar is changing.")
    else:
        parts.append(f"Based on {n} photos, your scar is ")
        parts[-1] += {
            "improving": "healing well.",
            "stable": "about the same as before.",
            "stagnating": "not changing much anymore, even though it still looks active.",
            "worsening": "getting worse.",
        }.get(status, "being tracked.")
        worse = [c["label"].lower() for c in r["changes"] if c["effect"] == "worse"]
        better = [c["label"].lower() for c in r["changes"] if c["effect"] == "better"]
        if worse:
            parts.append("It has become worse in: " + ", ".join(worse) + ".")
        if better:
            parts.append("It has improved in: " + ", ".join(better) + ".")
    if r["needs_professional_review"]:
        parts.append("We recommend showing this scar to a dermatologist soon.")
    elif status == "improving":
        parts.append("Keep going with your current care and keep taking photos every 1–2 weeks.")
    return " ".join(parts)
