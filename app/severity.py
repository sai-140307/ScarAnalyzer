"""Composite 0-100 severity score.

Weights follow the relative emphasis of VSS/POSAS observer items: colour
(vascularity + pigmentation + overall colour difference) carries the most
weight, followed by height, texture and pliability, plus patient symptoms.
"""
from __future__ import annotations

from typing import Tuple

from .schemas import ScarStateVector, SelfReport

WEIGHTS = {
    "vascularity": 0.24,
    "pigmentation": 0.12,
    "color_delta_e": 0.14,
    "height": 0.16,
    "pliability": 0.10,
    "texture": 0.12,
    "symptoms": 0.06,
    "growth": 0.06,
}

TYPE_MODIFIER = {
    "keloid": 1.15,
    "contracture": 1.10,
    "hypertrophic": 1.05,
    "atrophic": 1.00,
    "normal": 1.00,
}


def _c(x: float) -> float:
    return max(0.0, min(1.0, x))


def base_severity(v: ScarStateVector, r: SelfReport) -> float:
    comp = {
        "vascularity": _c(v.vascularity / 3.0),
        "pigmentation": _c(v.pigmentation / 3.0),
        "color_delta_e": _c(v.color_delta_e / 25.0),
        "height": _c(v.height / 3.0),
        "pliability": _c(v.pliability / 5.0),
        "texture": _c((1.0 - v.texture_regularity) / 0.5),
        "symptoms": 0.5 * float(r.itchy) + 0.5 * float(r.painful),
        "growth": float(r.growing_beyond_boundary),
    }
    # A sunken scar is also a visible irregularity even though it is not raised.
    if r.is_depressed:
        comp["height"] = max(comp["height"], 0.35)
    return 100.0 * sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)


def score(v: ScarStateVector, r: SelfReport, scar_type: str) -> float:
    return round(min(100.0, base_severity(v, r) * TYPE_MODIFIER.get(scar_type, 1.0)), 1)


def grade(sev: float) -> str:
    if sev < 20:
        return "minimal"
    if sev < 40:
        return "mild"
    if sev < 60:
        return "moderate"
    return "severe"


def grade_and_score(v: ScarStateVector, r: SelfReport, scar_type: str) -> Tuple[float, str]:
    s = score(v, r, scar_type)
    return s, grade(s)
