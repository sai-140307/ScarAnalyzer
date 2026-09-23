"""Weighted rule-based scar type classifier.

Kept deliberately interpretable. It exposes the same `classify()` interface a
trained model would, so a learned classifier can replace it later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .schemas import ScarStateVector, SelfReport

TYPES = ("normal", "hypertrophic", "keloid", "atrophic", "contracture")
JOINT_REGIONS = {"hand", "elbow", "knee", "neck", "shoulder", "ankle", "wrist", "finger", "joint"}


@dataclass
class TemporalSignals:
    scar_age_days: Optional[float] = None
    area_growth: float = 0.0              # fitted relative change in size (0.3 = +30%)
    area_accelerating: bool = False
    vascularity_trend: float = 0.0        # fitted change over the series
    n_observations: int = 1


@dataclass
class Classification:
    scar_type: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    needs_review: bool = False


def _softmax(scores: Dict[str, float], temp: float = 0.8) -> Dict[str, float]:
    m = max(scores.values())
    ex = {k: math.exp((v - m) / temp) for k, v in scores.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def classify(v: ScarStateVector, r: SelfReport, body_region: str = "",
             t: Optional[TemporalSignals] = None) -> Classification:
    t = t or TemporalSignals()
    age = t.scar_age_days if t.scar_age_days is not None else 0.0
    rough = max(0.0, 1.0 - v.texture_regularity)
    joint = 1.0 if body_region.lower() in JOINT_REGIONS else 0.0
    dep = float(r.is_depressed)
    tight = float(r.feels_tight)
    grow = float(r.growing_beyond_boundary)

    s = {
        "normal": 1.6 - 0.45 * v.vascularity - 0.7 * v.height - 1.2 * tight - 1.8 * dep
                  - 2.0 * grow - 3.0 * max(0.0, t.area_growth - 0.1),
        "hypertrophic": -0.4 + 0.75 * v.height + 0.35 * v.vascularity + 1.2 * rough
                        - 1.5 * dep + (0.6 if v.height >= 1 and v.vascularity >= 1.2 else 0.0),
        "keloid": -1.4 + 2.6 * grow + 4.0 * max(0.0, t.area_growth - 0.1)
                  + (0.8 if t.area_accelerating else 0.0)
                  + (0.8 if age > 180 and v.vascularity >= 1.8 else 0.0)
                  + (0.6 if age > 90 and t.area_growth > 0.15 else 0.0)
                  + 0.35 * v.height - 1.5 * dep,
        "atrophic": -1.0 + 3.2 * dep - 0.6 * v.height,
        "contracture": -1.2 + 2.2 * tight + 0.9 * joint * tight + 0.2 * v.pliability,
    }
    probs = _softmax(s)
    top = max(probs, key=probs.get)
    needs_review = top in ("keloid", "contracture") or (top == "hypertrophic" and probs[top] > 0.6 and age > 365)
    return Classification(scar_type=top, confidence=round(probs[top], 3),
                          probabilities={k: round(p, 3) for k, p in probs.items()},
                          needs_review=needs_review)
