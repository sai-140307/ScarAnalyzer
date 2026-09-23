"""Accuracy under harsh, realistic photo variation (lighting, white balance,
angle, and +/-12% camera distance). Run: python tests/test_robustness.py"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.feature_extractor import extract_features  # noqa: E402
from app.pipeline import build_report  # noqa: E402
from tests.synthetic import ScarSpec, make_photo  # noqa: E402
from tests.test_scenarios import roi_for  # noqa: E402

N = int(os.getenv("ROBUST_N", "15"))


def series(specs, seed, strength=2.0, dist=0.12):
    rng = np.random.default_rng(seed)
    obs = []
    for i, s in enumerate(specs):
        sp = ScarSpec(**{**s.__dict__, "lighting": 1 + rng.uniform(-.12, .12) * strength,
                         "wb_shift": rng.uniform(-2.5, 2.5) * strength,
                         "size": s.size * (1 + rng.uniform(-dist, dist)),
                         "angle": s.angle + rng.uniform(-15, 15), "seed": int(rng.integers(1_000_000))})
        v, q, _ = extract_features(make_photo(sp), None, roi_for(s))
        obs.append({"id": str(i), "taken_at": (datetime(2026, 2, 1) + timedelta(days=14 * i)).isoformat(),
                    "vector": v.to_json(), "report": {}, "quality": {"scar_found": q.scar_found}})
    return build_report({"id": "s", "body_region": "arm", "date_of_injury": "2026-01-01"}, obs)


def lerp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


CASES = [
    ("stable, 4 photos", [ScarSpec(redness=10, darkness=4, size=.18)] * 4, "stable"),
    ("stable, 2 photos", [ScarSpec(redness=10, darkness=4, size=.18)] * 2, "stable"),
    ("mild worsening, 4 photos", [ScarSpec(redness=x, darkness=4, size=.18) for x in lerp(9, 13, 4)], "worsening"),
    ("mild worsening, 2 photos", [ScarSpec(redness=9, darkness=4, size=.18), ScarSpec(redness=13, darkness=5, size=.18)], "worsening"),
    ("mild improvement, 4 photos", [ScarSpec(redness=x, darkness=4, size=.18) for x in lerp(14, 9, 4)], "improving"),
]


def accuracy():
    out = {}
    for name, specs, want in CASES:
        c = Counter(series(specs, s)["status"] for s in range(N))
        out[name] = (c[want] / N, dict(c))
    return out


def test_accuracy_above_90_percent():
    for name, (acc, dist) in accuracy().items():
        assert acc >= 0.9, (name, dist)


if __name__ == "__main__":
    ok = True
    for name, (acc, dist) in accuracy().items():
        print(f"{name:30s} {acc:5.0%}  {dist}")
        ok &= acc >= 0.9
    sys.exit(0 if ok else 1)
