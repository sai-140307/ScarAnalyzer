"""End-to-end validation on synthetic photo series with known outcomes.

Run:  python -m pytest tests -q      (or)      python tests/test_scenarios.py
Each photo gets random lighting / white-balance / angle changes, so the suite
also checks that photo conditions are not mistaken for healing or worsening.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.feature_extractor import extract_features  # noqa: E402
from app.pipeline import build_report  # noqa: E402
from app.schemas import SelfReport  # noqa: E402
from tests.synthetic import ScarSpec, jitter, make_photo  # noqa: E402


def roi_for(spec, W=640, H=480):
    """The box a patient would draw: the scar plus a margin of normal skin."""
    half = 1.6 * spec.size * W
    x0, y0 = max(0.0, (W / 2 - half) / W), max(0.0, (H / 2 - half) / H)
    return (x0, y0, min(1 - x0, 2 * half / W), min(1 - y0, 2 * half / H))


def run_series(specs, reports=None, start_days_after_injury=30, spacing_days=14, seed=0,
               region="forearm", cause="surgical", shuffle_upload=False, same_day=False, marked=True):
    rng = np.random.default_rng(seed)
    injury = datetime(2026, 1, 1)
    obs = []
    for i, spec in enumerate(specs):
        rep = (reports[i] if reports else None) or SelfReport()
        v, q, _ = extract_features(make_photo(jitter(spec, rng)), rep, roi_for(spec) if marked else None)
        taken = injury + timedelta(days=start_days_after_injury + (0 if same_day else i * spacing_days),
                                   minutes=i)
        obs.append({"id": f"o{i}", "taken_at": taken.isoformat(), "vector": v.to_json(),
                    "report": rep.to_dict(), "quality": {"scar_found": q.scar_found, "warnings": q.warnings}})
    if shuffle_upload:
        rng.shuffle(obs)
    scar = {"id": "s1", "body_region": region, "cause": cause, "date_of_injury": injury.date().isoformat()}
    return build_report(scar, obs)


def lerp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# --------------------------------------------------------------------------- #
def test_normal_healing_is_improving():
    specs = [ScarSpec(redness=r, darkness=d, size=0.18, roughness=ro)
             for r, d, ro in zip(lerp(16, 5, 6), lerp(6, 2, 6), lerp(0.6, 0.1, 6))]
    r = run_series(specs)
    assert r["status"] == "improving", r["status"]
    assert not r["needs_professional_review"]


def test_redder_and_bigger_is_worsening():
    specs = [ScarSpec(redness=r, darkness=4, size=s)
             for r, s in zip(lerp(9, 17, 5), lerp(0.15, 0.22, 5))]
    r = run_series(specs)
    assert r["status"] == "worsening", r
    assert any(c["key"] == "vascularity" and c["effect"] == "worse" for c in r["changes"])


def test_only_redness_increasing_is_worsening():
    specs = [ScarSpec(redness=x, darkness=4, size=0.18) for x in lerp(8, 14, 4)]
    r = run_series(specs)
    assert r["status"] == "worsening", r["status"]


def test_two_photos_second_worse():
    # The most common real-world case: patient uploads two photos, the second is worse.
    specs = [ScarSpec(redness=9, darkness=3, size=0.16), ScarSpec(redness=15, darkness=6, size=0.20, roughness=0.5)]
    r = run_series(specs)
    assert r["status"] == "worsening", r["status"]


def test_two_photos_second_better():
    specs = [ScarSpec(redness=15, darkness=6, size=0.18, roughness=0.5), ScarSpec(redness=8, darkness=3, size=0.18)]
    r = run_series(specs)
    assert r["status"] == "improving", r["status"]


def test_improving_then_reversal_is_worsening():
    reds = [16, 13, 10, 8, 14]
    specs = [ScarSpec(redness=x, darkness=4, size=0.18) for x in reds]
    r = run_series(specs)
    assert r["status"] == "worsening", r["status"]
    assert any(a["code"] == "reversal" for a in r["alerts"]), r["alerts"]


def test_keloid_growth_flags_review():
    specs = [ScarSpec(redness=14, darkness=5, size=s, roughness=0.6) for s in [0.12, 0.13, 0.15, 0.19, 0.25]]
    reps = [SelfReport(height_level=2)] * 5
    r = run_series(specs, reps, start_days_after_injury=100, region="chest")
    assert r["status"] == "worsening", r["status"]
    assert r["needs_professional_review"]
    assert r["scar_type"] == "keloid", r["scar_type"]
    assert any(s["key"] == "see_doctor" for s in r["suggestions"])


def test_stable_mature_scar_under_varied_lighting():
    specs = [ScarSpec(redness=6, darkness=2, size=0.18) for _ in range(5)]
    r = run_series(specs, seed=7)
    assert r["status"] in ("stable", "improving"), r["status"]
    assert r["status"] != "worsening"


def test_stable_across_skin_tones():
    for skin in ("light", "medium", "dark"):
        specs = [ScarSpec(redness=10, darkness=4, size=0.18, skin=skin) for _ in range(4)]
        r = run_series(specs, seed=3)
        assert r["status"] == "stable", (skin, r["status"])


def test_worsening_detected_on_dark_skin():
    specs = [ScarSpec(redness=x, darkness=d, size=0.18, skin="dark") for x, d in zip(lerp(7, 14, 4), lerp(3, 8, 4))]
    r = run_series(specs)
    assert r["status"] == "worsening", r["status"]


def test_upload_order_does_not_matter():
    # Photos uploaded out of order (e.g. old gallery photos added later).
    specs = [ScarSpec(redness=x, darkness=4, size=0.18) for x in lerp(8, 15, 5)]
    r = run_series(specs, shuffle_upload=True)
    assert r["status"] == "worsening", r["status"]


def test_same_day_uploads_still_detect_trend():
    specs = [ScarSpec(redness=x, darkness=4, size=0.18) for x in lerp(8, 15, 4)]
    r = run_series(specs, same_day=True)
    assert r["status"] == "worsening", r["status"]


def test_stagnating_hypertrophic():
    specs = [ScarSpec(redness=17, darkness=8, size=0.2, roughness=0.8) for _ in range(4)]
    reps = [SelfReport(height_level=2, itchy=True)] * 4
    r = run_series(specs, reps, start_days_after_injury=220)
    assert r["status"] == "stagnating", (r["status"], r["current_severity"])


def test_atrophic_classification():
    specs = [ScarSpec(redness=4, darkness=-6, size=0.12) for _ in range(3)]
    reps = [SelfReport(is_depressed=True, height_level=0)] * 3
    r = run_series(specs, reps)
    assert r["scar_type"] == "atrophic", r["scar_type"]
    assert any(s["key"] == "microneedling" for s in r["suggestions"])


def test_unmarked_photos_still_work():
    specs = [ScarSpec(redness=x, darkness=4, size=0.18) for x in lerp(8, 15, 4)]
    r = run_series(specs, marked=False)
    assert r["status"] == "worsening", r["status"]


def test_healed_scar_in_marked_area_is_improving():
    # The scar fades completely: nothing stands out inside the marked box any more.
    specs = [ScarSpec(redness=13, darkness=5, size=0.15), ScarSpec(redness=0.5, darkness=0.2, size=0.15)]
    r = run_series(specs)
    assert r["status"] == "improving", r["status"]


def test_zoom_change_is_not_growth():
    # Second photo taken much closer: the scar looks 3x bigger but is unchanged.
    specs = [ScarSpec(redness=10, darkness=4, size=0.10), ScarSpec(redness=10, darkness=4, size=0.18)]
    r = run_series(specs)
    assert r["status"] != "worsening", (r["status"], r["worsening_reasons"])
    assert any(a["code"] == "framing_changed" for a in r["alerts"])


def test_single_photo_baseline():
    r = run_series([ScarSpec(redness=12)])
    assert r["status"] == "baseline"
    assert r["current_severity"] is not None


def test_patient_reported_growth_is_worsening():
    specs = [ScarSpec(redness=10, darkness=4, size=0.18) for _ in range(3)]
    reps = [SelfReport(), SelfReport(), SelfReport(growing_beyond_boundary=True)]
    r = run_series(specs, reps)
    assert r["status"] == "worsening"
    assert r["needs_professional_review"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except AssertionError as e:  # noqa: PERF203
            print(f"FAIL  {name}: {str(e)[:300]}")
    print(f"\n{passed}/{len(tests)} scenarios passed")
    sys.exit(0 if passed == len(tests) else 1)
