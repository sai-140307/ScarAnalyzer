"""Evidence-ranked care suggestions (categories, never specific products)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

from .schemas import ScarStateVector, SelfReport

EVIDENCE_WEIGHT = {"strong": 1.0, "moderate": 0.8, "consensus": 0.65}

CATALOG = {
    "silicone": dict(
        category="Silicone gel or sheets", evidence="strong", professional_required=False,
        description="Silicone keeps the scar hydrated and helps it flatten and fade.",
        timing="Every day, for at least 2–3 months"),
    "corticosteroid": dict(
        category="Steroid injections", evidence="strong", professional_required=True,
        description="A doctor injects medicine into the scar to flatten raised or growing tissue.",
        timing="Every 4–6 weeks, given by a dermatologist"),
    "pressure": dict(
        category="Pressure garments", evidence="strong", professional_required=False,
        description="Firm, constant pressure limits thickening, especially for burn scars.",
        timing="Most of the day, for several months"),
    "laser": dict(
        category="Laser treatment", evidence="moderate", professional_required=True,
        description="Lasers can reduce redness and improve texture once the wound has closed.",
        timing="A few sessions, several weeks apart"),
    "microneedling": dict(
        category="Microneedling", evidence="moderate", professional_required=True,
        description="Tiny controlled punctures help fill in sunken scars.",
        timing="Sessions every 4–6 weeks"),
    "massage": dict(
        category="Scar massage", evidence="moderate", professional_required=False,
        description="Gentle firm massage softens tight scar tissue and improves movement.",
        timing="5–10 minutes, 2–3 times a day"),
    "sun": dict(
        category="Sun protection", evidence="consensus", professional_required=False,
        description="Sunlight can permanently darken a healing scar. Cover it or use SPF 30+.",
        timing="Every day while the scar is still pink or red"),
    "moisturise": dict(
        category="Moisturising", evidence="consensus", professional_required=False,
        description="Keeping the scar moisturised reduces itching and dryness.",
        timing="2–3 times a day"),
    "surgical": dict(
        category="Surgical scar revision", evidence="consensus", professional_required=True,
        description="A surgeon can reshape or release scars that stay thick or limit movement.",
        timing="Usually only after the scar is at least a year old"),
    "see_doctor": dict(
        category="See a dermatologist", evidence="strong", professional_required=True,
        description="Take your photos and this report with you. Early treatment works best.",
        timing="Book an appointment in the next few weeks"),
}


@dataclass
class Suggestion:
    key: str
    category: str
    description: str
    reason: str
    timing: str
    evidence: str
    professional_required: bool
    relevance: float

    def to_dict(self) -> dict:
        return asdict(self)


def suggest(v: ScarStateVector, r: SelfReport, scar_type: str, status: str,
            severity: float, age_days: Optional[float], cause: str = "",
            needs_review: bool = False, limit: int = 5) -> List[Suggestion]:
    age = age_days if age_days is not None else 60.0
    cause = (cause or "").lower()
    raised = v.height >= 1 or scar_type in ("hypertrophic", "keloid")
    bad = status in ("worsening", "stagnating")
    red = v.vascularity >= 1.2
    cand = {}

    def add(key, rel, reason):
        if rel <= 0:
            return
        cand[key] = max(cand.get(key, (0, ""))[0], rel), reason

    if needs_review or status == "worsening" or scar_type == "keloid" or severity >= 65:
        why = ("Your scar is getting worse, so it's worth having it checked." if status == "worsening"
               else "Some signs in your photos are worth showing to a specialist.")
        add("see_doctor", 1.2, why)

    if age >= 14:
        add("silicone", 0.95 if (raised or red) else 0.6,
            "Works well for scars that are red or raised." if (raised or red)
            else "A simple first step that helps most healing scars.")
    if scar_type in ("keloid", "hypertrophic") and (bad or scar_type == "keloid" or v.height >= 2):
        add("corticosteroid", 0.95 if bad else 0.75,
            "Recommended when a raised scar is growing or not flattening on its own.")
    if raised and ("burn" in cause or scar_type == "hypertrophic"):
        add("pressure", 0.8 if "burn" in cause else 0.55,
            "Helps stop raised scars from getting thicker." + (" Especially useful for burns." if "burn" in cause else ""))
    if red and age >= 30:
        add("laser", 0.7 if bad else 0.5, "Can speed up fading of redness that is lasting a while.")
    if scar_type == "atrophic" or r.is_depressed:
        add("microneedling", 0.9, "Designed for scars that are sunken below the skin.")
    if (r.feels_tight or scar_type == "contracture") and age >= 21:
        add("massage", 0.9, "Loosens tight scar tissue and helps you move more freely.")
    elif age >= 21 and scar_type in ("normal", "hypertrophic"):
        add("massage", 0.45, "Gentle massage can help the scar soften and flatten.")
    if age < 365 or v.pigmentation >= 1 or red:
        add("sun", 0.7, "Your scar is still changing colour, and sun can darken it for good.")
    add("moisturise", 0.75 if r.itchy else 0.4,
        "Helps with the itching you mentioned." if r.itchy else "Keeps the skin comfortable while it heals.")
    if age >= 365 and (severity >= 55 or scar_type == "contracture"):
        add("surgical", 0.6, "An option if the scar is still thick or restricting movement after a year.")

    out = []
    for key, (rel, reason) in cand.items():
        c = CATALOG[key]
        score = min(1.0, rel * EVIDENCE_WEIGHT[c["evidence"]]) if key != "see_doctor" else 1.0
        out.append(Suggestion(key=key, category=c["category"], description=c["description"], reason=reason,
                              timing=c["timing"], evidence=c["evidence"],
                              professional_required=c["professional_required"], relevance=round(score, 2)))
    out.sort(key=lambda s: s.relevance, reverse=True)
    return out[:limit]
