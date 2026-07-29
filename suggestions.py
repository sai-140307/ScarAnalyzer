from dataclasses import dataclass
from typing import Dict, List, Optional

from schema import ScarType
from severity import SeverityBreakdown
from trajectory import TrajectoryStats


@dataclass
class Suggestion:
    """A single treatment category suggestion."""

    category: str
    relevance: float
    reason: str
    timing: str
    professional_required: bool
    description: str


@dataclass
class SuggestionReport:
    """Complete suggestion output for a scar."""

    suggestions: List[Suggestion]
    general_advice: List[str]
    disclaimer: str


class SuggestionEngine:
    """Generates treatment category suggestions based on scar analysis."""

    DISCLAIMER = (
        "These suggestions are informational only and do not constitute "
        "medical advice, diagnosis, or treatment recommendations. Always "
        "consult a qualified healthcare professional before starting any "
        "treatment. Individual results vary based on skin type, scar age, "
        "medical history, and other factors."
    )

    TREATMENTS = {
        "silicone_therapy": {
            "category": "Silicone Therapy",
            "description": (
                "Silicone sheets or gel applied to the scar surface. "
                "One of the most evidence-backed non-invasive options "
                "for reducing scar thickness and discoloration."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.KELOID,
                ScarType.FLAT_MATURE,
                ScarType.STRETCH_MARK,
            ],
            "severity_range": (5, 80),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["height", "vascularity", "color_deviation"],
        },
        "pressure_therapy": {
            "category": "Pressure Therapy",
            "description": (
                "Compression garments or bandages worn over the scar. "
                "Commonly used for burn scars and large hypertrophic scars."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.KELOID,
                ScarType.CONTRACTURE,
            ],
            "severity_range": (25, 100),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["height", "pliability", "surface_area"],
        },
        "massage_therapy": {
            "category": "Scar Massage",
            "description": (
                "Regular manual massage of the scar tissue to improve "
                "pliability and break down adhesions. Simple and free."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.CONTRACTURE,
                ScarType.FLAT_MATURE,
                ScarType.STRETCH_MARK,
            ],
            "severity_range": (5, 60),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["pliability", "texture"],
        },
        "sun_protection": {
            "category": "Sun Protection",
            "description": (
                "Consistent SPF 30+ sunscreen or physical coverage over "
                "the scar. UV exposure worsens pigmentation and delays healing."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.KELOID,
                ScarType.ATROPHIC,
                ScarType.FLAT_MATURE,
                ScarType.STRETCH_MARK,
                ScarType.CONTRACTURE,
            ],
            "severity_range": (0, 100),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["pigmentation", "color_deviation"],
        },
        "moisturization": {
            "category": "Moisturization",
            "description": (
                "Regular application of fragrance-free moisturizer to "
                "keep scar tissue hydrated and improve texture."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.ATROPHIC,
                ScarType.FLAT_MATURE,
                ScarType.STRETCH_MARK,
                ScarType.CONTRACTURE,
            ],
            "severity_range": (0, 50),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["texture", "pliability"],
        },
        "corticosteroid_injection": {
            "category": "Corticosteroid Injections",
            "description": (
                "Injections directly into the scar to reduce inflammation, "
                "flatten raised tissue, and relieve itching. Requires a "
                "dermatologist."
            ),
            "applicable_types": [ScarType.KELOID, ScarType.HYPERTROPHIC],
            "severity_range": (30, 100),
            "professional_required": True,
            "timing": "immediate",
            "drivers": ["height", "vascularity", "surface_area"],
        },
        "laser_treatment": {
            "category": "Laser Treatment",
            "description": (
                "Various laser therapies (pulsed dye, fractional CO2, etc.) "
                "to reduce redness, improve texture, or stimulate remodeling. "
                "Requires a specialist."
            ),
            "applicable_types": [
                ScarType.HYPERTROPHIC,
                ScarType.KELOID,
                ScarType.ATROPHIC,
                ScarType.STRETCH_MARK,
            ],
            "severity_range": (15, 90),
            "professional_required": True,
            "timing": "after_stabilization",
            "drivers": [
                "vascularity",
                "pigmentation",
                "texture",
                "color_deviation",
            ],
        },
        "microneedling": {
            "category": "Microneedling",
            "description": (
                "Controlled micro-injuries to stimulate collagen remodeling. "
                "Particularly effective for atrophic (depressed) scars. "
                "Professional sessions recommended."
            ),
            "applicable_types": [ScarType.ATROPHIC, ScarType.STRETCH_MARK],
            "severity_range": (10, 70),
            "professional_required": True,
            "timing": "after_stabilization",
            "drivers": ["texture", "height", "pigmentation"],
        },
        "surgical_revision": {
            "category": "Surgical Revision",
            "description": (
                "Surgical removal or modification of the scar. Considered "
                "for severe or functionally limiting scars that haven't "
                "responded to other treatments."
            ),
            "applicable_types": [ScarType.KELOID, ScarType.CONTRACTURE],
            "severity_range": (60, 100),
            "professional_required": True,
            "timing": "after_stabilization",
            "drivers": ["pliability", "surface_area", "height"],
        },
        "physical_therapy": {
            "category": "Physical Therapy",
            "description": (
                "Stretching and range-of-motion exercises for scars "
                "that limit movement. Important for contracture scars "
                "near joints."
            ),
            "applicable_types": [ScarType.CONTRACTURE],
            "severity_range": (20, 100),
            "professional_required": True,
            "timing": "immediate",
            "drivers": ["pliability"],
        },
        "topical_retinoids": {
            "category": "Topical Retinoids",
            "description": (
                "Prescription or OTC retinoid creams to improve skin "
                "turnover and texture. Can help with pigmentation and "
                "mild textural irregularities."
            ),
            "applicable_types": [
                ScarType.ATROPHIC,
                ScarType.FLAT_MATURE,
                ScarType.STRETCH_MARK,
            ],
            "severity_range": (5, 45),
            "professional_required": False,
            "timing": "ongoing",
            "drivers": ["pigmentation", "texture"],
        },
    }

    def suggest(
        self,
        scar_type: ScarType,
        severity: SeverityBreakdown,
        trajectory_stats: Optional[List[TrajectoryStats]] = None,
        alerts: Optional[List[dict]] = None,
    ) -> SuggestionReport:
        """Generate ranked suggestions based on current scar analysis."""
        candidates: List[Suggestion] = []

        for treatment in self.TREATMENTS.values():
            if scar_type not in treatment["applicable_types"]:
                continue

            sev_min, sev_max = treatment["severity_range"]
            if severity.composite_score < sev_min or severity.composite_score > sev_max:
                continue

            relevance = self._compute_relevance(
                treatment, scar_type, severity, trajectory_stats, alerts
            )
            if relevance <= 0.0:
                continue

            reason = self._generate_reason(treatment, severity, trajectory_stats)
            candidates.append(
                Suggestion(
                    category=treatment["category"],
                    relevance=relevance,
                    reason=reason,
                    timing=treatment["timing"],
                    professional_required=treatment["professional_required"],
                    description=treatment["description"],
                )
            )

        candidates.sort(key=lambda s: s.relevance, reverse=True)
        top_suggestions = candidates[:5]
        general_advice = self._generate_general_advice(
            scar_type, severity, trajectory_stats, alerts
        )

        return SuggestionReport(
            suggestions=top_suggestions,
            general_advice=general_advice,
            disclaimer=self.DISCLAIMER,
        )

    def _compute_relevance(
        self,
        treatment: dict,
        scar_type: ScarType,
        severity: SeverityBreakdown,
        trajectory_stats: Optional[List[TrajectoryStats]],
        alerts: Optional[List[dict]],
    ) -> float:
        """Compute how relevant a treatment is to this specific scar."""
        score = 0.0
        sev_min, sev_max = treatment["severity_range"]
        sev_range = sev_max - sev_min
        if sev_range > 0:
            center = (sev_min + sev_max) / 2.0
            distance = abs(severity.composite_score - center) / (sev_range / 2.0)
            score += max(0.0, 1.0 - distance) * 0.3

        contributions = severity.dimension_contributions
        driver_dims = treatment.get("drivers", [])
        for dim in driver_dims:
            contrib = contributions.get(dim, 0.0)
            if contrib > 0.15:
                score += 0.15
            elif contrib > 0.1:
                score += 0.08

        if alerts:
            for alert in alerts:
                alert_dim = alert.get("dimension", "")
                if alert_dim in driver_dims:
                    score += 0.15
                if alert.get("type") == "accelerating_growth" and "surface_area" in driver_dims:
                    score += 0.2

        if trajectory_stats:
            for stat in trajectory_stats:
                if stat.dimension in driver_dims and stat.trend == "worsening":
                    score += 0.1

        if not treatment["professional_required"]:
            score += 0.05

        return min(1.0, score)

    def _generate_reason(
        self,
        treatment: dict,
        severity: SeverityBreakdown,
        trajectory_stats: Optional[List[TrajectoryStats]],
    ) -> str:
        """Generate a specific reason why this treatment is suggested."""
        drivers = treatment.get("drivers", [])
        contributions = severity.dimension_contributions
        reasons: List[str] = []

        active_drivers = []
        for dim in drivers:
            contrib = contributions.get(dim, 0.0)
            if contrib > 0.1:
                active_drivers.append(dim)

        if active_drivers:
            dim_names = {
                "height": "elevation",
                "vascularity": "redness",
                "pigmentation": "discoloration",
                "pliability": "tightness",
                "texture": "texture irregularity",
                "surface_area": "scar size",
                "color_deviation": "color difference from surrounding skin",
            }
            named = [dim_names.get(d, d) for d in active_drivers[:2]]
            reasons.append("Your scar shows notable {}".format(" and ".join(named)))

        if trajectory_stats:
            worsening = [
                s.dimension for s in trajectory_stats
                if s.dimension in drivers and s.trend == "worsening"
            ]
            if worsening:
                dim_names_short = {
                    "height": "elevation",
                    "vascularity": "redness",
                    "surface_area": "size",
                    "pliability": "tightness",
                }
                named_w = [dim_names_short.get(d, d) for d in worsening[:2]]
                reasons.append(
                    "{} trending upward".format(" and ".join(named_w)).capitalize()
                )

        if not reasons:
            reasons.append(
                "Generally recommended for this type of scar in this severity range"
            )

        return ". ".join(reasons) + "."

    def _generate_general_advice(
        self,
        scar_type: ScarType,
        severity: SeverityBreakdown,
        trajectory_stats: Optional[List[TrajectoryStats]],
        alerts: Optional[List[dict]],
    ) -> List[str]:
        """Generate general care advice based on the scar's current state."""
        advice: List[str] = []
        advice.append(
            "Continue tracking your scar regularly — consistent photos in "
            "similar lighting help detect changes early."
        )

        if severity.composite_score > 50:
            advice.append(
                "Your scar severity is in the moderate-to-severe range. "
                "Consider scheduling a dermatology consultation to discuss "
                "treatment options."
            )
        elif severity.composite_score > 25:
            advice.append(
                "Your scar is in the mild-to-moderate range. Many scars "
                "in this range respond well to consistent home care."
            )

        if trajectory_stats:
            worsening_count = sum(1 for s in trajectory_stats if s.trend == "worsening")
            if worsening_count >= 2:
                advice.append(
                    "Multiple dimensions are trending in the wrong direction. "
                    "If this continues at your next observation, professional "
                    "evaluation is recommended."
                )
            improving_count = sum(1 for s in trajectory_stats if s.trend == "improving")
            if improving_count >= 3:
                advice.append(
                    "Good news — multiple aspects of your scar are improving. "
                    "Continue what you're doing."
                )

        if scar_type == ScarType.KELOID:
            advice.append(
                "Keloid scars can recur after treatment. Long-term monitoring "
                "and follow-up with a dermatologist is especially important."
            )
        elif scar_type == ScarType.CONTRACTURE:
            advice.append(
                "Contracture scars can limit range of motion over time. "
                "Regular stretching and physical therapy can help maintain mobility."
            )
        elif scar_type == ScarType.ATROPHIC:
            advice.append(
                "Atrophic scars respond well to treatments that stimulate "
                "collagen production. Consistent long-term care tends to "
                "show the best results."
            )

        if alerts:
            has_growth_alert = any(a.get("type") == "accelerating_growth" for a in alerts)
            if has_growth_alert:
                advice.append(
                    "Your scar is showing accelerating growth — this should be "
                    "evaluated by a dermatologist promptly."
                )

        return advice
