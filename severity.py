import numpy as np
from dataclasses import dataclass
from typing import Optional
from schema import ScarStateVector, ScarType


@dataclass
class SeverityBreakdown:
    """Detailed severity scoring across multiple clinical dimensions."""

    vascularity_score: float  # 0-3
    pigmentation_score: float  # 0-3
    pliability_score: float  # 0-5
    height_score: float  # 0-3
    surface_area_score: float  # 0-3 (relative to body region norms)
    texture_score: float  # 0-3
    color_deviation_score: float  # 0-3
    composite_score: float  # 0-100 (weighted aggregate, 0 = invisible, 100 = severe)
    clinical_grade: str  # "minimal", "mild", "moderate", "severe"
    dimension_contributions: dict  # which dimensions are driving severity most


class SeverityScorer:
    """Computes severity scores from a ScarStateVector. Combines Vancouver Scar Scale logic with POSAS-inspired weighting and a normalized composite for longitudinal tracking."""

    # clinical weights — how much each dimension matters to overall severity
    # derived from POSAS observer scale relative importance
    DIMENSION_WEIGHTS = {
        "vascularity": 0.18,
        "pigmentation": 0.14,
        "pliability": 0.20,
        "height": 0.16,
        "surface_area": 0.10,
        "texture": 0.12,
        "color_deviation": 0.10,
    }

    # surface area thresholds in mm2 for scoring
    AREA_THRESHOLDS = {
        "small": 100,       # < 1 cm2
        "medium": 500,      # < 5 cm2
        "large": 2000,      # < 20 cm2
        "very_large": 5000, # >= 50 cm2
    }

    # scar type modifiers — some types are inherently more severe
    TYPE_MODIFIERS = {
        ScarType.FLAT_MATURE: 0.7,
        ScarType.STRETCH_MARK: 0.8,
        ScarType.ATROPHIC: 0.9,
        ScarType.HYPERTROPHIC: 1.0,
        ScarType.CONTRACTURE: 1.15,
        ScarType.KELOID: 1.2,
        ScarType.UNCLASSIFIED: 1.0,
    }

    # composite score to clinical grade mapping
    GRADE_THRESHOLDS = {
        "minimal": (0, 15),
        "mild": (15, 35),
        "moderate": (35, 60),
        "severe": (60, 100),
    }

    def score(
        self,
        state: ScarStateVector,
        scar_type: ScarType = ScarType.UNCLASSIFIED,
    ) -> SeverityBreakdown:
        """Main entry point. Compute full severity breakdown from a state vector."""
        vascularity_score = self._score_vascularity(state.vascularity)
        pigmentation_score = self._score_pigmentation(state.pigmentation)
        pliability_score = self._score_pliability(state.pliability)
        height_score = self._score_height(state.height)
        surface_area_score = self._score_surface_area(state.surface_area_mm2)
        texture_score = self._score_texture(state.texture_regularity)
        color_deviation_score = self._score_color_deviation(state.color_delta_e)

        normalized = {
            "vascularity": vascularity_score / 3.0,
            "pigmentation": pigmentation_score / 3.0,
            "pliability": pliability_score / 5.0,
            "height": height_score / 3.0,
            "surface_area": surface_area_score / 3.0,
            "texture": texture_score / 3.0,
            "color_deviation": color_deviation_score / 3.0,
        }

        composite_raw = sum(
            normalized[dim] * weight
            for dim, weight in self.DIMENSION_WEIGHTS.items()
        )

        type_mod = self.TYPE_MODIFIERS.get(scar_type, 1.0)
        composite = np.clip(composite_raw * type_mod * 100, 0, 100)

        clinical_grade = self._get_clinical_grade(composite)
        contributions = self._compute_contributions(normalized)

        return SeverityBreakdown(
            vascularity_score=vascularity_score,
            pigmentation_score= pigmentation_score,
            pliability_score=pliability_score,
            height_score=height_score,
            surface_area_score=surface_area_score,
            texture_score=texture_score,
            color_deviation_score=color_deviation_score,
            composite_score=float(composite),
            clinical_grade=clinical_grade,
            dimension_contributions=contributions,
        )

    def score_change(
        self,
        previous: SeverityBreakdown,
        current: SeverityBreakdown,
    ) -> dict:
        """Compute the change between two severity assessments. Returns per-dimension deltas and overall change."""
        dims = [
            ("vascularity", "vascularity_score", 3.0),
            ("pigmentation", "pigmentation_score", 3.0),
            ("pliability", "pliability_score", 5.0),
            ("height", "height_score", 3.0),
            ("surface_area", "surface_area_score", 3.0),
            ("texture", "texture_score", 3.0),
            ("color_deviation", "color_deviation_score", 3.0),
        ]

        changes = {}
        for name, attr, max_val in dims:
            prev_val = getattr(previous, attr)
            curr_val = getattr(current, attr)
            delta = curr_val - prev_val
            pct = (delta / max_val) * 100 if max_val > 0 else 0

            if abs(delta) < 0.05:
                direction = "stable"
            elif delta < 0:
                direction = "improved"
            else:
                direction = "worsened"

            changes[name] = {
                "previous": prev_val,
                "current": curr_val,
                "delta": float(delta),
                "percent_change": float(pct),
                "direction": direction,
            }

        changes["composite"] = {
            "previous": previous.composite_score,
            "current": current.composite_score,
            "delta": current.composite_score - previous.composite_score,
            "previous_grade": previous.clinical_grade,
            "current_grade": current.clinical_grade,
            "direction": (
                "improved"
                if current.composite_score < previous.composite_score
                else "worsened"
                if current.composite_score > previous.composite_score
                else "stable"
            ),
        }

        return changes

    def generate_summary(
        self,
        breakdown: SeverityBreakdown,
        change: dict = None,
    ) -> str:
        """Generate a human-readable severity summary. NOT medical advice — informational only."""
        lines = []
        lines.append(
            f"Overall Severity: {breakdown.composite_score:.0f}/100 ({breakdown.clinical_grade})"
        )
        lines.append("")

        sorted_contribs = sorted(
            breakdown.dimension_contributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_drivers = [
            dim for dim, _ in sorted_contribs[:3] if sorted_contribs[0][1] > 0
        ]
        if top_drivers:
            lines.append(f"Primary factors: {', '.join(top_drivers)}")

        if change and "composite" in change:
            c = change["composite"]
            if c["direction"] == "improved":
                lines.append(
                    f"Change: improved by {abs(c['delta']):.1f} points "
                    f"(was {c['previous']:.0f}, now {c['current']:.0f})"
                )
            elif c["direction"] == "worsened":
                lines.append(
                    f"Change: worsened by {abs(c['delta']):.1f} points "
                    f"(was {c['previous']:.0f}, now {c['current']:.0f})"
                )
            else:
                lines.append("Change: stable since last assessment")

            if c["previous_grade"] != c["current_grade"]:
                lines.append(
                    f"Grade changed: {c['previous_grade']} → {c['current_grade']}"
                )

            worsened = [
                dim for dim, info in change.items()
                if dim != "composite"
                and isinstance(info, dict)
                and info.get("direction") == "worsened"
            ]
            if worsened:
                lines.append(f"⚠ Worsened dimensions: {', '.join(worsened)}")

        lines.append("")
        lines.append("This is an informational assessment, not medical advice.")
        return "\n".join(lines)

    def _score_vascularity(self, raw: float) -> float:
        """Vancouver Scar Scale vascularity: 0 = normal (closest to surrounding skin) 1 = pink 2 = red 3 = purple"""
        return float(np.clip(raw, 0, 3))

    def _score_pigmentation(self, raw: float) -> float:
        """0 = normal 1 = slightly abnormal (hypo or hyper) 2 = moderately abnormal 3 = severely abnormal"""
        return float(np.clip(raw, 0, 3))

    def _score_pliability(self, raw: float) -> float:
        """Vancouver Scar Scale pliability: 0 = normal 1 = supple (flexible with minimal resistance) 2 = yielding (gives way to pressure) 3 = firm (inflexible, not easily moved) 4 = banding (rope-like) 5 = contracture (permanent shortening)"""
        return float(np.clip(raw, 0, 5))

    def _score_height(self, raw: float) -> float:
        """0 = flat 1 = <2mm 2 = 2-5mm 3 = >5mm"""
        return float(np.clip(raw, 0, 3))

    def _score_surface_area(self, area_mm2: float) -> float:
        """Convert surface area to 0-3 severity score."""
        if area_mm2 <= self.AREA_THRESHOLDS["small"]:
            return area_mm2 / self.AREA_THRESHOLDS["small"]
        if area_mm2 <= self.AREA_THRESHOLDS["medium"]:
            return 1.0 + (
                area_mm2 - self.AREA_THRESHOLDS["small"]
            ) / (self.AREA_THRESHOLDS["medium"] - self.AREA_THRESHOLDS["small"])
        if area_mm2 <= self.AREA_THRESHOLDS["large"]:
            return 2.0 + (
                area_mm2 - self.AREA_THRESHOLDS["medium"]
            ) / (self.AREA_THRESHOLDS["large"] - self.AREA_THRESHOLDS["medium"])
        return 3.0

    def _score_texture(self, regularity: float) -> float:
        """Invert texture regularity (1 = smooth) to severity (0 = normal)."""
        return float(np.clip((1.0 - regularity) * 3.0, 0, 3))

    def _score_color_deviation(self, delta_e: float) -> float:
        """Map CIE Delta E to severity. <5: barely noticeable 5-15: noticeable 15-30: significant >30: severe"""
        if delta_e < 5:
            return delta_e / 5.0
        if delta_e < 15:
            return 1.0 + (delta_e - 5) / 10.0
        if delta_e < 30:
            return 2.0 + (delta_e - 15) / 15.0
        return 3.0

    def _get_clinical_grade(self, composite: float) -> str:
        """Map composite score to clinical grade."""
        for grade, (low, high) in self.GRADE_THRESHOLDS.items():
            if low <= composite < high:
                return grade
        return "severe"

    def _compute_contributions(self, normalized: dict) -> dict:
        """Compute each dimension's weighted contribution as a percentage of total severity. Shows what's driving the score."""
        weighted = {
            dim: normalized[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in normalized
        }
        total = sum(weighted.values())
        if total == 0:
            return {dim: 0.0 for dim in normalized}
        return {dim: float(val / total) for dim, val in weighted.items()}
