import numpy as np
from dataclasses import dataclass
from typing import Optional

from schema import ScarType, ScarStateVector


@dataclass
class ClassificationResult:
    """Output of scar type classification."""

    predicted_type: ScarType
    confidence: float  # 0-1
    probabilities: dict  # {ScarType: probability}
    reasoning: list[str]  # human-readable explanation of why
    needs_professional_review: bool  # flag if confidence is low or ambiguous


class ScarClassifier:
    """Classifies scar type from extracted features.

    Phase 1: Rule-based classifier using clinical heuristics.
    Phase 2: Drop-in replacement with trained model (same interface).

    Clinical basis:
    - Keloid: extends beyond original wound boundary, elevated, high vascularity
    - Hypertrophic: stays within wound boundary, elevated, may be red
    - Atrophic: depressed below skin surface (acne scars, pockmarks)
    - Contracture: tight, pulls surrounding skin (typically from burns)
    - Stretch mark: linear, often parallel lines, minimal elevation
    - Flat/mature: flush with skin, minimal color deviation
    """

    REVIEW_THRESHOLD = 0.6

    def classify(
        self,
        state: ScarStateVector,
        metadata: dict = None,
    ) -> ClassificationResult:
        """Classify scar type from its state vector and optional metadata.

        Args:
            state: current ScarStateVector from feature extraction
            metadata: optional dict with keys like:
                - "cause": str ("burn", "surgical", "acne", "trauma")
                - "age_months": float (how old the scar is)
                - "growing_beyond_boundary": bool (patient-reported)
                - "feels_tight": bool (patient-reported)
                - "is_depressed": bool (patient-reported)
        """
        if metadata is None:
            metadata = {}

        scores = self._compute_type_scores(state, metadata)
        probabilities = self._normalize_scores(scores)

        predicted_type = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_type]
        reasoning = self._generate_reasoning(state, metadata, predicted_type)
        needs_review = self._check_needs_review(probabilities, confidence)

        return ClassificationResult(
            predicted_type=predicted_type,
            confidence=confidence,
            probabilities={k.value: v for k, v in probabilities.items()},
            reasoning=reasoning,
            needs_professional_review=needs_review,
        )

    def classify_with_history(
        self,
        states: list[ScarStateVector],
        timestamps: list,
        metadata: dict = None,
    ) -> ClassificationResult:
        """Classify using multiple observations over time.

        Temporal patterns improve classification accuracy significantly.

        Key temporal signals:
        - Keloids keep growing months after injury
        - Hypertrophic scars stabilize or regress after 6-12 months
        - Atrophic scars don't change much in elevation
        - Contractures progressively tighten
        """
        if metadata is None:
            metadata = {}

        if len(states) < 2:
            return self.classify(states[-1], metadata)

        temporal = self._extract_temporal_features(states, timestamps)
        metadata.update(temporal)

        return self.classify(states[-1], metadata)

    def _compute_type_scores(self, state: ScarStateVector, metadata: dict) -> dict:
        """Compute raw likelihood scores for each scar type. Uses clinical heuristic rules with weighted evidence."""
        scores = {
            ScarType.KELOID: 0.0,
            ScarType.HYPERTROPHIC: 0.0,
            ScarType.ATROPHIC: 0.0,
            ScarType.CONTRACTURE: 0.0,
            ScarType.STRETCH_MARK: 0.0,
            ScarType.FLAT_MATURE: 0.0,
        }

        if state.height >= 2.0:
            scores[ScarType.KELOID] += 2.0
            scores[ScarType.HYPERTROPHIC] += 1.5
        elif state.height >= 1.0:
            scores[ScarType.HYPERTROPHIC] += 1.5
            scores[ScarType.KELOID] += 0.8
        elif state.height <= 0.3:
            scores[ScarType.FLAT_MATURE] += 1.5
            scores[ScarType.ATROPHIC] += 1.0
            scores[ScarType.STRETCH_MARK] += 1.0

        if state.vascularity >= 2.0:
            scores[ScarType.KELOID] += 1.5
            scores[ScarType.HYPERTROPHIC] += 1.2
        elif state.vascularity <= 0.5:
            scores[ScarType.FLAT_MATURE] += 1.0
            scores[ScarType.STRETCH_MARK] += 0.8
            scores[ScarType.ATROPHIC] += 0.5

        if state.pliability >= 4.0:
            scores[ScarType.CONTRACTURE] += 3.0
        elif state.pliability >= 3.0:
            scores[ScarType.CONTRACTURE] += 1.5
            scores[ScarType.KELOID] += 1.0
        elif state.pliability <= 1.0:
            scores[ScarType.FLAT_MATURE] += 1.0
            scores[ScarType.STRETCH_MARK] += 0.8

        if state.surface_area_mm2 > 2000:
            scores[ScarType.KELOID] += 1.0
            scores[ScarType.CONTRACTURE] += 0.8
        elif state.surface_area_mm2 < 50:
            scores[ScarType.ATROPHIC] += 0.8

        if state.texture_regularity < 0.3:
            scores[ScarType.KELOID] += 0.8
            scores[ScarType.HYPERTROPHIC] += 0.5
        elif state.texture_regularity > 0.7:
            scores[ScarType.FLAT_MATURE] += 0.8
            scores[ScarType.STRETCH_MARK] += 0.6

        if state.color_delta_e > 20:
            scores[ScarType.KELOID] += 0.5
            scores[ScarType.HYPERTROPHIC] += 0.5
        elif state.color_delta_e < 5:
            scores[ScarType.FLAT_MATURE] += 1.2

        cause = metadata.get("cause", "").lower()
        if cause == "burn":
            scores[ScarType.CONTRACTURE] += 2.0
            scores[ScarType.HYPERTROPHIC] += 1.0
        elif cause == "acne":
            scores[ScarType.ATROPHIC] += 2.5
        elif cause == "surgical":
            scores[ScarType.HYPERTROPHIC] += 1.0
            scores[ScarType.KELOID] += 0.5
        elif cause in ("pregnancy", "growth", "weight"):
            scores[ScarType.STRETCH_MARK] += 3.0

        if metadata.get("growing_beyond_boundary"):
            scores[ScarType.KELOID] += 3.0
            scores[ScarType.HYPERTROPHIC] -= 1.0

        if metadata.get("feels_tight"):
            scores[ScarType.CONTRACTURE] += 2.0

        if metadata.get("is_depressed"):
            scores[ScarType.ATROPHIC] += 3.0
            scores[ScarType.KELOID] -= 2.0
            scores[ScarType.HYPERTROPHIC] -= 2.0

        if metadata.get("area_increasing") and metadata.get("age_months", 0) > 6:
            scores[ScarType.KELOID] += 2.5
            scores[ScarType.HYPERTROPHIC] -= 1.0

        if metadata.get("area_decreasing") and metadata.get("age_months", 0) > 6:
            scores[ScarType.HYPERTROPHIC] += 1.5

        if metadata.get("pliability_worsening"):
            scores[ScarType.CONTRACTURE] += 1.5

        if metadata.get("vascularity_decreasing") and metadata.get("age_months", 0) > 12:
            scores[ScarType.FLAT_MATURE] += 1.5

        scores = {k: max(0.0, v) for k, v in scores.items()}
        return scores

    def _normalize_scores(self, scores: dict) -> dict:
        """Convert raw scores to probabilities via softmax."""
        values = np.array(list(scores.values()))
        temperature = 1.5
        exp_values = np.exp(values / temperature)
        total = np.sum(exp_values)

        if total == 0:
            uniform = 1.0 / len(scores)
            return {k: uniform for k in scores}

        probs = exp_values / total
        return dict(zip(scores.keys(), probs.tolist()))

    def _generate_reasoning(
        self,
        state: ScarStateVector,
        metadata: dict,
        predicted: ScarType,
    ) -> list[str]:
        """Generate human-readable explanation for the classification."""
        reasons = []

        if predicted == ScarType.KELOID:
            if state.height >= 2.0:
                reasons.append("Significant elevation above skin surface")
            if state.vascularity >= 2.0:
                reasons.append("High vascularity (red/purple coloring)")
            if metadata.get("growing_beyond_boundary"):
                reasons.append("Reported growth beyond original wound boundary")
            if metadata.get("area_increasing"):
                reasons.append("Surface area has been increasing over time")

        elif predicted == ScarType.HYPERTROPHIC:
            if 1.0 <= state.height < 2.5:
                reasons.append("Moderately elevated above skin surface")
            if state.vascularity >= 1.0:
                reasons.append("Elevated vascularity indicating active remodeling")
            if metadata.get("area_decreasing"):
                reasons.append("Area is decreasing, consistent with hypertrophic regression")

        elif predicted == ScarType.ATROPHIC:
            if metadata.get("is_depressed"):
                reasons.append("Depressed below surrounding skin surface")
            if metadata.get("cause") == "acne":
                reasons.append("Acne-related origin is strongly associated with atrophic scarring")
            if state.height <= 0.3:
                reasons.append("Minimal elevation detected")

        elif predicted == ScarType.CONTRACTURE:
            if state.pliability >= 3.0:
                reasons.append(
                    f"High pliability score ({state.pliability:.1f}) indicating tissue tightening"
                )
            if metadata.get("cause") == "burn":
                reasons.append("Burn-related origin commonly produces contracture")
            if metadata.get("feels_tight"):
                reasons.append("Patient reports tightness/pulling sensation")

        elif predicted == ScarType.STRETCH_MARK:
            if metadata.get("cause") in ("pregnancy", "growth", "weight"):
                reasons.append(
                    f"Cause ({metadata['cause']}) is typical for stretch marks"
                )
            if state.height <= 0.3 and state.vascularity <= 1.0:
                reasons.append("Flat with low vascularity, consistent with striae")

        elif predicted == ScarType.FLAT_MATURE:
            if state.color_delta_e < 5:
                reasons.append("Minimal color deviation from surrounding skin")
            if state.height <= 0.3:
                reasons.append("Flush with skin surface")
            if state.pliability <= 1.0:
                reasons.append("Normal pliability")

        if not reasons:
            reasons.append("Classification based on aggregate feature analysis")

        return reasons

    def _check_needs_review(self, probabilities: dict, top_confidence: float) -> bool:
        """Flag for professional review if the result is low-confidence or ambiguous."""
        if top_confidence < self.REVIEW_THRESHOLD:
            return True

        sorted_probs = sorted(probabilities.values(), reverse=True)
        if len(sorted_probs) >= 2:
            margin = sorted_probs[0] - sorted_probs[1]
            if margin < 0.15:
                return True

        if probabilities.get(ScarType.KELOID, 0) > 0.3:
            return True

        return False

    def _extract_temporal_features(
        self,
        states: list[ScarStateVector],
        timestamps: list,
    ) -> dict:
        """Extract classification-relevant temporal features."""
        features = {}
        if len(states) < 2:
            return features

        areas = [s.surface_area_mm2 for s in states]
        if areas[-1] > areas[0] * 1.1:
            features["area_increasing"] = True
        elif areas[-1] < areas[0] * 0.9:
            features["area_decreasing"] = True

        vasc = [s.vascularity for s in states]
        if vasc[-1] < vasc[0] - 0.3:
            features["vascularity_decreasing"] = True

        pliab = [s.pliability for s in states]
        if pliab[-1] > pliab[0] + 0.3:
            features["pliability_worsening"] = True

        if len(timestamps) >= 2:
            span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
            features["observation_span_days"] = span_days

        return features
