import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from schema import (
    ScarType,
    BodyRegion,
    ScarStateVector,
    ScarObservation,
    ScarProfile,
    HealingTrajectory,
    
)
from feature_extractor import ScarFeatureExtractor
from trajectory import TrajectoryEngine, TrajectoryStats, PopulationComparison
from severity import SeverityScorer, SeverityBreakdown
from classifier import ScarClassifier, ClassificationResult
from storage import ScarDatabase
from suggestions import SuggestionEngine, SuggestionReport


@dataclass
class ObservationReport:
    """Complete output from processing a single new photograph."""

    observation: ScarObservation
    severity: SeverityBreakdown
    severity_change: Optional[dict]
    classification: ClassificationResult
    trajectory: Optional[HealingTrajectory]
    alerts: list[dict]
    summary: str


@dataclass
class FullScarReport:
    """Comprehensive report on a scar's entire history."""

    profile: ScarProfile
    current_severity: SeverityBreakdown
    classification: ClassificationResult
    trajectory: HealingTrajectory
    detailed_stats: list[TrajectoryStats]
    population_comparison: Optional[PopulationComparison]
    alerts: list[dict]
    severity_timeline: list[dict]
    summary: str


class ScarAnalysisPipeline:
    """Main orchestrator.

    Two primary operations:
    1. process_observation() — handle a new photograph
    2. generate_full_report() — comprehensive scar history report

    Manages the lifecycle: extract → classify → score → track → alert.
    """

    def __init__(self, db_path: str = "scars.db", population_curves: dict = None):
        """Initialize pipeline.

        Args:
            db_path: path to the SQLite database file.
            population_curves: precomputed healing curves by scar type.
                Format: {scar_type_str: {dimension: [(day, value), ...]}}
                Pass None to skip population comparison.
        """
        self.extractor = ScarFeatureExtractor()
        self.trajectory_engine = TrajectoryEngine()
        self.severity_scorer = SeverityScorer()
        self.classifier = ScarClassifier()
        self.population_curves = population_curves or {}
        self.db = ScarDatabase(db_path)
        self.suggestion_engine = SuggestionEngine()

    def register_scar(
        self,
        patient_id: str,
        body_region: BodyRegion,
        cause: str = None,
        date_of_injury: datetime = None,
    ) -> str:
        """Register a new scar for tracking.

        Call once before first observation. Returns the scar_id.
        """
        scar_id = str(uuid.uuid4())
        profile = ScarProfile(
            scar_id=scar_id,
            patient_id=patient_id,
            scar_type=ScarType.UNCLASSIFIED,
            body_region=body_region,
            cause=cause,
            date_of_injury=date_of_injury,
            observations=[],
        )
        self.db.save_profile(profile)
        return scar_id

    def process_observation(
        self,
        scar_id: str,
        image_path: str,
        scar_mask=None,
        patient_metadata: dict = None,
        notes: str = None,
        reference_scale_mm_per_px: float = None,
        timestamp: datetime = None,
    ) -> ObservationReport:
        """Process a new photograph.

        This is the primary entry point called every time the user takes a photo
        of their scar.

        Full pipeline:
        1. Extract features from image → ScarStateVector
        2. Classify scar type (using history if available)
        3. Score severity
        4. Compare to previous severity
        5. Update trajectory
        6. Check for anomalies/alerts
        7. Generate summary

        Returns an ObservationReport with everything the UI needs.
        """
        profile = self.db.get_profile(scar_id)
        if profile is None:
            raise ValueError(f"Unknown scar_id: {scar_id}. Call register_scar() first.")

        ts = timestamp or datetime.now()

        # step 1: extract features
        if reference_scale_mm_per_px:
            self.extractor.scale = reference_scale_mm_per_px
        state = self.extractor.extract(image_path, scar_mask)

        # step 2: create observation
        observation = ScarObservation(
            observation_id=str(uuid.uuid4()),
            scar_id=scar_id,
            timestamp=ts,
            image_path=image_path,
            state=state,
            notes=notes,
            reference_scale_present=reference_scale_mm_per_px is not None,
        )

        # step 3: classify
        metadata = patient_metadata or {}
        if profile.cause:
            metadata.setdefault("cause", profile.cause)
        if profile.date_of_injury:
            age_days = (ts - profile.date_of_injury).total_seconds() / 86400
            metadata["age_months"] = age_days / 30.0

        if len(profile.observations) >= 1:
            all_states = [o.state for o in profile.observations] + [state]
            all_timestamps = [o.timestamp for o in profile.observations] + [ts]
            classification = self.classifier.classify_with_history(
                all_states,
                all_timestamps,
                metadata,
            )
        else:
            classification = self.classifier.classify(state, metadata)

        if classification.confidence > 0.65:
            profile.scar_type = classification.predicted_type

        # step 4: score severity
        severity = self.severity_scorer.score(state, profile.scar_type)

        # step 5: compare to previous
        severity_change = None
        if len(profile.observations) >= 1:
            previous_state = profile.observations[-1].state
            previous_severity = self.severity_scorer.score(
                previous_state,
                profile.scar_type,
            )
            severity_change = self.severity_scorer.score_change(
                previous_severity,
                severity,
            )

        # step 6: save observation and compute trajectory
        self.db.save_observation(observation)
        if classification.confidence > 0.65:
            self.db.update_scar_type(scar_id, classification.predicted_type)

        profile = self.db.get_profile(scar_id)
        trajectory = None
        alerts = []
        if profile and len(profile.observations) >= 2:
            trajectory = self.trajectory_engine.compute_trajectory(profile)
            alerts = self.trajectory_engine.detect_anomalies(profile)

        # step 7: generate summary
        summary = self._build_observation_summary(
            observation_number=len(profile.observations),
            severity=severity,
            severity_change=severity_change,
            classification=classification,
            trajectory=trajectory,
            alerts=alerts,
        )

        return ObservationReport(
            observation=observation,
            severity=severity,
            severity_change=severity_change,
            classification=classification,
            trajectory=trajectory,
            alerts=alerts,
            summary=summary,
        )

    def generate_full_report(self, scar_id: str) -> FullScarReport:
        """Generate a comprehensive report on a scar's entire history."""
        profile = self.db.get_profile(scar_id)
        if profile is None:
            raise ValueError(f"Unknown scar_id: {scar_id}")
        if not profile.observations:
            raise ValueError("No observations recorded yet.")

        latest_state = profile.observations[-1].state
        metadata = {}
        if profile.cause:
            metadata["cause"] = profile.cause
        if profile.date_of_injury:
            latest_ts = profile.observations[-1].timestamp
            age_days = (latest_ts - profile.date_of_injury).total_seconds() / 86400
            metadata["age_months"] = age_days / 30.0

        all_states = [o.state for o in profile.observations]
        all_timestamps = [o.timestamp for o in profile.observations]
        classification = self.classifier.classify_with_history(
            all_states,
            all_timestamps,
            metadata,
        )

        current_severity = self.severity_scorer.score(latest_state, profile.scar_type)
        trajectory = self.trajectory_engine.compute_trajectory(profile)
        detailed_stats = self.trajectory_engine.compute_detailed_stats(profile)

        population_comparison = None
        if self.population_curves:
            population_comparison = self.trajectory_engine.compare_to_population(
                profile,
                self.population_curves,
            )

        alerts = self.trajectory_engine.detect_anomalies(profile)
        severity_timeline = self._build_severity_timeline(profile)

        summary = self._build_full_summary(
            profile=profile,
            severity=current_severity,
            classification=classification,
            trajectory=trajectory,
            detailed_stats=detailed_stats,
            population_comparison=population_comparison,
            alerts=alerts,
        )

        return FullScarReport(
            profile=profile,
            current_severity=current_severity,
            classification=classification,
            trajectory=trajectory,
            detailed_stats=detailed_stats,
            population_comparison=population_comparison,
            alerts=alerts,
            severity_timeline=severity_timeline,
            summary=summary,
        )

    def get_suggestions(self, scar_id: str) -> SuggestionReport:
        """Get treatment category suggestions for a scar."""
        profile = self.db.get_profile(scar_id)
        if profile is None:
            raise ValueError("Unknown scar_id: {}".format(scar_id))
        if not profile.observations:
            raise ValueError("No observations recorded yet.")

        latest_state = profile.observations[-1].state
        severity = self.severity_scorer.score(latest_state, profile.scar_type)
        trajectory_stats = None
        alerts = []
        if len(profile.observations) >= 2:
            trajectory_stats = self.trajectory_engine.compute_detailed_stats(profile)
            alerts = self.trajectory_engine.detect_anomalies(profile)

        return self.suggestion_engine.suggest(
            scar_type=profile.scar_type,
            severity=severity,
            trajectory_stats=trajectory_stats,
            alerts=alerts,
        )

    def get_all_scars(self, patient_id: str) -> list[dict]:
        """Get summary of all scars for a patient."""
        profiles = self.db.get_patient_profiles(patient_id)
        results = []

        for profile in profiles:
            entry = {
                "scar_id": profile.scar_id,
                "body_region": profile.body_region.value,
                "scar_type": profile.scar_type.value,
                "cause": profile.cause,
                "observation_count": len(profile.observations),
                "first_observed": profile.observations[0].timestamp.isoformat()
                if profile.observations
                else None,
                "last_observed": profile.observations[-1].timestamp.isoformat()
                if profile.observations
                else None,
            }

            if profile.observations:
                latest_severity = self.severity_scorer.score(
                    profile.observations[-1].state,
                    profile.scar_type,
                )
                entry["current_severity"] = latest_severity.composite_score
                entry["clinical_grade"] = latest_severity.clinical_grade

            results.append(entry)
        return results

    def _build_observation_summary(
        self,
        observation_number: int,
        severity: SeverityBreakdown,
        severity_change: Optional[dict],
        classification: ClassificationResult,
        trajectory: Optional[HealingTrajectory],
        alerts: list[dict],
    ) -> str:
        """Build human-readable summary for a single observation."""
        lines = []
        lines.append(f"── Observation #{observation_number} ──")
        lines.append("")

        lines.append(
            f"Scar Type: {classification.predicted_type.value} "
            f"({classification.confidence:.0%} confidence)"
        )
        if classification.needs_professional_review:
            lines.append("⚠ Professional review recommended for classification confirmation")
        for reason in classification.reasoning:
            lines.append(f"  • {reason}")

        lines.append("")

        severity_summary = self.severity_scorer.generate_summary(severity, severity_change)
        lines.append(severity_summary)
        lines.append("")

        if trajectory and trajectory.trend:
            lines.append(f"Healing Trend: {trajectory.trend}")
        lines.append("")

        if alerts:
            lines.append("⚠ Alerts:")
            for alert in alerts:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
        else:
            lines.append("No concerning patterns detected.")

        lines.append("")
        lines.append("This analysis is informational only and does not constitute medical advice.")
        return "\n".join(lines)

    def _build_full_summary(
        self,
        profile: ScarProfile,
        severity: SeverityBreakdown,
        classification: ClassificationResult,
        trajectory: HealingTrajectory,
        detailed_stats: list[TrajectoryStats],
        population_comparison: Optional[PopulationComparison],
        alerts: list[dict],
    ) -> str:
        """Build comprehensive report summary."""
        lines = []
        lines.append("═══════════════════════════════════════")
        lines.append(" SCAR ANALYSIS REPORT ")
        lines.append("═══════════════════════════════════════")
        lines.append("")

        lines.append(f"Location: {profile.body_region.value}")
        lines.append(f"Cause: {profile.cause or 'Not specified'}")
        lines.append(
            f"Type: {classification.predicted_type.value} "
            f"({classification.confidence:.0%} confidence)"
        )
        lines.append(f"Observations: {len(profile.observations)}")
        if profile.date_of_injury:
            age = (profile.observations[-1].timestamp - profile.date_of_injury).days
            lines.append(f"Scar Age: {age} days ({age/30:.1f} months)")
        lines.append("")

        lines.append(
            f"Current Severity: {severity.composite_score:.0f}/100 "
            f"({severity.clinical_grade})"
        )
        top_factors = sorted(
            severity.dimension_contributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        lines.append(
            f"Primary factors: {', '.join(f[0] for f in top_factors)}"
        )
        lines.append("")

        lines.append(f"Overall Trend: {trajectory.trend}")
        if trajectory.velocity:
            improving = []
            worsening = []
            stable = []
            for stat in detailed_stats:
                if stat.trend == "improving":
                    improving.append(stat.dimension)
                elif stat.trend == "worsening":
                    worsening.append(stat.dimension)
                else:
                    stable.append(stat.dimension)

            if improving:
                lines.append(f" Improving: {', '.join(improving)}")
            if worsening:
                lines.append(f" Worsening: {', '.join(worsening)}")
            if stable:
                lines.append(f" Stable: {', '.join(stable)}")
        lines.append("")

        if population_comparison:
            lines.append(
                f"Compared to similar scars: {population_comparison.percentile:.0f}th percentile"
            )
            if population_comparison.faster_than_typical:
                lines.append("  Healing faster than typical")
            else:
                lines.append("  Healing at or below typical rate")
            if population_comparison.estimated_days_to_plateau:
                lines.append(
                    f"  Estimated days to plateau: "
                    f"{population_comparison.estimated_days_to_plateau:.0f}"
                )
            lines.append("")

        if alerts:
            lines.append("⚠ ALERTS:")
            for alert in alerts:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
            lines.append("")

        if classification.needs_professional_review:
            lines.append(
                "⚠ RECOMMENDATION: This scar should be reviewed by a dermatologist."
            )
        lines.append("")
        lines.append("───────────────────────────────────────")
        lines.append("This report is informational only and")
        lines.append("does not constitute medical advice.")
        lines.append("───────────────────────────────────────")
        return "\n".join(lines)

    def _build_severity_timeline(self, profile: ScarProfile) -> list[dict]:
        """Build severity scores over time for charting."""
        timeline = []
        for obs in sorted(profile.observations, key=lambda o: o.timestamp):
            severity = self.severity_scorer.score(obs.state, profile.scar_type)
            timeline.append(
                {
                    "timestamp": obs.timestamp.isoformat(),
                    "composite_score": severity.composite_score,
                    "clinical_grade": severity.clinical_grade,
                    "vascularity": severity.vascularity_score,
                    "pigmentation": severity.pigmentation_score,
                    "pliability": severity.pliability_score,
                    "height": severity.height_score,
                    "texture": severity.texture_score,
                    "color_deviation": severity.color_deviation_score,
                }
            )
        return timeline
