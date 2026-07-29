import os
import numpy as np
from datetime import datetime, timedelta
from schema import ScarType, BodyRegion, ScarStateVector
from pipeline import ScarAnalysisPipeline
from feature_extractor import ScarFeatureExtractor


class SyntheticScarGenerator:
    """Generates realistic synthetic scar observation sequences for testing the pipeline without real images."""

    TRAJECTORIES = {
        ScarType.HYPERTROPHIC: {
            "healing": {
                "vascularity": (2.5, 0.8, 0.15),
                "pigmentation": (2.0, 1.2, 0.1),
                "pliability": (3.0, 1.5, 0.2),
                "height": (2.0, 0.8, 0.1),
                "surface_area_mm2": (800, 600, 30),
                "texture_regularity": (0.3, 0.6, 0.05),
                "color_delta_e": (25, 10, 2),
            },
            "worsening": {
                "vascularity": (1.5, 2.5, 0.15),
                "pigmentation": (1.5, 2.2, 0.1),
                "pliability": (2.0, 3.5, 0.2),
                "height": (1.5, 2.5, 0.15),
                "surface_area_mm2": (600, 900, 30),
                "texture_regularity": (0.5, 0.25, 0.05),
                "color_delta_e": (15, 28, 2),
            },
        },
        ScarType.KELOID: {
            "healing": {
                "vascularity": (2.8, 2.0, 0.2),
                "pigmentation": (2.5, 1.8, 0.15),
                "pliability": (3.5, 2.5, 0.2),
                "height": (2.5, 2.0, 0.1),
                "surface_area_mm2": (1200, 1000, 50),
                "texture_regularity": (0.2, 0.35, 0.05),
                "color_delta_e": (30, 22, 3),
            },
            "growing": {
                "vascularity": (2.0, 2.8, 0.15),
                "pigmentation": (2.0, 2.5, 0.1),
                "pliability": (2.5, 3.5, 0.2),
                "height": (1.8, 2.8, 0.15),
                "surface_area_mm2": (800, 1500, 50),
                "texture_regularity": (0.35, 0.15, 0.05),
                "color_delta_e": (20, 35, 3),
            },
        },
        ScarType.ATROPHIC: {
            "stable": {
                "vascularity": (0.5, 0.3, 0.1),
                "pigmentation": (1.8, 1.5, 0.1),
                "pliability": (1.0, 0.8, 0.1),
                "height": (0.2, 0.2, 0.05),
                "surface_area_mm2": (40, 38, 5),
                "texture_regularity": (0.4, 0.45, 0.05),
                "color_delta_e": (12, 8, 1.5),
            },
        },
        ScarType.FLAT_MATURE: {
            "stable": {
                "vascularity": (0.3, 0.2, 0.05),
                "pigmentation": (0.8, 0.5, 0.1),
                "pliability": (0.5, 0.3, 0.05),
                "height": (0.1, 0.1, 0.02),
                "surface_area_mm2": (200, 195, 10),
                "texture_regularity": (0.75, 0.8, 0.03),
                "color_delta_e": (4, 2.5, 0.5),
            },
        },
    }

    def generate_sequence(
        self,
        scar_type: ScarType,
        scenario: str,
        num_observations: int = 8,
        interval_days: int = 14,
        start_date: datetime = None,
    ) -> list[tuple[datetime, ScarStateVector]]:
        """Generate a time series of ScarStateVectors simulating a specific scar type and healing scenario."""
        if start_date is None:
            start_date = datetime(2025, 1, 15)

        if scar_type not in self.TRAJECTORIES:
            raise ValueError(f"No trajectory defined for {scar_type}")

        scenarios = self.TRAJECTORIES[scar_type]
        if scenario not in scenarios:
            raise ValueError(
                f"Unknown scenario '{scenario}' for {scar_type}. "
                f"Available: {list(scenarios.keys())}"
            )

        trajectory_params = scenarios[scenario]
        observations = []

        for i in range(num_observations):
            t = i / max(1, num_observations - 1)
            timestamp = start_date + timedelta(days=i * interval_days)
            values = {}

            for dim, (start, end, noise) in trajectory_params.items():
                base = start + (end - start) * t
                noisy = base + np.random.normal(0, noise)

                if dim == "surface_area_mm2":
                    noisy = max(1, noisy)
                elif dim == "texture_regularity":
                    noisy = np.clip(noisy, 0, 1)
                else:
                    noisy = max(0, noisy)

                values[dim] = noisy

            state = ScarStateVector(**values)
            observations.append((timestamp, state))

        return observations


class MockFeatureExtractor(ScarFeatureExtractor):
    """Bypasses image loading for testing. Accepts pre-computed ScarStateVectors instead of image paths."""

    def __init__(self):
        super().__init__()
        self._next_state: Optional[ScarStateVector] = None

    def set_next_state(self, state: ScarStateVector):
        """Pre-load the state that the next extract() call will return."""
        self._next_state = state

    def extract(self, image_path: str, scar_mask=None) -> ScarStateVector:
        """Return the pre-loaded state instead of processing an image."""
        if self._next_state is None:
            raise RuntimeError("No state pre-loaded. Call set_next_state() first.")

        state = self._next_state
        self._next_state = None
        return state


def run_test_scenario(
    name: str,
    scar_type: ScarType,
    scenario: str,
    body_region: BodyRegion,
    cause: str,
    patient_metadata: dict = None,
    expected_type: ScarType = None,
    num_observations: int = 8,
) -> object:
    """Run a single test scenario through the full pipeline."""
    print(f"\n{'='*60}")
    print(f" TEST: {name}")
    print(f" Type: {scar_type.value} | Scenario: {scenario}")
    print(f"{'='*60}\n")

    generator = SyntheticScarGenerator()
    observations = generator.generate_sequence(
        scar_type,
        scenario,
        num_observations=num_observations,
    )

    db_path = f"test_{name.replace(' ', '_').lower()}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    pipeline = ScarAnalysisPipeline(db_path=db_path)
    mock_extractor = MockFeatureExtractor()
    pipeline.extractor = mock_extractor

    injury_date = observations[0][0] - timedelta(days=7)
    scar_id = pipeline.register_scar(
        patient_id="test_patient_001",
        body_region=body_region,
        cause=cause,
        date_of_injury=injury_date,
    )

    metadata = patient_metadata or {}
    reports = []

    for i, (timestamp, state) in enumerate(observations):
        mock_extractor.set_next_state(state)
        report = pipeline.process_observation(
            scar_id=scar_id,
            image_path=f"synthetic_{scar_type.value}_{i}.jpg",
            patient_metadata=metadata.copy(),
            timestamp=timestamp,
        )
        reports.append(report)

        print(
            f" Observation {i+1}: severity={report.severity.composite_score:.1f} "
            f"({report.severity.clinical_grade}) | "
            f"type={report.classification.predicted_type.value} "
            f"({report.classification.confidence:.0%})"
        )
        if report.alerts:
            for alert in report.alerts:
                print(f" ⚠ [{alert['severity']}] {alert['message']}")

    print(f"\n{'─'*60}")
    full_report = pipeline.generate_full_report(scar_id)
    print(full_report.summary)
    print(f"\n{'─'*60}")
    print("  VALIDATION:")

    final_classification = full_report.classification.predicted_type
    expected = expected_type if expected_type is not None else scar_type
    print(
        f" Expected type: {expected.value} | Got: {final_classification.value} | "
        f"{'✓ PASS' if final_classification == expected else '✗ FAIL'}"
    )

    expected_trends = {
        "healing": "improving",
        "worsening": "worsening",
        "growing": "worsening",
        "stable": "stable",
    }
    expected_trend = expected_trends.get(scenario, "unknown")
    actual_trend = full_report.trajectory.trend
    print(
        f" Expected trend: {expected_trend} | Got: {actual_trend} | "
        f"{'✓ PASS' if actual_trend == expected_trend else '~ CLOSE' if actual_trend == 'stable' else '✗ FAIL'}"
    )

    severity_start = reports[0].severity.composite_score
    severity_end = reports[-1].severity.composite_score
    if scenario in ("healing",):
        severity_dropped = severity_end < severity_start
        print(
            f" Severity dropped: {severity_start:.1f} → {severity_end:.1f} | "
            f"{'✓ PASS' if severity_dropped else '✗ FAIL'}"
        )
    elif scenario in ("worsening", "growing"):
        severity_rose = severity_end > severity_start
        print(
            f" Severity rose: {severity_start:.1f} → {severity_end:.1f} | "
            f"{'✓ PASS' if severity_rose else '✗ FAIL'}"
        )

    if os.path.exists(db_path):
        os.remove(db_path)

    return full_report


def run_all_tests():
    """Run the complete test suite."""
    np.random.seed(42)
    results = []

    results.append(
        run_test_scenario(
            name="Hypertrophic Healing",
            scar_type=ScarType.HYPERTROPHIC,
            scenario="healing",
            body_region=BodyRegion.UPPER_ARM,
            cause="surgical",
            patient_metadata={"cause": "surgical"},
        )
    )

    results.append(
        run_test_scenario(
            name="Hypertrophic Worsening",
            scar_type=ScarType.HYPERTROPHIC,
            scenario="worsening",
            body_region=BodyRegion.CHEST,
            cause="trauma",
            patient_metadata={"cause": "trauma"},
            expected_type=ScarType.KELOID,
        )
    )

    results.append(
        run_test_scenario(
            name="Keloid Growing",
            scar_type=ScarType.KELOID,
            scenario="growing",
            body_region=BodyRegion.CHEST,
            cause="surgical",
            patient_metadata={
                "cause": "surgical",
                "growing_beyond_boundary": True,
            },
        )
    )

    results.append(
        run_test_scenario(
            name="Keloid Under Treatment",
            scar_type=ScarType.KELOID,
            scenario="healing",
            body_region=BodyRegion.UPPER_ARM,
            cause="trauma",
            patient_metadata={"cause": "trauma"},
        )
    )

    results.append(
        run_test_scenario(
            name="Atrophic Stable",
            scar_type=ScarType.ATROPHIC,
            scenario="stable",
            body_region=BodyRegion.FACE,
            cause="acne",
            patient_metadata={"cause": "acne", "is_depressed": True},
        )
    )

    results.append(
        run_test_scenario(
            name="Flat Mature Stable",
            scar_type=ScarType.FLAT_MATURE,
            scenario="stable",
            body_region=BodyRegion.FOREARM,
            cause="trauma",
            patient_metadata={"cause": "trauma"},
        )
    )

    print(f"\n{'='*60}")
    print(f"  ALL TESTS COMPLETE: {len(results)} scenarios executed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
