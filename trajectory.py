import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from schema import (
    ScarType,
    ScarStateVector,
    ScarProfile,
    HealingTrajectory,
)


@dataclass
class TrajectoryStats:
    """
    Per-dimension statistics computed over the trajectory.
    """

    dimension: str
    values: list[float]
    timestamps: list[datetime]

    velocity: float        # Units per day (negative = improving for most dimensions)
    acceleration: float    # Change in velocity over time
    total_change: float    # First observation → last observation

    trend: str             # "improving", "stable", "worsening"
    confidence: float      # 0–1 based on observation count and consistency


@dataclass
class PopulationComparison:
    """
    How this scar compares to population healing curves.
    """

    percentile: float      # 0–100 (50 = typical healing rate)
    faster_than_typical: bool
    estimated_days_to_plateau: Optional[float] = None


class TrajectoryEngine:
    """
    Computes healing trajectories from sequential scar observations.
    """

    # Dimensions where lower values indicate improvement.
    LOWER_IS_BETTER = {
        "vascularity",
        "pigmentation",
        "pliability",
        "height",
        "surface_area_mm2",
        "color_delta_e",
    }

    # Dimensions where higher values indicate improvement.
    HIGHER_IS_BETTER = {
        "texture_regularity",
    }

    # Minimum observations needed for reliable trajectory estimation.
    MIN_OBSERVATIONS = 3

    STABILITY_THRESHOLDS = {
        "vascularity": 0.02,
        "pigmentation": 0.02,
        "pliability": 0.03,
        "height": 0.02,
        "surface_area_mm2": 0.5,
        "texture_regularity": 0.005,
        "color_delta_e": 0.1,
    }

    DEFAULT_STABILITY_THRESHOLD = 0.03

    def compute_trajectory(
        self,
        profile: ScarProfile,
    ) -> HealingTrajectory:
        """
        Main entry point.

        Takes a ScarProfile with observations and returns a fully
        computed HealingTrajectory.
        """

        if len(profile.observations) < 2:
            return HealingTrajectory(
                scar_id=profile.scar_id,
                state_sequence=[
                    o.state for o in profile.observations
                ],
                timestamps=[
                    o.timestamp for o in profile.observations
                ],
                velocity=None,
                trend="insufficient_data",
            )

        # Sort observations chronologically.
        sorted_obs = sorted(
            profile.observations,
            key=lambda o: o.timestamp,
        )

        states = [o.state for o in sorted_obs]
        timestamps = [o.timestamp for o in sorted_obs]

        # Compute per-dimension velocities.
        velocity = self._compute_velocity(
            states,
            timestamps,
        )

        # Compute overall trend.
        trend = self._compute_overall_trend(
            states,
            timestamps,
        )

        return HealingTrajectory(
            scar_id=profile.scar_id,
            state_sequence=states,
            timestamps=timestamps,
            velocity=velocity,
            trend=trend,
        )

    def compute_detailed_stats(
        self,
        profile: ScarProfile,
    ) -> list[TrajectoryStats]:
        """
        Compute detailed per-dimension trajectory statistics.
        """

        sorted_obs = sorted(
            profile.observations,
            key=lambda o: o.timestamp,
        )

        states = [o.state for o in sorted_obs]
        timestamps = [o.timestamp for o in sorted_obs]

        dimensions = [
            "vascularity",
            "pigmentation",
            "pliability",
            "height",
            "surface_area_mm2",
            "texture_regularity",
            "color_delta_e",
        ]

        stats = []

        for dim in dimensions:
            values = [getattr(s, dim) for s in states]

            stat = self._compute_dimension_stats(
                dim,
                values,
                timestamps,
            )

            stats.append(stat)

        return stats

    def compare_to_population(
        self,
        profile: ScarProfile,
        population_curves: dict,
    ) -> PopulationComparison:
        if len(profile.observations) < self.MIN_OBSERVATIONS:
            return PopulationComparison(
                percentile=50.0,
                faster_than_typical=False,
            )

        scar_type = profile.scar_type.value
        if scar_type not in population_curves:
            return PopulationComparison(
                percentile=50.0,
                faster_than_typical=False,
            )

        sorted_obs = sorted(
            profile.observations,
            key=lambda o: o.timestamp,
        )

        reference_date = profile.date_of_injury or sorted_obs[0].timestamp
        expected_curves = population_curves[scar_type]
        rates = []

        for dim in [
            "vascularity",
            "color_delta_e",
            "surface_area_mm2",
        ]:
            if dim not in expected_curves:
                continue

            actual_values = [getattr(o.state, dim) for o in sorted_obs]
            actual_days = [
                (o.timestamp - reference_date).total_seconds() / 86400
                for o in sorted_obs
            ]

            if len(actual_values) < 2:
                continue

            actual_rate = (
                actual_values[-1] - actual_values[0]
            ) / max(1.0, actual_days[-1] - actual_days[0])

            expected_rate = self._interpolate_expected_rate(
                expected_curves[dim],
                actual_days[0],
                actual_days[-1],
            )

            if expected_rate == 0:
                continue

            if dim in self.LOWER_IS_BETTER:
                ratio = actual_rate / expected_rate
            else:
                ratio = expected_rate / actual_rate if actual_rate != 0 else 1.0

            rates.append(ratio)

        if not rates:
            return PopulationComparison(
                percentile=50.0,
                faster_than_typical=False,
            )

        mean_ratio = np.mean(rates)
        percentile = float(
            np.clip(100.0 / (1.0 + np.exp(-2.0 * (mean_ratio - 1.0))), 1.0, 99.0)
        )

        return PopulationComparison(
            percentile=percentile,
            faster_than_typical=percentile > 55.0,
            estimated_days_to_plateau=self._estimate_plateau(
                profile,
                population_curves,
            ),
        )

    def detect_anomalies(
        self,
        profile: ScarProfile,
    ) -> list[dict]:
        """
        Detect concerning patterns in the trajectory.

        Returns:
            A list of alert dictionaries.
        """

        alerts = []

        sorted_obs = sorted(
            profile.observations,
            key=lambda o: o.timestamp,
        )

        if len(sorted_obs) < 2:
            return alerts

        states = [o.state for o in sorted_obs]
        timestamps = [o.timestamp for o in sorted_obs]
                # Check for sudden worsening (reversal after improvement).
        for dim in [
            "vascularity",
            "surface_area_mm2",
            "color_delta_e",
        ]:
            values = [getattr(s, dim) for s in states]

            reversal = self._detect_reversal(
                dim,
                values,
                timestamps,
            )

            if reversal:
                alerts.append(reversal)

        # Check for no improvement over an extended period.
        if len(sorted_obs) >= 4:
            stagnation = self._detect_stagnation(
                states,
                timestamps,
                profile.scar_type,
            )

            if stagnation:
                alerts.append(stagnation)

        # Check for accelerating growth (possible keloid formation).
        area_values = [
            s.surface_area_mm2
            for s in states
        ]

        if len(area_values) >= 3:
            keloid_risk = self._detect_accelerating_growth(
                area_values,
                timestamps,
            )

            if keloid_risk:
                alerts.append(keloid_risk)

        return alerts

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _compute_velocity(
        self,
        states: list[ScarStateVector],
        timestamps: list[datetime],
    ) -> dict:
        """
        Compute the rate of change per dimension using linear regression.
        """

        dimensions = [
            "vascularity",
            "pigmentation",
            "pliability",
            "height",
            "surface_area_mm2",
            "texture_regularity",
            "color_delta_e",
        ]

        days = np.array([
            (t - timestamps[0]).total_seconds() / 86400
            for t in timestamps
        ])

        velocity = {}

        for dim in dimensions:
            values = np.array([
                getattr(s, dim)
                for s in states
            ])

            if len(days) < 2 or np.std(days) == 0:
                velocity[dim] = 0.0
                continue

            # Simple linear regression.
            coeffs = np.polyfit(days, values, 1)
            velocity[dim] = float(coeffs[0])

        return velocity

    def _compute_overall_trend(
        self,
        states: list[ScarStateVector],
        timestamps: list[datetime],
    ) -> str:
        """
        Aggregate trend across all dimensions,
        weighted by clinical importance.
        """

        weights = {
            "vascularity": 2.0,
            "pigmentation": 1.5,
            "surface_area_mm2": 2.0,
            "color_delta_e": 1.5,
            "texture_regularity": 1.0,
            "pliability": 1.5,
            "height": 2.0,
        }

        velocity = self._compute_velocity(
            states,
            timestamps,
        )

        weighted_score = 0.0
        total_weight = 0.0

        for dim, vel in velocity.items():
            weight = weights.get(dim, 1.0)
            threshold = self.STABILITY_THRESHOLDS.get(
                dim,
                self.DEFAULT_STABILITY_THRESHOLD,
            )

            if abs(vel) < threshold:
                continue

            if dim in self.LOWER_IS_BETTER:
                contribution = -vel
            else:
                contribution = vel

            weighted_score += contribution * weight
            total_weight += weight

        if total_weight == 0:
            return "stable"

        normalized = weighted_score / total_weight

        if normalized > self.DEFAULT_STABILITY_THRESHOLD:
            return "improving"
        elif normalized < -self.DEFAULT_STABILITY_THRESHOLD:
            return "worsening"

        return "stable"

    def _compute_dimension_stats(
        self,
        dim: str,
        values: list[float],
        timestamps: list[datetime],
    ) -> TrajectoryStats:
        """
        Compute statistics for a single dimension.
        """

        days = np.array([
            (t - timestamps[0]).total_seconds() / 86400
            for t in timestamps
        ])

        # Velocity via linear fit.
        if len(days) >= 2 and np.std(days) > 0:
            coeffs = np.polyfit(days, values, 1)
            velocity = float(coeffs[0])
        else:
            velocity = 0.0

        # Acceleration via quadratic fit.
        if len(days) >= 3 and np.std(days) > 0:
            coeffs2 = np.polyfit(days, values, 2)
            acceleration = float(coeffs2[0] * 2)
        else:
            acceleration = 0.0

        total_change = (
            values[-1] - values[0]
            if len(values) >= 2
            else 0.0
        )

        threshold = self.STABILITY_THRESHOLDS.get(
            dim,
            self.DEFAULT_STABILITY_THRESHOLD,
        )

        if dim in self.LOWER_IS_BETTER:
            if velocity < -threshold:
                trend = "improving"
            elif velocity > threshold:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            if velocity > threshold:
                trend = "improving"
            elif velocity < -threshold:
                trend = "worsening"
            else:
                trend = "stable"

        # Confidence based on observation count and fit quality.
        n = len(values)

        if n >= 2 and np.std(days) > 0:
            predicted = np.polyval(
                np.polyfit(days, values, 1),
                days,
            )

            residuals = np.array(values) - predicted

            r_squared = 1 - (
                np.var(residuals)
                / max(np.var(values), 1e-10)
            )

            confidence = min(
                1.0,
                max(0.0, r_squared)
                * min(1.0, n / self.MIN_OBSERVATIONS),
            )
        else:
            confidence = 0.0

        return TrajectoryStats(
            dimension=dim,
            values=values,
            timestamps=timestamps,
            velocity=velocity,
            acceleration=acceleration,
            total_change=total_change,
            trend=trend,
            confidence=confidence,
        )

    def _detect_reversal(
        self,
        dim: str,
        values: list[float],
        timestamps: list[datetime],
    ) -> Optional[dict]:
        """
        Detect if a dimension was improving and then worsened.
        """

        if len(values) < 3:
            return None

        if dim == "color_delta_e" and max(abs(v) for v in values) < 5.0:
            return None

        mid = len(values) // 2
        first_half_trend = values[mid] - values[0]
        second_half_trend = values[-1] - values[mid]

        max_val = max(abs(v) for v in values)
        if max_val < 1.0 and dim != "texture_regularity":
            return None

        value_range = max(max_val, 1.0)
        improvement_threshold = value_range * 0.1
        reversal_threshold = value_range * 0.15

        is_lower_better = dim in self.LOWER_IS_BETTER
        if is_lower_better:
            was_improving = first_half_trend < -improvement_threshold
            now_worsening = second_half_trend > reversal_threshold
        else:
            was_improving = first_half_trend > improvement_threshold
            now_worsening = second_half_trend < -reversal_threshold

        if was_improving and now_worsening:
            return {
                "type": "reversal",
                "dimension": dim,
                "severity": "warning",
                "message": (
                    f"{dim} was improving but has reversed direction. "
                    "Consider consulting a dermatologist."
                ),
                "detected_at": timestamps[-1].isoformat(),
            }

        return None

    def _interpolate_expected_rate(
        self,
        curve,
        start_day: float,
        end_day: float,
    ) -> float:
        """Interpolate an expected rate from a population healing curve."""
        if end_day == start_day:
            return 0.0

        if isinstance(curve, dict):
            points = sorted(curve.items())
        else:
            points = sorted(curve)

        if not points:
            return 0.0

        def sample(day: float) -> float:
            if day <= points[0][0]:
                return points[0][1]
            if day >= points[-1][0]:
                return points[-1][1]

            for i in range(1, len(points)):
                prev_day, prev_value = points[i - 1]
                next_day, next_value = points[i]
                if prev_day <= day <= next_day:
                    fraction = (day - prev_day) / max(1e-6, next_day - prev_day)
                    return prev_value + fraction * (next_value - prev_value)

            return points[-1][1]

        return (sample(end_day) - sample(start_day)) / max(1e-6, end_day - start_day)

    def _estimate_plateau(
        self,
        profile: ScarProfile,
        population_curves: dict,
    ) -> Optional[float]:
        """Estimate when healing is likely to plateau."""
        sorted_obs = sorted(
            profile.observations,
            key=lambda o: o.timestamp,
        )

        if len(sorted_obs) < 2:
            return None

        timestamps = [o.timestamp for o in sorted_obs]
        states = [o.state for o in sorted_obs]
        days = np.array([
            (t - timestamps[0]).total_seconds() / 86400
            for t in timestamps
        ])

        velocities = self._compute_velocity(states, timestamps)
        dims = ["vascularity", "surface_area_mm2", "color_delta_e"]
        avg_velocity = float(np.mean([velocities.get(dim, 0.0) for dim in dims]))

        if abs(avg_velocity) < self.DEFAULT_STABILITY_THRESHOLD:
            return float(days[-1])

        plateau_window = min(max(1.0, 30.0 / abs(avg_velocity)), 90.0)
        return float(days[-1] + plateau_window)

    def _detect_stagnation(
        self,
        states: list[ScarStateVector],
        timestamps: list[datetime],
        scar_type: ScarType,
    ) -> Optional[dict]:
        """Detect little or no improvement over multiple observations."""
        if scar_type == ScarType.FLAT_MATURE:
            return None

        key_dims = ["vascularity", "surface_area_mm2", "color_delta_e"]
        stagnation_count = 0

        for dim in key_dims:
            values = np.array([getattr(s, dim) for s in states])
            if len(values) < 2:
                continue
            if abs(values[-1] - values[0]) < 0.2:
                stagnation_count += 1

        if stagnation_count >= 2:
            return {
                "type": "stagnation",
                "severity": "notice",
                "message": (
                    "Scar appearance has shown minimal change over time. "
                    "Review the treatment plan or follow-up frequency."
                ),
                "detected_at": timestamps[-1].isoformat(),
            }

        return None

    def _detect_accelerating_growth(
        self,
        area_values: list[float],
        timestamps: list[datetime],
    ) -> Optional[dict]:
        """Detect accelerating surface area growth, which can indicate keloid risk."""
        days = np.array([
            (t - timestamps[0]).total_seconds() / 86400
            for t in timestamps
        ])

        if len(days) < 3 or np.std(days) == 0:
            return None

        if max(area_values) < 300:
            return None

        coeffs = np.polyfit(days, area_values, 2)
        acceleration = coeffs[0] * 2
        velocity = coeffs[1]
        mean_area = np.mean(area_values)
        relative_acceleration = acceleration / max(mean_area, 1.0)

        if relative_acceleration > 0.0001 and velocity > 0.5:
            return {
                "type": "accelerating_growth",
                "severity": "alert",
                "message": (
                    "Scar area is growing at an increasing rate. "
                    "This pattern is associated with keloid formation. "
                    "Consult a dermatologist promptly."
                ),
                "detected_at": timestamps[-1].isoformat(),
            }

        return None