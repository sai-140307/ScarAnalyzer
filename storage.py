import sqlite3
import os
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from schema import (
    ScarType,
    BodyRegion,
    ScarStateVector,
    ScarObservation,
    ScarProfile,
    HealingTrajectory,
)


class ScarDatabase:
    """SQLite persistence layer for scar tracking.

    Stores profiles, observations, and summarized state vectors.
    """

    def __init__(self, db_path: str = "scars.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS scar_profiles (
                    scar_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    scar_type TEXT NOT NULL DEFAULT 'unclassified',
                    body_region TEXT NOT NULL,
                    cause TEXT,
                    date_of_injury TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                );

                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    scar_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    notes TEXT,
                    reference_scale_present INTEGER NOT NULL DEFAULT 0,
                    vascularity REAL NOT NULL,
                    pigmentation REAL NOT NULL,
                    pliability REAL NOT NULL,
                    height REAL NOT NULL,
                    surface_area_mm2 REAL NOT NULL,
                    texture_regularity REAL NOT NULL,
                    color_delta_e REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (scar_id) REFERENCES scar_profiles(scar_id)
                );

                CREATE INDEX IF NOT EXISTS idx_observations_scar_time
                    ON observations(scar_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_profiles_patient
                    ON scar_profiles(patient_id);
                """
            )

    # patient operations
    def ensure_patient(self, patient_id: str):
        with self._connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id) VALUES (?)",
                (patient_id,),
            )

    # profile operations
    def save_profile(self, profile: ScarProfile):
        self.ensure_patient(profile.patient_id)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scar_profiles (
                    scar_id,
                    patient_id,
                    scar_type,
                    body_region,
                    cause,
                    date_of_injury
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.scar_id,
                    profile.patient_id,
                    profile.scar_type.value,
                    profile.body_region.value,
                    profile.cause,
                    profile.date_of_injury.isoformat()
                    if profile.date_of_injury
                    else None,
                ),
            )

    def get_profile(self, scar_id: str) -> Optional[ScarProfile]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scar_profiles WHERE scar_id = ?",
                (scar_id,),
            ).fetchone()

            if not row:
                return None

            observations = self.get_observations(scar_id)
            return ScarProfile(
                scar_id=row["scar_id"],
                patient_id=row["patient_id"],
                scar_type=ScarType(row["scar_type"]),
                body_region=BodyRegion(row["body_region"]),
                cause=row["cause"],
                date_of_injury=
                    datetime.fromisoformat(row["date_of_injury"])
                    if row["date_of_injury"]
                    else None,
                observations=observations,
            )

    def get_patient_profiles(self, patient_id: str) -> list[ScarProfile]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT scar_id FROM scar_profiles WHERE patient_id = ?",
                (patient_id,),
            ).fetchall()

        return [self.get_profile(row["scar_id"]) for row in rows]

    def update_scar_type(self, scar_id: str, scar_type: ScarType):
        with self._connection() as conn:
            conn.execute(
                "UPDATE scar_profiles SET scar_type = ? WHERE scar_id = ?",
                (scar_type.value, scar_id),
            )

    # observation operations
    def save_observation(self, observation: ScarObservation):
        with self._connection() as conn:
            state = observation.state
            conn.execute(
                """
                INSERT OR REPLACE INTO observations (
                    observation_id,
                    scar_id,
                    timestamp,
                    image_path,
                    notes,
                    reference_scale_present,
                    vascularity,
                    pigmentation,
                    pliability,
                    height,
                    surface_area_mm2,
                    texture_regularity,
                    color_delta_e
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.scar_id,
                    observation.timestamp.isoformat(),
                    observation.image_path,
                    observation.notes,
                    1 if observation.reference_scale_present else 0,
                    state.vascularity,
                    state.pigmentation,
                    state.pliability,
                    state.height,
                    state.surface_area_mm2,
                    state.texture_regularity,
                    state.color_delta_e,
                ),
            )

    def get_observations(self, scar_id: str) -> list[ScarObservation]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM observations
                WHERE scar_id = ?
                ORDER BY timestamp ASC
                """,
                (scar_id,),
            ).fetchall()

        return [self._row_to_observation(row) for row in rows]

    def get_latest_observation(self, scar_id: str) -> Optional[ScarObservation]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM observations
                WHERE scar_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (scar_id,),
            ).fetchone()

        if not row:
            return None
        return self._row_to_observation(row)

    def get_observation_count(self, scar_id: str) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM observations WHERE scar_id = ?",
                (scar_id,),
            ).fetchone()

        return row["cnt"]

    # query operations
    def get_scars_by_type(self, scar_type: ScarType) -> list[ScarProfile]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT scar_id FROM scar_profiles WHERE scar_type = ?",
                (scar_type.value,),
            ).fetchall()

        return [self.get_profile(row["scar_id"]) for row in rows]

    def get_severity_timeline(self, scar_id: str) -> list[dict]:
        """Raw state vectors over time for external severity computation."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, vascularity, pigmentation, pliability,
                    height, surface_area_mm2, texture_regularity,
                    color_delta_e
                FROM observations
                WHERE scar_id = ?
                ORDER BY timestamp ASC
                """,
                (scar_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_population_stats(self, scar_type: ScarType) -> dict:
        """Aggregate statistics across scar type for population comparison."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT o.scar_id) as scar_count,
                    COUNT(*) as observation_count,
                    AVG(o.vascularity) as avg_vascularity,
                    AVG(o.pigmentation) as avg_pigmentation,
                    AVG(o.surface_area_mm2) as avg_surface_area,
                    AVG(o.color_delta_e) as avg_color_delta_e,
                    AVG(o.texture_regularity) as avg_texture_regularity
                FROM observations o
                JOIN scar_profiles p ON o.scar_id = p.scar_id
                WHERE p.scar_type = ?
                """,
                (scar_type.value,),
            ).fetchone()

        return dict(row) if row else {}

    # deletion
    def delete_observation(self, observation_id: str):
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM observations WHERE observation_id = ?",
                (observation_id,),
            )

    def delete_scar(self, scar_id: str):
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM observations WHERE scar_id = ?",
                (scar_id,),
            )
            conn.execute(
                "DELETE FROM scar_profiles WHERE scar_id = ?",
                (scar_id,),
            )

    # helpers
    def _row_to_observation(self, row) -> ScarObservation:
        state = ScarStateVector(
            vascularity=row["vascularity"],
            pigmentation=row["pigmentation"],
            pliability=row["pliability"],
            height=row["height"],
            surface_area_mm2=row["surface_area_mm2"],
            texture_regularity=row["texture_regularity"],
            color_delta_e=row["color_delta_e"],
        )
        return ScarObservation(
            observation_id=row["observation_id"],
            scar_id=row["scar_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            image_path=row["image_path"],
            state=state,
            notes=row["notes"],
            reference_scale_present=bool(row["reference_scale_present"]),
        )
