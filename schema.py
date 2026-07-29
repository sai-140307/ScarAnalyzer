from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional


class ScarType(Enum):
    HYPERTROPHIC = "hypertrophic"
    KELOID = "keloid"
    ATROPHIC = "atrophic"
    CONTRACTURE = "contracture"
    STRETCH_MARK = "stretch_mark"
    FLAT_MATURE = "flat_mature"
    UNCLASSIFIED = "unclassified"


class BodyRegion(Enum):
    FACE = "face"
    NECK = "neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    UPPER_BACK = "upper_back"
    LOWER_BACK = "lower_back"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    HAND = "hand"
    UPPER_LEG = "upper_leg"
    LOWER_LEG = "lower_leg"
    FOOT = "foot"
    OTHER = "other"


@dataclass
class ScarStateVector:
    """
    Quantified scar state at a single point in time.
    """

    vascularity: float          # 0–3: pale → red → purple
    pigmentation: float         # 0–3: hypopigmented → normal → hyperpigmented
    pliability: float           # 0–5: normal → banding/contracture
    height: float               # 0–3: flat → >5 mm elevation
    surface_area_mm2: float     # Measured from image with reference scale
    texture_regularity: float   # 0–1: smooth → irregular (computed from image)
    color_delta_e: float        # CIE Delta E from surrounding skin


@dataclass
class ScarObservation:
    """
    One photograph session—a snapshot in the timeline.
    """

    observation_id: str
    scar_id: str
    timestamp: datetime
    image_path: str
    state: ScarStateVector

    notes: Optional[str] = None
    reference_scale_present: bool = False  # Did the patient include a ruler/coin?


@dataclass
class ScarProfile:
    """
    The full scar identity—created once and tracked over time.
    """

    scar_id: str
    patient_id: str
    scar_type: ScarType
    body_region: BodyRegion

    cause: Optional[str] = None            # surgical, burn, acne, trauma, etc.
    date_of_injury: Optional[datetime] = None

    observations: list[ScarObservation] = field(default_factory=list)


@dataclass
class HealingTrajectory:
    """
    Computed from observations—the scar's journey through state space.
    """

    scar_id: str

    state_sequence: list[ScarStateVector] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    velocity: Optional[dict] = None        # Rate of change per dimension
    trend: Optional[str] = None            # "improving", "stable", or "worsening"