"""Core domain data structures (framework independent).

The ScarStateVector mirrors the Vancouver Scar Scale / POSAS dimensions:
vascularity, pigmentation, pliability, height (VSS) plus surface area,
texture regularity and colour deviation (POSAS-style observer items).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Direction of each dimension: +1 means "a larger value is clinically worse",
# -1 means "a larger value is clinically better". Getting this mapping wrong is
# exactly what makes a worsening scar look like it is healing.
DIMENSION_DIRECTION: Dict[str, int] = {
    "vascularity": +1,
    "pigmentation": +1,
    "pliability": +1,          # stiffness: 0 = normal .. 5 = contracture
    "height": +1,              # 0 = flat .. 3 = > 5 mm raised
    "surface_area": +1,        # fraction of frame covered by scar
    "texture_regularity": -1,  # 1 = as smooth as surrounding skin
    "color_delta_e": +1,       # CIE76 distance from surrounding skin
}

# Friendly names shown to patients.
DIMENSION_LABELS: Dict[str, str] = {
    "vascularity": "Redness",
    "pigmentation": "Darkness",
    "pliability": "Tightness",
    "height": "Raised",
    "surface_area": "Size",
    "texture_regularity": "Smoothness",
    "color_delta_e": "Colour difference",
}


@dataclass
class SelfReport:
    """What the patient tells us that a 2D photo cannot show."""
    height_level: Optional[int] = None      # 0 flat, 1 slightly raised, 2 raised, 3 very raised
    is_depressed: bool = False              # sunken / pitted
    feels_tight: bool = False
    growing_beyond_boundary: bool = False
    itchy: bool = False
    painful: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SelfReport":
        d = d or {}
        default = cls()
        return cls(**{k: d.get(k, getattr(default, k)) for k in cls.__dataclass_fields__})


@dataclass
class ImageQuality:
    ok: bool = True
    blur_score: float = 0.0
    brightness: float = 0.0
    scar_found: bool = True
    warnings: List[str] = field(default_factory=list)


DIMENSIONS = (
    "vascularity", "pigmentation", "pliability", "height",
    "surface_area", "texture_regularity", "color_delta_e",
)


@dataclass
class ScarStateVector:
    vascularity: float          # 0..3
    pigmentation: float         # 0..3
    pliability: float           # 0..5
    height: float               # 0..3
    surface_area: float         # 0..1 fraction of frame
    texture_regularity: float   # 0..1
    color_delta_e: float        # CIE76
    raw: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in DIMENSIONS}

    def to_json(self) -> dict:
        out: dict = self.as_dict()
        out["raw"] = {k: float(v) for k, v in self.raw.items()}
        return out

    @classmethod
    def from_json(cls, d: dict) -> "ScarStateVector":
        return cls(**{k: float(d[k]) for k in DIMENSIONS}, raw=d.get("raw", {}))
