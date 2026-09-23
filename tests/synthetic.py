"""Synthetic scar photographs with known ground truth.

Each photo is a skin patch with pore-level noise, a smooth illumination
gradient, a random exposure / white-balance shift (to mimic different rooms
and phones), and an elliptical scar with controllable redness, darkness,
size and roughness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

SKIN_TONES = {
    # Approximate LAB for Fitzpatrick-like tones (L, a, b)
    "light": (72.0, 12.0, 18.0),
    "medium": (58.0, 14.0, 22.0),
    "dark": (40.0, 13.0, 18.0),
}


@dataclass
class ScarSpec:
    redness: float = 12.0        # a* above skin
    darkness: float = 4.0        # L* below skin (negative = paler)
    size: float = 0.18           # ellipse semi-major axis as fraction of width
    aspect: float = 0.35
    roughness: float = 0.0       # 0 smooth .. 1 very bumpy
    skin: str = "medium"
    lighting: float = 1.0        # exposure multiplier
    wb_shift: float = 0.0        # white balance shift on a*/b*
    angle: float = 20.0
    seed: int = 0


def make_photo(spec: ScarSpec, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(spec.seed)
    L0, a0, b0 = SKIN_TONES[spec.skin]

    lab = np.zeros((h, w, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = L0, a0, b0

    # skin micro-texture
    noise = rng.normal(0, 1.2, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2) * 2.0
    lab[..., 0] += noise

    # scar
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2 + rng.uniform(-15, 15), h / 2 + rng.uniform(-15, 15)
    ax, ay = spec.size * w, spec.size * w * spec.aspect
    t = np.deg2rad(spec.angle)
    xr = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    yr = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    r = np.sqrt((xr / ax) ** 2 + (yr / ay) ** 2)
    prof = np.clip(1.25 - r, 0, 1) / 0.25       # soft edge
    prof = np.clip(prof, 0, 1).astype(np.float32)

    lab[..., 1] += prof * spec.redness
    lab[..., 0] -= prof * spec.darkness
    lab[..., 2] += prof * spec.redness * 0.15
    if spec.roughness > 0:
        bumps = rng.normal(0, 1, (h, w)).astype(np.float32)
        bumps = cv2.GaussianBlur(bumps, (0, 0), 2.0)
        bumps = bumps / (bumps.std() + 1e-6)
        lab[..., 0] += prof * bumps * 7.0 * spec.roughness

    # smooth illumination gradient + exposure + white balance
    grad = (xx / w - 0.5) * 6.0
    lab[..., 0] = lab[..., 0] * spec.lighting + grad
    lab[..., 1] += spec.wb_shift
    lab[..., 2] += spec.wb_shift * 1.5

    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = np.clip(bgr * 255.0 + rng.normal(0, 1.5, bgr.shape), 0, 255).astype(np.uint8)
    return bgr


def encode_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def jitter(spec: ScarSpec, rng: np.random.Generator, strength: float = 1.0) -> ScarSpec:
    """Realistic photo-to-photo variation: different lighting and phone colour."""
    return ScarSpec(**{**spec.__dict__,
                       "lighting": 1.0 + rng.uniform(-0.12, 0.12) * strength,
                       "wb_shift": rng.uniform(-2.5, 2.5) * strength,
                       "angle": spec.angle + rng.uniform(-8, 8),
                       "seed": int(rng.integers(0, 1_000_000))})
