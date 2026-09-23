"""Image -> ScarStateVector.

Pipeline: decode (EXIF-aware) -> resize -> LAB -> skin reference ->
CIE76 delta-E map -> Otsu segmentation -> largest central component ->
local skin ring -> colour / texture / area measurements.

Every colour measurement is taken RELATIVE to the patient's own surrounding
skin in the same photo. That cancels most of the lighting, white balance and
skin-tone differences between photos, which is what makes comparisons over
weeks trustworthy. (Comparing absolute colour between photos taken in
different rooms is a common reason a worsening scar looks "better".)
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import graycomatrix, graycoprops

from .config import ANALYSIS_MAX_SIDE
from .schemas import ImageQuality, ScarStateVector, SelfReport


class ImageDecodeError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> Tuple[np.ndarray, Optional[datetime]]:
    """Return (BGR uint8 image, EXIF capture time or None)."""
    try:
        pil = Image.open(io.BytesIO(data))
        taken_at = _exif_datetime(pil)
        pil = ImageOps.exif_transpose(pil)  # phone photos are often rotated
        pil = pil.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError("This file could not be read as a photo. Use a JPG, PNG or WEBP image.") from exc
    rgb = np.asarray(pil)
    if rgb.shape[0] < 120 or rgb.shape[1] < 120:
        raise ImageDecodeError("This photo is too small. Use a photo at least 300 pixels wide.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), taken_at


def _exif_datetime(pil: Image.Image) -> Optional[datetime]:
    try:
        exif = pil.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal / DateTime
        if raw:
            return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        pass
    return None


def _resize(img: np.ndarray, max_side: int = ANALYSIS_MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Float LAB with L in 0..100 and a*, b* in roughly -128..127."""
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
Roi = Tuple[float, float, float, float]   # x, y, w, h as fractions of the image


def parse_roi(text: Optional[str]) -> Optional[Roi]:
    """'x,y,w,h' with values 0..1 -> tuple, or None."""
    if not text:
        return None
    try:
        x, y, w, h = [float(v) for v in str(text).split(",")]
    except ValueError:
        raise ValueError("The marked area is not valid. Mark the scar again.")
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
    if w < 0.01 or h < 0.01:
        raise ValueError("The marked area is too small. Draw a box around the whole scar.")
    return x, y, w, h


def skin_mask(img_bgr: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Pixels that look like skin (any tone). Excludes hair, eyebrows, dark
    shadows and most non-skin backgrounds."""
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1].astype(np.int16), ycc[..., 2].astype(np.int16)
    m = (cr >= 131) & (cr <= 185) & (cb >= 70) & (cb <= 135) & (lab[..., 0] > 18) & (lab[..., 0] < 97)
    return m


def _smooth_background(lab: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Estimate what the skin 'should' look like at every pixel from nearby
    weighted skin pixels (normalised convolution, computed at low resolution)."""
    h, w = weight.shape
    f = max(1, int(sigma / 6))
    small = cv2.resize(lab, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    ws = cv2.resize(weight.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    s = max(1.0, sigma / f)
    num = cv2.GaussianBlur(small * ws[..., None], (0, 0), s)
    den = cv2.GaussianBlur(ws, (0, 0), s)[..., None]
    fallback = np.median(lab[weight > 0], axis=0) if weight.any() else np.median(lab.reshape(-1, 3), axis=0)
    bg = np.where(den > 1e-3, num / np.maximum(den, 1e-6), fallback)
    return cv2.resize(bg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _otsu(values: np.ndarray) -> float:
    if values.size < 20:
        return 0.0
    v = np.clip(values, 0, 40)
    u8 = (v / 40.0 * 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0 * 40.0


def segment_scar(img_bgr: np.ndarray, lab: np.ndarray, roi: Optional[Roi] = None):
    """Return (scar mask, background-skin LAB map, usable-skin mask, search mask, threshold).

    The scar is whatever differs from the skin immediately around it. Hair,
    eyebrows, eyes and background are excluded from both the scar and the
    reference skin, so a face in the frame is never mistaken for a scar.
    """
    h, w = lab.shape[:2]
    smooth = cv2.GaussianBlur(lab, (0, 0), sigmaX=max(1.0, min(h, w) / 400.0))
    skin = skin_mask(img_bgr, lab)
    usable = skin.copy()

    if roi is not None:
        x, y, rw, rh = roi
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = max(x0 + 2, int((x + rw) * w)), max(y0 + 2, int((y + rh) * h))
        search = np.zeros((h, w), bool)
        search[y0:y1, x0:x1] = True
        size = max(x1 - x0, y1 - y0)
        sigma = max(6.0, 0.22 * size)
        # inside the marked box, accept scar-coloured pixels even if they fail
        # the skin test (very red / very pale scars), but never hair-dark ones
        local_ref = np.median(smooth[skin & ~search], axis=0) if (skin & ~search).any() else np.median(smooth[search], axis=0)
        usable = usable | (search & (smooth[..., 0] > local_ref[0] - 12))
        exclude = np.zeros((h, w), bool)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        min_area = 0.015 * (x1 - x0) * (y1 - y0)
    else:
        sigma = min(h, w) / 6.0
        search = cv2.erode(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        exclude = np.zeros((h, w), bool)
        center = (w / 2.0, h / 2.0)
        min_area = 0.0015 * h * w

    thr = 4.5
    mask = np.zeros((h, w), bool)
    bg = _smooth_background(smooth, usable & ~exclude, sigma)
    for _ in range(2):  # 2nd pass re-estimates the skin without the scar
        de = np.linalg.norm(smooth - bg, axis=2)
        vals = de[search & usable]
        if vals.size:
            med = float(np.median(vals))
            noise = med + 3.0 * 1.4826 * float(np.median(np.abs(vals - med)))
            thr = max(_otsu(vals), 4.5, noise)
        else:
            thr = 4.5
        mask = (de > thr) & search & usable
        k = max(3, int(min(h, w) / 200)) | 1
        mk = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k)))
        mask = mk > 0
        grown = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * k + 1, 4 * k + 1))) > 0
        bg = _smooth_background(smooth, usable & ~exclude & ~grown, sigma)

    de = np.linalg.norm(smooth - bg, axis=2)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best, best_score = 0, -1.0
    diag = np.hypot(h, w)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if roi is None:  # auto mode: things touching the photo edge are hair/background, not a scar
            bx, by, bw, bh = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
            if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
                continue
        dist = np.hypot(cents[i][0] - center[0], cents[i][1] - center[1]) / diag
        score = area * np.exp(-(dist / 0.25) ** 2) * float(de[labels == i].mean())
        if score > best_score:
            best, best_score = i, score
    out = np.zeros((h, w), np.uint8)
    if best:
        out = (labels == best).astype(np.uint8)
        if roi is not None:  # a scar can have several nearby pieces; keep all sizeable ones in the box
            for i in range(1, n):
                if i != best and stats[i, cv2.CC_STAT_AREA] >= min_area:
                    out[labels == i] = 1
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(out)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        out = filled & search.astype(np.uint8)
    return out, bg, usable, search, thr


def _skin_ring(mask: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Band of normal skin just outside the scar, used as the texture reference."""
    area = int(mask.sum())
    r = int(max(8, np.sqrt(max(area, 1)) * 0.35))
    gap = max(3, r // 4)
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)))
    return (outer > 0) & (inner == 0) & ok


# --------------------------------------------------------------------------- #
# Texture (GLCM homogeneity restricted to a mask)
# --------------------------------------------------------------------------- #
def _glcm_homogeneity(L: np.ndarray, region: np.ndarray, levels: int = 32) -> float:
    ys, xs = np.nonzero(region)
    if len(ys) < 60:
        return float("nan")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = L[y0:y1, x0:x1]
    reg = region[y0:y1, x0:x1].astype(bool)
    # Local contrast normalisation so global brightness does not change texture.
    vals = crop[reg]
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    span = max(hi - lo, 12.0)  # avoid amplifying noise on flat regions
    q = np.clip((crop - lo) / span, 0, 0.999) * (levels - 1)
    q = q.astype(np.uint8) + 1          # levels 1..levels-1 are real pixels
    q[~reg] = 0                        # level 0 = outside the region
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels + 1, symmetric=True, normed=False).astype(np.float64)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1
    glcm /= sums
    return float(graycoprops(glcm, "homogeneity").mean())


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
def assess_quality(img_bgr: np.ndarray, lab: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(lab[..., 0].mean())
    q = ImageQuality(blur_score=blur, brightness=bright)
    if blur < 6:
        q.warnings.append("The photo looks blurry. Hold the phone steady and tap the scar to focus.")
    if bright < 22:
        q.warnings.append("The photo is too dark. Take it in daylight or a well-lit room.")
    elif bright > 93:
        q.warnings.append("The photo is too bright. Avoid direct flash or strong sunlight.")
    clipped = float((gray > 250).mean())
    if clipped > 0.08:
        q.warnings.append("Part of the photo is washed out by glare. Turn off the flash.")
    q.ok = not q.warnings
    return q


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def extract_features(img_bgr: np.ndarray, report: Optional[SelfReport] = None, roi: Optional[Roi] = None
                     ) -> Tuple[ScarStateVector, ImageQuality, np.ndarray]:
    report = report or SelfReport()
    img = _resize(img_bgr)
    lab = _to_lab(img)
    quality = assess_quality(img, lab)
    h, w = lab.shape[:2]

    mask, bg, usable, search, thr = segment_scar(img, lab, roi)
    area_px = int(mask.sum())
    faded = False

    if area_px == 0 and roi is None:
        quality.scar_found = False
        quality.ok = False
        quality.warnings.append(
            "We couldn't clearly find the scar. Mark the scar on the photo so we know where to look."
        )
        delta_a = delta_L = delta_b = de_val = 0.0
        tex = 1.0
        region = np.zeros((h, w), bool)
    else:
        if area_px == 0:
            # The patient marked where the scar is and nothing stands out from
            # the surrounding skin any more: the scar has faded. Measure the
            # centre of the marked area so the result reflects that.
            faded = True
            ys, xs = np.nonzero(search)
            cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
            ry, rx = max(2.0, (ys.max() - ys.min()) * 0.3), max(2.0, (xs.max() - xs.min()) * 0.3)
            yy, xx = np.mgrid[0:h, 0:w]
            region = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0) & usable
            if not region.any():
                region = search & usable
        else:
            region = mask.astype(bool)
        diff = lab[region] - bg[region]
        d = np.median(diff, axis=0) if len(diff) else np.zeros(3)
        delta_L = float(-d[0])      # + = scar darker than skin
        delta_a = float(d[1])       # + = scar redder than skin
        delta_b = float(d[2])
        de_val = float(np.sqrt(delta_L ** 2 + delta_a ** 2 + delta_b ** 2))
        if faded:
            de_val = min(de_val, thr)

        ring = _skin_ring(region.astype(np.uint8), usable & ~region)
        hom_scar = _glcm_homogeneity(lab[..., 0], region.astype(np.uint8))
        hom_skin = _glcm_homogeneity(lab[..., 0], ring.astype(np.uint8)) if ring.any() else float("nan")
        if faded or np.isnan(hom_scar):
            tex = 1.0
        elif np.isnan(hom_skin) or hom_skin <= 0:
            tex = _clip(hom_scar / 0.6, 0, 1)
        else:
            tex = _clip(hom_scar / hom_skin, 0, 1)

    # --- map raw measurements to clinical-style scales -------------------- #
    vascularity = _clip((delta_a - 1.5) / 5.5, 0, 3)           # a* 7 -> 1, 12.5 -> 2, 18 -> 3
    pigmentation = _clip((abs(delta_L) - 2.0) / 6.0, 0, 3)     # L* 8 -> 1, 14 -> 2, 20 -> 3

    height = _clip(float(report.height_level), 0, 3) if report.height_level is not None else 0.0
    pliability = 0.0
    if report.feels_tight:
        pliability += 3.0
    if report.painful:
        pliability += 0.5
    if report.growing_beyond_boundary:
        pliability += 0.5
    pliability = _clip(pliability, 0, 5)

    roi_frac = float(roi[2] * roi[3]) if roi else 0.0
    vec = ScarStateVector(
        vascularity=vascularity,
        pigmentation=pigmentation,
        pliability=pliability,
        height=height,
        surface_area=_clip(area_px / float(h * w), 0, 1),
        texture_regularity=tex,
        color_delta_e=de_val,
        raw={
            "delta_a": delta_a, "delta_L": delta_L, "delta_b": delta_b,
            "area_px": float(area_px), "threshold": thr, "faded": float(faded),
            "roi_frac": roi_frac, "marked": float(roi is not None),
            "height_reported": 0.0 if report.height_level is None else 1.0,
        },
    )
    return vec, quality, mask
