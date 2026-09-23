"""ScarAnalyzer REST API (FastAPI).

Serves both the JSON API and the patient web app from one origin, so the same
deployment works on a laptop, in Docker, or on a cloud host with no changes.

Run locally:   uvicorn app.main:app --reload
Production:    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import storage
from .config import CORS_ORIGINS, MAX_UPLOAD_MB, STATIC_DIR
from .feature_extractor import ImageDecodeError, decode_image, extract_features, parse_roi
from .pipeline import DISCLAIMER, build_report
from .schemas import SelfReport

@asynccontextmanager
async def lifespan(_app):
    storage.init_db()
    yield


app = FastAPI(title="ScarAnalyzer API", version="2.0.0", lifespan=lifespan,
              description="Track how a scar changes over time from smartphone photos.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class RegisterScar(BaseModel):
    patient_id: str = Field(..., min_length=3, max_length=64)
    nickname: Optional[str] = Field(None, max_length=60)
    body_region: str = Field("other", max_length=40)
    cause: Optional[str] = Field(None, max_length=40)
    date_of_injury: Optional[date] = None


class UpdateScar(BaseModel):
    nickname: Optional[str] = Field(None, max_length=60)
    body_region: Optional[str] = Field(None, max_length=40)
    cause: Optional[str] = Field(None, max_length=40)
    date_of_injury: Optional[date] = None


def _scar_out(s: dict) -> dict:
    return {k: s.get(k) for k in ("id", "patient_id", "nickname", "body_region", "cause", "date_of_injury",
                                  "created_at")} | {"scar_id": s["id"]}


def _obs_out(o: dict) -> dict:
    return {"id": o["id"], "scar_id": o["scar_id"], "taken_at": o["taken_at"], "uploaded_at": o["uploaded_at"],
            "image_url": f"/observations/{o['id']}/image" if o.get("image_path") else None,
            "notes": o.get("notes"), "self_report": o.get("report") or {},
            "photo_ok": bool((o.get("quality") or {}).get("scar_found", True)),
            "photo_tips": (o.get("quality") or {}).get("warnings", []),
            "roi": (o.get("quality") or {}).get("roi"),
            "faded": bool((o.get("quality") or {}).get("faded"))}


def _require_scar(scar_id: str) -> dict:
    s = storage.get_scar(scar_id)
    if not s:
        raise HTTPException(404, "Scar not found.")
    return s


def _bool(v: Optional[str]) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if v is not None else False


def _parse_taken_at(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(v.strip()[:19])
    except ValueError:
        raise HTTPException(422, "taken_at must be a date like 2026-05-01.")
    if dt > datetime.now():
        raise HTTPException(422, "The photo date can't be in the future.")
    return dt


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/scars/register", status_code=201)
def register_scar(body: RegisterScar) -> dict:
    s = storage.create_scar(body.patient_id.strip(), (body.nickname or "").strip() or None,
                            body.body_region.strip().lower(), (body.cause or "").strip().lower() or None,
                            body.date_of_injury.isoformat() if body.date_of_injury else None)
    return _scar_out(s)


@app.get("/patients/{patient_id}/scars")
def list_scars(patient_id: str) -> List[dict]:
    out = []
    for s in storage.list_scars(patient_id):
        obs = storage.list_observations(s["id"])
        rep = build_report(s, obs)
        latest = obs[-1] if obs else None
        out.append({**_scar_out(s), "observation_count": len(obs), "status": rep["status"],
                    "headline": rep["headline"], "current_severity": rep["current_severity"],
                    "needs_professional_review": rep["needs_professional_review"],
                    "last_photo_at": latest["taken_at"] if latest else None,
                    "thumbnail_url": f"/observations/{latest['id']}/image" if latest else None})
    return out


@app.get("/scars/{scar_id}")
def get_scar(scar_id: str) -> dict:
    return _scar_out(_require_scar(scar_id))


@app.patch("/scars/{scar_id}")
def update_scar(scar_id: str, body: UpdateScar) -> dict:
    _require_scar(scar_id)
    data = body.model_dump(exclude_unset=True)
    if data.get("date_of_injury"):
        data["date_of_injury"] = data["date_of_injury"].isoformat()
    return _scar_out(storage.update_scar(scar_id, **data))


@app.delete("/scars/{scar_id}", status_code=204)
def delete_scar(scar_id: str) -> None:
    if not storage.delete_scar(scar_id):
        raise HTTPException(404, "Scar not found.")


@app.post("/scars/{scar_id}/observe", status_code=201)
def observe(
    scar_id: str,
    image: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    taken_at: Optional[str] = Form(None),
    height_level: Optional[str] = Form(None),
    is_depressed: Optional[str] = Form(None),
    feels_tight: Optional[str] = Form(None),
    growing_beyond_boundary: Optional[str] = Form(None),
    itchy: Optional[str] = Form(None),
    painful: Optional[str] = Form(None),
    roi: Optional[str] = Form(None),
) -> dict:
    scar = _require_scar(scar_id)
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = image.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"The photo is larger than {MAX_UPLOAD_MB:.0f} MB. Choose a smaller photo.")
    if not data:
        raise HTTPException(422, "The photo is empty.")

    try:
        img, exif_time = decode_image(data)
    except ImageDecodeError as e:
        raise HTTPException(422, str(e))

    hl = None
    if height_level not in (None, "", "null"):
        try:
            hl = max(0, min(3, int(height_level)))
        except ValueError:
            raise HTTPException(422, "height_level must be 0, 1, 2 or 3.")
    report = SelfReport(height_level=hl, is_depressed=_bool(is_depressed), feels_tight=_bool(feels_tight),
                        growing_beyond_boundary=_bool(growing_beyond_boundary), itchy=_bool(itchy),
                        painful=_bool(painful))

    try:
        box = parse_roi(roi)
    except ValueError as e:
        raise HTTPException(422, str(e))

    vector, quality, _ = extract_features(img, report, box)

    # Store a privacy-safe copy: resized, re-encoded, with EXIF (GPS etc.) removed.
    h, w = img.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=88)
    rel = storage.save_image(scar_id, buf.getvalue())

    when = _parse_taken_at(taken_at) or exif_time or datetime.now()
    obs = storage.add_observation(
        scar_id, when, rel, vector.to_json(), report.to_dict(),
        {"ok": quality.ok, "scar_found": quality.scar_found, "warnings": quality.warnings,
         "blur_score": round(quality.blur_score, 1), "brightness": round(quality.brightness, 1),
         "roi": list(box) if box else None, "faded": bool(vector.raw.get("faded"))},
        (notes or "").strip()[:500] or None)

    full = build_report(scar, storage.list_observations(scar_id))
    return {"observation": _obs_out(obs), "report": full}


@app.get("/scars/{scar_id}/observations")
def list_observations(scar_id: str) -> List[dict]:
    _require_scar(scar_id)
    return [_obs_out(o) for o in storage.list_observations(scar_id)]


@app.get("/observations/{obs_id}/image")
def observation_image(obs_id: str):
    o = storage.get_observation(obs_id)
    p = storage.image_file(o["image_path"]) if o and o.get("image_path") else None
    if not p:
        raise HTTPException(404, "Photo not found.")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.delete("/observations/{obs_id}", status_code=204)
def delete_observation(obs_id: str) -> None:
    if not storage.delete_observation(obs_id):
        raise HTTPException(404, "Photo not found.")


@app.get("/scars/{scar_id}/report")
def report(scar_id: str) -> dict:
    s = _require_scar(scar_id)
    return build_report(s, storage.list_observations(scar_id))


@app.get("/scars/{scar_id}/suggestions")
def suggestions(scar_id: str) -> dict:
    s = _require_scar(scar_id)
    r = build_report(s, storage.list_observations(scar_id))
    advice = ("Take your next photo in 1–2 weeks, in the same light and from the same distance."
              if r["status"] != "worsening" else
              "Keep taking photos every week and bring them to your appointment.")
    return {"suggestions": r["suggestions"], "general_advice": advice, "disclaimer": DISCLAIMER}


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our side. Please try again."})


# --------------------------------------------------------------------------- #
# Web app (must be mounted last so API routes take priority)
# --------------------------------------------------------------------------- #
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")
