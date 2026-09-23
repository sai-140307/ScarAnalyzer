"""Runtime configuration, read from environment variables so the same code
runs locally, in Docker, and on a cloud host (Render, Railway, a VM...)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("SCAR_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = Path(os.getenv("SCAR_DB_PATH", str(DATA_DIR / "scaranalyzer.db")))
UPLOAD_DIR = Path(os.getenv("SCAR_UPLOAD_DIR", str(DATA_DIR / "images")))
STATIC_DIR = Path(os.getenv("SCAR_STATIC_DIR", str(BASE_DIR / "static")))

MAX_UPLOAD_MB = float(os.getenv("SCAR_MAX_UPLOAD_MB", "12"))

# Comma separated list. "*" allows any origin (fine when the frontend is served
# by this same app; tighten it if you host the frontend elsewhere).
CORS_ORIGINS = [o.strip() for o in os.getenv("SCAR_CORS_ORIGINS", "*").split(",") if o.strip()]

# Longest side images are resized to before analysis (speed + consistency).
ANALYSIS_MAX_SIDE = int(os.getenv("SCAR_ANALYSIS_MAX_SIDE", "900"))

for _d in (DATA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
