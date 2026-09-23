"""SQLite persistence (WAL mode). One short-lived connection per operation,
so it is safe with FastAPI's threadpool."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from .config import DB_PATH, UPLOAD_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS scars (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    nickname TEXT,
    body_region TEXT,
    cause TEXT,
    date_of_injury TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scars_patient ON scars(patient_id);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    scar_id TEXT NOT NULL REFERENCES scars(id) ON DELETE CASCADE,
    taken_at TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    image_path TEXT,
    vector TEXT NOT NULL,
    report TEXT,
    quality TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_scar ON observations(scar_id, taken_at);
"""


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as c:
        c.execute("PRAGMA journal_mode = WAL")
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
def create_scar(patient_id: str, nickname: Optional[str], body_region: str,
                cause: Optional[str], date_of_injury: Optional[str]) -> dict:
    sid = uuid.uuid4().hex[:12]
    row = dict(id=sid, patient_id=patient_id, nickname=nickname, body_region=body_region,
               cause=cause, date_of_injury=date_of_injury, created_at=_now())
    with connect() as c:
        c.execute("INSERT INTO scars VALUES (:id,:patient_id,:nickname,:body_region,:cause,:date_of_injury,:created_at)", row)
    return row


def get_scar(scar_id: str) -> Optional[dict]:
    with connect() as c:
        r = c.execute("SELECT * FROM scars WHERE id = ?", (scar_id,)).fetchone()
    return dict(r) if r else None


def list_scars(patient_id: str) -> List[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM scars WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
    return [dict(r) for r in rows]


def update_scar(scar_id: str, **fields) -> Optional[dict]:
    allowed = {k: v for k, v in fields.items() if k in ("nickname", "body_region", "cause", "date_of_injury")}
    if allowed:
        sets = ", ".join(f"{k} = :{k}" for k in allowed)
        with connect() as c:
            c.execute(f"UPDATE scars SET {sets} WHERE id = :id", {**allowed, "id": scar_id})  # noqa: S608
    return get_scar(scar_id)


def delete_scar(scar_id: str) -> bool:
    for o in list_observations(scar_id):
        _remove_image(o.get("image_path"))
    with connect() as c:
        c.execute("DELETE FROM observations WHERE scar_id = ?", (scar_id,))
        cur = c.execute("DELETE FROM scars WHERE id = ?", (scar_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
def save_image(scar_id: str, jpeg_bytes: bytes) -> str:
    folder = UPLOAD_DIR / scar_id
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:12]}.jpg"
    (folder / name).write_bytes(jpeg_bytes)
    return f"{scar_id}/{name}"


def image_file(rel: str) -> Optional[Path]:
    p = (UPLOAD_DIR / rel).resolve()
    if UPLOAD_DIR.resolve() not in p.parents or not p.exists():
        return None
    return p


def _remove_image(rel: Optional[str]) -> None:
    if rel:
        p = image_file(rel)
        if p:
            p.unlink(missing_ok=True)


def add_observation(scar_id: str, taken_at: datetime, image_path: Optional[str], vector: dict,
                    report: dict, quality: dict, notes: Optional[str]) -> dict:
    oid = uuid.uuid4().hex[:12]
    row = dict(id=oid, scar_id=scar_id, taken_at=taken_at.replace(microsecond=0).isoformat(),
               uploaded_at=_now(), image_path=image_path, vector=json.dumps(vector),
               report=json.dumps(report), quality=json.dumps(quality), notes=notes)
    with connect() as c:
        c.execute("INSERT INTO observations VALUES (:id,:scar_id,:taken_at,:uploaded_at,:image_path,"
                  ":vector,:report,:quality,:notes)", row)
    return _decode(row)


def _decode(r) -> dict:
    d = dict(r)
    for k in ("vector", "report", "quality"):
        d[k] = json.loads(d[k]) if d.get(k) else {}
    return d


def list_observations(scar_id: str) -> List[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM observations WHERE scar_id = ? ORDER BY taken_at, uploaded_at",
                         (scar_id,)).fetchall()
    return [_decode(r) for r in rows]


def get_observation(obs_id: str) -> Optional[dict]:
    with connect() as c:
        r = c.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
    return _decode(r) if r else None


def delete_observation(obs_id: str) -> bool:
    o = get_observation(obs_id)
    if not o:
        return False
    _remove_image(o.get("image_path"))
    with connect() as c:
        c.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
    return True
