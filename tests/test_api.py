"""HTTP-level tests with FastAPI's TestClient.  Run: python -m pytest tests -q"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

os.environ.setdefault("SCAR_DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from tests.synthetic import ScarSpec, encode_jpeg, make_photo  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, sid, spec, day, **fields):
    files = {"image": ("scar.jpg", encode_jpeg(make_photo(spec)), "image/jpeg")}
    data = {"taken_at": f"2026-07-{day:02d}", **{k: str(v).lower() for k, v in fields.items()}}
    return client.post(f"/scars/{sid}/observe", files=files, data=data)


def test_health_and_frontend(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "<html" in page.text.lower()


def test_worsening_flow(client):
    r = client.post("/scars/register", json={"patient_id": "test-patient", "body_region": "chest",
                                             "cause": "surgical", "date_of_injury": "2026-05-01"})
    assert r.status_code == 201
    sid = r.json()["scar_id"]
    statuses = []
    for i, (red, size) in enumerate([(9, .15), (12, .17), (16, .20)]):
        res = _upload(client, sid, ScarSpec(redness=red, size=size, seed=i), 1 + 7 * i, height_level=1)
        assert res.status_code == 201, res.text
        statuses.append(res.json()["report"]["status"])
    assert statuses[0] == "baseline"
    assert statuses[-1] == "worsening"
    rep = client.get(f"/scars/{sid}/report").json()
    assert rep["needs_professional_review"]
    obs = client.get(f"/scars/{sid}/observations").json()
    assert client.get(obs[0]["image_url"]).status_code == 200
    assert client.get("/patients/test-patient/scars").json()[0]["status"] == "worsening"
    assert client.get(f"/scars/{sid}/suggestions").json()["suggestions"]


def test_bad_upload(client):
    sid = client.post("/scars/register", json={"patient_id": "test-patient"}).json()["scar_id"]
    r = client.post(f"/scars/{sid}/observe", files={"image": ("x.jpg", b"nope", "image/jpeg")})
    assert r.status_code == 422
    assert client.get("/scars/doesnotexist/report").status_code == 404
