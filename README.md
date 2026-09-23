# ScarAnalyzer

Track how a scar heals over time using smartphone photos. Patients add a scar,
upload a photo every 1–2 weeks, mark where the scar is on the photo, and see whether it is **healing well**,
**not changing**, or **getting worse**, with care suggestions and a prompt to
see a dermatologist when needed.

The FastAPI backend serves both the REST API and the web app from one address,
so the same code runs on a laptop, in Docker, or on a cloud host.

> ScarAnalyzer is informational only and is not a medical device.

## Project layout

```
app/
  main.py               FastAPI app: REST API + serves the web app
  config.py             settings from environment variables
  schemas.py            ScarStateVector (7 dimensions), self-report, directions
  feature_extractor.py  photo -> LAB -> segmentation -> colour/texture/area
  classifier.py         rule-based scar type (normal/hypertrophic/keloid/atrophic/contracture)
  severity.py           0–100 composite score (VSS/POSAS-weighted)
  trajectory.py         trend, worsening detection, alerts
  suggestions.py        evidence-ranked care suggestions
  pipeline.py           builds the full report from all observations
  storage.py            SQLite (WAL) + image files
static/index.html       patient web app
tests/                  synthetic photo generator + validation suites
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the app and http://127.0.0.1:8000/docs for the
interactive API documentation.

To test on your phone on the same Wi-Fi, run
`uvicorn app.main:app --host 0.0.0.0 --port 8000` and open
`http://<your-computer-ip>:8000` on the phone.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
# or individually, with readable output:
python tests/test_scenarios.py
python tests/test_robustness.py
```

## Deploy

### Docker (any server or VM)

```bash
docker build -t scaranalyzer .
docker run -d -p 8000:8000 -v scar-data:/data scaranalyzer
```

### Render

1. Push this repository to GitHub.
2. On Render choose **New → Blueprint** and select the repository. `render.yaml`
   creates the web service with a persistent disk for the database and photos.
3. Open the URL Render gives you (e.g. `https://scaranalyzer.onrender.com`).

Without a persistent disk, data is lost whenever the service restarts, so keep
the disk for anything beyond a demo.

### Railway / Heroku-style hosts

The `Procfile` starts the app on `$PORT`. Set `SCAR_DATA_DIR` to a mounted
volume path.

### Hosting the web app separately (optional)

If you put `static/index.html` on Netlify, GitHub Pages, etc., set the API
address in the page:

```html
<meta name="scar-api" content="https://your-api.onrender.com">
```

and set `SCAR_CORS_ORIGINS` on the API to that site's address.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SCAR_DATA_DIR` | `./data` | Folder for the database and photos |
| `SCAR_DB_PATH` | `$SCAR_DATA_DIR/scaranalyzer.db` | SQLite file |
| `SCAR_UPLOAD_DIR` | `$SCAR_DATA_DIR/images` | Stored photos |
| `SCAR_MAX_UPLOAD_MB` | `12` | Maximum photo size |
| `SCAR_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PORT` | `8000` | Port (Docker / Procfile) |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/scars/register` | Create a scar (`patient_id`, `nickname`, `body_region`, `cause`, `date_of_injury`) |
| GET | `/patients/{patient_id}/scars` | A patient's scars with latest status |
| GET / PATCH / DELETE | `/scars/{id}` | Read, edit, delete a scar |
| POST | `/scars/{id}/observe` | Upload a photo (multipart: `image`, optional `taken_at`, `height_level` 0–3, `is_depressed`, `feels_tight`, `itchy`, `painful`, `growing_beyond_boundary`, `notes`, and `roi` = the marked scar area as `x,y,w,h` fractions of the image, strongly recommended). Returns the observation and the updated report |
| GET | `/scars/{id}/observations` | All photos for a scar |
| GET | `/observations/{id}/image` | Photo file |
| DELETE | `/observations/{id}` | Delete a photo (report is recalculated) |
| GET | `/scars/{id}/report` | Full report: status, severity, changes, alerts, suggestions |
| GET | `/scars/{id}/suggestions` | Care suggestions only |

## How worsening is detected

The earlier version could report a worsening scar as "healing". The current
trajectory engine fixes this by:

0. **The patient marks the scar.** Real photos contain hair, eyebrows, faces
   and backgrounds. The scar is compared only with the skin right around it
   (hair, brows and background are excluded), and if nothing stands out in the
   marked area any more the scar is recorded as *faded*, i.e. healed.
1. **Direction per dimension.** Rising redness, size, colour difference,
   height or tightness is *worse*; rising smoothness is *better*.
2. **Relative colour.** Colour is measured against the surrounding skin in the
   same photo, so different lighting or phones don't look like change.
3. **Correct ordering.** Photos are ordered by the date taken (from the form or
   the photo's EXIF), not upload order; same-day uploads are handled.
4. **Worsening checked first.** Any of: rising fitted severity, a jump since the
   last photo, confirmed size growth, new pain/tightness, or the patient
   reporting spread marks the scar as worsening. A long improving history
   cannot hide a recent turn for the worse.
5. **Noise-aware thresholds.** Absolute and percentage thresholds per
   dimension; size growth alone needs stronger evidence because camera distance
   changes apparent size. If the visible area more than doubles within two
   months (or the marked box changes 4x), the photos are treated as taken at
   different distances and size is not compared.

## Validation (synthetic photos)

Photos vary in lighting, white balance, angle and ±12% camera distance.

| Case | Result |
|---|---|
| Stable scar, 2 or 4 photos (never flagged worse) | 100% |
| Mild worsening, 2 or 4 photos | 100% |
| Mild improvement, 4 photos | 93–100% |
| 19 clinical scenarios (reversal, keloid growth, dark skin, out-of-order uploads, atrophic, stagnating, fully healed scar, zoom change...) | 19/19 |
| Real forehead scar, close-up then full-face photo one month later | "Healing well", score 26 → 2 |

Apart from the one real example, these results are on generated images;
accuracy on real patient photos still needs to be measured with
clinician-labelled data.

## Limitations

- No user accounts: each browser gets a private tracking code (shown under
  "Use on another device"). Add authentication before real patient use.
- Height and tightness come from the patient's answers; a 2D photo can't
  measure them.
- Photos analysed by an older version (before scar marking) should be deleted
  and uploaded again with the scar marked.
- Size depends on camera distance; a reference object (e.g. a coin) would allow
  real millimetre measurements.
- Colour calibration has not been validated across the full Fitzpatrick range
  on real photos.
