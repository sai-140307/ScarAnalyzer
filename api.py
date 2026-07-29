import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from schema import ScarType, BodyRegion
from pipeline import ScarAnalysisPipeline

# config
UPLOAD_DIR = 'uploads'
DB_PATH = 'scars.db'

os.makedirs(UPLOAD_DIR, exist_ok=True)

# app setup
app = FastAPI(
    title='Scar Analyzer API',
    description='Track, classify, and analyze scar healing over time.',
    version='0.1.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/uploads', StaticFiles(directory=UPLOAD_DIR), name='uploads')

pipeline = ScarAnalysisPipeline(db_path=DB_PATH)


# request/response models
class RegisterScarRequest(BaseModel):
    patient_id: str
    body_region: str
    cause: Optional[str] = None
    date_of_injury: Optional[str] = None


class RegisterScarResponse(BaseModel):
    scar_id: str
    message: str


class ObservationResponse(BaseModel):
    observation_id: str
    severity_score: float
    clinical_grade: str
    scar_type: str
    classification_confidence: float
    trend: Optional[str] = None
    alerts: List[Dict[str, Any]] = []
    severity_change: Optional[Dict[str, Any]] = None
    summary: str
    needs_professional_review: bool


class ScarListItem(BaseModel):
    scar_id: str
    body_region: str
    scar_type: str
    cause: Optional[str] = None
    observation_count: int
    first_observed: Optional[str] = None
    last_observed: Optional[str] = None
    current_severity: Optional[float] = None
    clinical_grade: Optional[str] = None


class FullReportResponse(BaseModel):
    scar_id: str
    body_region: str
    scar_type: str
    classification_confidence: float
    cause: Optional[str] = None
    observation_count: int
    scar_age_days: Optional[int] = None
    current_severity: float
    clinical_grade: str
    trend: str
    dimension_stats: List[Dict[str, Any]] = []
    severity_timeline: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []
    needs_professional_review: bool
    summary: str


# endpoints
@app.get('/health')
def health_check():
    return {'status': 'ok', 'version': '0.1.0'}


@app.post('/scars/register', response_model=RegisterScarResponse)
def register_scar(request: RegisterScarRequest):
    try:
        body_region = BodyRegion(request.body_region)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail='Invalid body_region: {}. Valid options: {}'.format(
                request.body_region,
                [r.value for r in BodyRegion],
            ),
        )

    date_of_injury = None
    if request.date_of_injury:
        try:
            date_of_injury = datetime.fromisoformat(request.date_of_injury)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail='Invalid date_of_injury format. Use ISO format: YYYY-MM-DD',
            )

    scar_id = pipeline.register_scar(
        patient_id=request.patient_id,
        body_region=body_region,
        cause=request.cause,
        date_of_injury=date_of_injury,
    )

    return RegisterScarResponse(
        scar_id=scar_id,
        message='Scar registered. Upload your first observation photo.',
    )


@app.post('/scars/{scar_id}/observe', response_model=ObservationResponse)
async def add_observation(
    scar_id: str,
    image: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    is_depressed: Optional[bool] = Form(None),
    feels_tight: Optional[bool] = Form(None),
    growing_beyond_boundary: Optional[bool] = Form(None),
    reference_scale_mm_per_px: Optional[float] = Form(None),
):
    if image.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(
            status_code=400,
            detail='Image must be JPEG, PNG, or WebP',
        )

    file_ext = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
    image_filename = '{}_{}.{}'.format(scar_id, uuid.uuid4().hex[:8], file_ext)
    image_path = os.path.join(UPLOAD_DIR, image_filename)

    with open(image_path, 'wb') as f:
        shutil.copyfileobj(image.file, f)

    metadata = {}
    if is_depressed is not None:
        metadata['is_depressed'] = is_depressed
    if feels_tight is not None:
        metadata['feels_tight'] = feels_tight
    if growing_beyond_boundary is not None:
        metadata['growing_beyond_boundary'] = growing_beyond_boundary

    try:
        report = pipeline.process_observation(
            scar_id=scar_id,
            image_path=image_path,
            patient_metadata=metadata if metadata else None,
            notes=notes,
            reference_scale_mm_per_px=reference_scale_mm_per_px,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ObservationResponse(
        observation_id=report.observation.observation_id,
        severity_score=report.severity.composite_score,
        clinical_grade=report.severity.clinical_grade,
        scar_type=report.classification.predicted_type.value,
        classification_confidence=report.classification.confidence,
        trend=report.trajectory.trend if report.trajectory else None,
        alerts=report.alerts,
        severity_change=report.severity_change,
        summary=report.summary,
        needs_professional_review=report.classification.needs_professional_review,
    )


@app.get('/scars/{scar_id}/report', response_model=FullReportResponse)
def get_full_report(scar_id: str):
    try:
        report = pipeline.generate_full_report(scar_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    scar_age_days = None
    if report.profile.date_of_injury and report.profile.observations:
        latest = report.profile.observations[-1].timestamp
        scar_age_days = (latest - report.profile.date_of_injury).days

    dimension_stats = []
    for stat in report.detailed_stats:
        dimension_stats.append(
            {
                'dimension': stat.dimension,
                'trend': stat.trend,
                'velocity': stat.velocity,
                'acceleration': stat.acceleration,
                'total_change': stat.total_change,
                'confidence': stat.confidence,
            }
        )

    return FullReportResponse(
        scar_id=scar_id,
        body_region=report.profile.body_region.value,
        scar_type=report.classification.predicted_type.value,
        classification_confidence=report.classification.confidence,
        cause=report.profile.cause,
        observation_count=len(report.profile.observations),
        scar_age_days=scar_age_days,
        current_severity=report.current_severity.composite_score,
        clinical_grade=report.current_severity.clinical_grade,
        trend=report.trajectory.trend,
        dimension_stats=dimension_stats,
        severity_timeline=report.severity_timeline,
        alerts=report.alerts,
        needs_professional_review=report.classification.needs_professional_review,
        summary=report.summary,
    )


@app.get('/scars/{scar_id}/timeline')
def get_severity_timeline(scar_id: str):
    profile = pipeline.db.get_profile(scar_id)
    if not profile:
        raise HTTPException(status_code=404, detail='Scar not found')

    timeline = pipeline._build_severity_timeline(profile)
    return {'scar_id': scar_id, 'timeline': timeline}


@app.get('/patients/{patient_id}/scars', response_model=List[ScarListItem])
def list_patient_scars(patient_id: str):
    scars = pipeline.get_all_scars(patient_id)
    if not scars:
        return []
    return scars


@app.delete('/scars/{scar_id}')
def delete_scar(scar_id: str):
    profile = pipeline.db.get_profile(scar_id)
    if not profile:
        raise HTTPException(status_code=404, detail='Scar not found')

    for obs in profile.observations:
        if os.path.exists(obs.image_path):
            os.remove(obs.image_path)

    pipeline.db.delete_scar(scar_id)
    return {'message': 'Scar {} deleted'.format(scar_id)}


@app.delete('/scars/{scar_id}/observations/{observation_id}')
def delete_observation(scar_id: str, observation_id: str):
    pipeline.db.delete_observation(observation_id)
    return {'message': 'Observation {} deleted'.format(observation_id)}


@app.get('/scars/{scar_id}/suggestions')
def get_suggestions(scar_id: str):
    """Get treatment category suggestions for a scar."""
    try:
        report = pipeline.get_suggestions(scar_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        'scar_id': scar_id,
        'suggestions': [
            {
                'category': s.category,
                'relevance': round(s.relevance, 2),
                'reason': s.reason,
                'timing': s.timing,
                'professional_required': s.professional_required,
                'description': s.description,
            }
            for s in report.suggestions
        ],
        'general_advice': report.general_advice,
        'disclaimer': report.disclaimer,
    }


@app.get('/stats/{scar_type}')
def get_population_stats(scar_type: str):
    try:
        st = ScarType(scar_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail='Invalid scar_type. Valid options: {}'.format(
                [t.value for t in ScarType],
            ),
        )

    stats = pipeline.db.get_population_stats(st)
    return {'scar_type': scar_type, 'stats': stats}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
