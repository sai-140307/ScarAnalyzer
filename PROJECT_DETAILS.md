# ScarAnalyzer: Intelligent Scar Tracking and Analysis System

Project Overview
ScarAnalyzer is a computer vision and data-driven system that enables individuals to track, classify, and analyze the healing progression of skin scars using nothing more than a smartphone camera. The system processes photographs of scars taken over time, extracts clinically relevant features, classifies the scar type, scores severity on standardized scales, models the healing trajectory, detects anomalies, and provides informational treatment category suggestions, all through a REST API that can be consumed by any mobile or web frontend.

The core problem ScarAnalyzer addresses is simple: most people with scars, whether from surgery, burns, acne, or trauma, have no objective way to know whether their scar is healing normally, stagnating, or getting worse. They rely on subjective memory ("I think it looked redder last month?"), and they visit a dermatologist only when something is visibly wrong. By the time a keloid is obviously growing or a contracture is limiting movement, the window for early intervention has narrowed significantly.

ScarAnalyzer closes this gap by turning a phone camera into a longitudinal clinical instrument. Each photograph is converted into a structured state vector of seven clinically grounded dimensions, vascularity, pigmentation, pliability, height, surface area, texture regularity, and color deviation from surrounding skin. Over multiple observations, these state vectors form a healing trajectory that the system analyzes for trends, reversals, stagnation, and accelerating growth patterns.

The system is explicitly designed to be informational, not diagnostic. It does not claim to replace dermatologists. Instead, it arms patients with objective data they can bring to medical appointments, and it flags concerning patterns early enough to prompt timely professional evaluation.

System Architecture
ScarAnalyzer is built as a modular pipeline architecture where each component has a single responsibility and a clean interface, allowing any module to be upgraded independently, for example, swapping the rule-based classifier for a deep learning model, without touching the rest of the system.

Module Overview
The system consists of eight core modules:

1. Schema Layer (schema.py)

Defines the foundational data structures that every other module operates on. The key entities are:

ScarStateVector: A seven-dimensional representation of a scar's physical state at a single point in time. The dimensions are vascularity (0-3, measuring redness from pale to purple), pigmentation (0-3, measuring deviation from normal skin color), pliability (0-5, from normal flexibility to permanent contracture), height (0-3, from flat to elevated above 5mm), surface area in square millimeters, texture regularity (0-1, from rough/irregular to smooth), and CIE Delta E color deviation from surrounding skin.
ScarObservation: A single photograph session linking a timestamp, image path, extracted state vector, and optional notes.
ScarProfile: The identity of a tracked scar, its location on the body, suspected cause, date of injury, classification, and the complete list of observations over time.
HealingTrajectory: A computed representation of the scar's journey through state space, including the time-ordered sequence of states, per-dimension velocity of change, and overall trend classification.
The schema uses Python dataclasses and enums for scar types (hypertrophic, keloid, atrophic, contracture, stretch mark, flat/mature) and body regions (face, neck, chest, abdomen, upper/lower back, arms, hands, legs, feet).

2. Feature Extractor (feature_extractor.py)

Transforms a raw photograph into a ScarStateVector. This module uses classical computer vision techniques from OpenCV and scikit-image rather than deep learning, providing a transparent and interpretable baseline.

The extraction pipeline works as follows:

The input image is converted to LAB color space, which separates luminance from color channels and aligns with how human vision perceives color differences.
Scar segmentation is performed using adaptive thresholding on the a* channel (red-green axis), which naturally highlights vascular scar tissue. Morphological operations clean up the resulting binary mask.
A surrounding skin mask is computed by dilating the scar mask and subtracting the original, creating a ring of normal skin for comparison.
Vascularity is computed from the mean a* value of scar pixels relative to the neutral point, mapped to the 0-3 Vancouver Scar Scale.
Pigmentation is computed from the L* (lightness) difference between scar and surrounding skin, capturing both hypopigmentation and hyperpigmentation.
Color deviation uses the CIE76 Delta E formula, Euclidean distance in LAB space between mean scar color and mean surrounding skin color.
Texture regularity uses Gray-Level Co-occurrence Matrix (GLCM) analysis, specifically the homogeneity property, which measures how uniform the texture is. A smooth scar scores close to 1, an irregular one close to 0.
Surface area is computed from the pixel count of the scar mask, optionally calibrated to physical millimeters if a reference object (ruler, coin) is detected in the frame.
Pliability and height cannot be determined from a 2D photograph alone and are set to 0 by default. These can be supplemented with patient self-reported values through the API.
3. Scar Classifier (classifier.py)

Determines what type of scar is present based on the extracted features and optional patient-reported metadata. The classifier uses a weighted rule-based scoring system grounded in clinical heuristics.

Each scar type has characteristic feature signatures. For example, keloids are distinguished by significant elevation (height ≥ 2.0), high vascularity, large surface area, and, critically, growth beyond the original wound boundary. Hypertrophic scars share some features but stay within the wound boundary and tend to regress over time. Atrophic scars (common from acne) are depressed below the skin surface. Contractures, typically from burns, show high pliability scores indicating tissue tightening.

The classifier computes a raw score for each scar type by accumulating evidence from the state vector dimensions and metadata, then normalizes these scores into probabilities using a temperature-controlled softmax function. It generates human-readable reasoning for its classification and flags cases for professional review when confidence is low, when the top two classifications are close, or when keloid formation is suspected.

A key feature is temporal classification: when multiple observations are available, the classifier extracts temporal features, whether area is increasing or decreasing, whether vascularity is trending down, whether pliability is worsening, and uses these to improve accuracy. A scar that was initially classified as hypertrophic but shows continued growth beyond six months is reclassified toward keloid, mirroring clinical reasoning.

4. Severity Scorer (severity.py)

Computes a standardized severity score modeled after the Vancouver Scar Scale (VSS) and the Patient and Observer Scar Assessment Scale (POSAS). Each dimension of the state vector is scored on its clinical scale, normalized to 0-1, and combined using clinically derived weights.

The weights reflect the relative clinical importance of each dimension: pliability (0.20) and vascularity (0.18) carry the most weight because they are the strongest indicators of active scar remodeling, followed by height (0.16), pigmentation (0.14), texture (0.12), and surface area and color deviation (0.10 each).

A scar type modifier adjusts the composite score to account for inherent severity differences, keloids receive a 1.2x multiplier, contractures 1.15x, while flat mature scars receive 0.7x.

The final composite score maps to a 0-100 scale and a clinical grade: minimal (0-15), mild (15-35), moderate (35-60), or severe (60-100). The module also computes which dimensions are contributing most to the severity score, enabling the suggestion engine to target its recommendations, and provides a change detection function that compares two severity assessments to identify improvement, worsening, or stability at both the per-dimension and aggregate level.

5. Trajectory Engine (trajectory.py)

This is the analytical core of the system. It takes the time-ordered sequence of ScarStateVectors and computes a comprehensive healing trajectory.

Per-dimension velocity is computed using linear regression over time (units per day). Acceleration is computed from a quadratic fit, indicating whether healing is speeding up or slowing down. Dimension-specific stability thresholds prevent noise from being misinterpreted as trends, surface area requires a velocity of at least 0.5 mm²/day to register as changing, while vascularity only needs 0.02 units/day.

The overall trend is a weighted aggregate of per-dimension trends, using the same clinical importance weights as the severity scorer. Only dimensions whose velocity exceeds their stability threshold contribute to the overall assessment.

The anomaly detection system identifies three concerning patterns:

Reversal: A dimension that was improving then suddenly worsened, using percentage-based thresholds relative to the value range to avoid false alarms on clinically insignificant fluctuations.
Stagnation: Minimal improvement (less than 5% change) across multiple key dimensions over more than 30 days, suggesting the scar has plateaued or that current management is insufficient.
Accelerating Growth: Surface area increasing at an accelerating rate, which is the hallmark pattern of keloid formation. This uses quadratic fit analysis and requires both meaningful absolute acceleration and positive velocity, with a minimum area threshold of 300 mm² to suppress false alarms on small scars.
The engine also supports population comparison, matching an individual scar's healing rate against population-level healing curves (when available) to report whether healing is faster or slower than typical and to estimate days until the scar reaches a plateau.

6. Suggestion Engine (suggestions.py)

Generates treatment category suggestions ranked by relevance to the specific scar's current state. The engine maintains a knowledge base of 11 treatment categories, from self-care options like silicone therapy, scar massage, and sun protection to professional interventions like corticosteroid injections, laser treatment, microneedling, and surgical revision.

Each treatment has defined applicability constraints: which scar types it applies to, what severity range it is appropriate for, which state vector dimensions it targets, whether it requires professional administration, and whether it should be started immediately or after the scar stabilizes.

Relevance scoring considers severity alignment (how well the scar's severity falls within the treatment's effective range), dimension driver matching (whether the scar's most problematic dimensions align with the treatment's target dimensions), alert boosting (treatments that address active alerts receive higher scores), trajectory context (treatments targeting worsening dimensions are boosted), and a slight accessibility preference for non-professional options.

The engine generates specific, personalized reasons for each suggestion, not generic descriptions but statements tied to the individual scar's measurements, such as "Your scar shows notable redness and elevation" or "Size trending upward."

General advice is generated based on severity level, trajectory direction, scar type, and active alerts, always ending with a clear disclaimer that the output is informational, not medical advice.

7. Storage Layer (storage.py)

A SQLite persistence layer that stores all scar data across sessions. The schema includes three tables: patients, scar_profiles, and observations. State vector dimensions are stored as flat columns rather than serialized JSON, enabling direct SQL queries for population statistics and timeline analysis.

The storage layer provides methods for CRUD operations on profiles and observations, querying scars by type, retrieving severity timelines for charting, and computing aggregate population statistics across all tracked scars of a given type. Write-ahead logging (WAL) mode is enabled for better concurrent read performance.

8. API Layer (api.py)

A FastAPI REST server exposing the complete pipeline over HTTP. The API supports the following operations:

POST /scars/register, Register a new scar for tracking
POST /scars/{scar_id}/observe, Upload a photograph and receive full analysis
GET /scars/{scar_id}/report, Comprehensive history report
GET /scars/{scar_id}/timeline, Severity scores over time for charting
GET /scars/{scar_id}/suggestions, Treatment category suggestions
GET /patients/{patient_id}/scars, List all tracked scars for a patient
GET /stats/{scar_type}, Population aggregate statistics
DELETE endpoints for scars and individual observations
Image uploads are handled via multipart form data. Patient-reported metadata (is the scar depressed, does it feel tight, is it growing beyond the wound boundary) can be submitted alongside each photograph to improve classification accuracy.

9. Pipeline Orchestrator (pipeline.py)

Wires all modules together into two primary operations:

process_observation(), The main entry point called each time a user takes a photo. Runs the full pipeline: extract features → classify type → score severity → compare to previous → update trajectory → detect anomalies → generate summary. Returns a complete ObservationReport with everything the frontend needs to display.
generate_full_report(), Produces a comprehensive FullScarReport covering the scar's entire history, including classification with full temporal analysis, current severity with dimensional breakdown, complete trajectory statistics, population comparison, anomaly alerts, severity timeline for charting, and a formatted summary.
10. Test Harness (test_pipeline.py)

A synthetic data generator and validation suite that tests the complete pipeline without requiring real images. The generator produces realistic scar observation sequences for six scenarios: hypertrophic healing, hypertrophic worsening (correctly reclassified as keloid), keloid growing, keloid under treatment, atrophic stable, and flat mature stable. Each scenario validates classification accuracy, trend detection, and severity direction against known expectations, achieving a 6/6 pass rate across all scenarios.

Technical Approach
Feature Extraction
The system uses LAB color space as its primary representation because it is perceptually uniform, a given numerical distance in LAB space corresponds to roughly the same perceived color difference regardless of where in the color space the measurement falls. This makes the CIE Delta E color deviation metric physiologically meaningful rather than just mathematically convenient.

Texture analysis uses GLCM (Gray-Level Co-occurrence Matrix) homogeneity, which captures the spatial distribution of pixel intensity pairs. Scars with irregular, rough surfaces produce lower homogeneity scores because neighboring pixels have more variable intensity relationships. This is more clinically relevant than simple variance or edge detection because it captures the structural irregularity that dermatologists palpate.

Classification Strategy
The rule-based classifier was chosen over a deep learning model for the initial implementation for three reasons: interpretability (every classification comes with human-readable reasoning), minimal data requirements (no training set needed), and clinical transparency (the scoring rules can be reviewed and validated by dermatologists). The system is architected so the classifier can be replaced with a trained model, convolutional neural network or vision transformer, using the same ClassificationResult interface, once sufficient labeled data is available.

Trajectory Modeling
The trajectory engine uses polynomial regression rather than more complex time-series models because scar healing trajectories are inherently low-frequency signals sampled sparsely (typically every 1-4 weeks). A linear fit captures the primary trend, and a quadratic fit captures acceleration/deceleration. More sophisticated models (Gaussian processes, recurrent networks) would overfit on the 5-15 observations typical of a scar's tracking lifetime.

Dimension-specific stability thresholds are critical for avoiding false trend signals. Surface area measured in mm² has fundamentally different noise characteristics than vascularity measured on a 0-3 scale. A one-unit-per-day velocity means nothing for surface area (normal fluctuation from slight differences in photo angle) but would be extreme for vascularity. The thresholds were tuned against synthetic trajectories to minimize false positive trend detections while maintaining sensitivity to clinically meaningful changes.

Anomaly Detection
The percentage-based approach to anomaly thresholds was adopted after testing revealed that absolute thresholds produce excessive false alarms on scars with small baseline values. A flat mature scar with a color Delta E of 3 that fluctuates to 4 is not experiencing a meaningful reversal, but a hypertrophic scar with a Delta E of 25 that drops to 15 then jumps back to 22 is. The percentage-based approach naturally adapts to the magnitude of the values being monitored.

Technology Stack
Component	Technology
Language	Python 3.9+
Image Processing	OpenCV, scikit-image
Numerical Computing	NumPy
API Framework	FastAPI with Uvicorn
Data Validation	Pydantic
Database	SQLite with WAL mode
Testing	Custom synthetic data generator and validator
Data Flow
User takes photo
       │
       ▼
  API receives image upload (POST /scars/{id}/observe)
       │
       ▼
  Feature Extractor
  ├── Convert to LAB color space
  ├── Segment scar from surrounding skin
  ├── Compute vascularity (a* channel analysis)
  ├── Compute pigmentation (L* channel comparison)
  ├── Compute color deviation (CIE Delta E)
  ├── Compute texture regularity (GLCM homogeneity)
  └── Compute surface area (pixel count with optional calibration)
       │
       ▼
  ScarStateVector (7 dimensions)
       │
       ▼
  Classifier
  ├── Score each scar type from features + metadata
  ├── Apply temporal signals if history exists
  ├── Normalize to probabilities (softmax)
  └── Flag for professional review if ambiguous
       │
       ▼
  Severity Scorer
  ├── Score each dimension on clinical scale
  ├── Apply clinical weights
  ├── Apply scar type modifier
  ├── Compute composite 0-100 score
  └── Compare to previous observation
       │
       ▼
  Trajectory Engine
  ├── Compute per-dimension velocity and acceleration
  ├── Determine per-dimension and overall trend
  ├── Detect anomalies (reversal, stagnation, accelerating growth)
  └── Compare to population baselines
       │
       ▼
  Suggestion Engine
  ├── Filter applicable treatments
  ├── Rank by relevance to this specific scar
  └── Generate personalized reasons and general advice
       │
       ▼
  Storage Layer (SQLite)
  ├── Persist observation and state vector
  └── Update scar type if classification confidence is high
       │
       ▼
  API Response (JSON)
  ├── Severity score and clinical grade
  ├── Classification with confidence and reasoning
  ├── Trend and trajectory data
  ├── Alerts if any
  └── Summary text

Limitations and Ethical Considerations
Skin Tone Bias: The current feature extraction relies on LAB color space analysis, which is more robust than RGB but still performs differently across skin tones. The a* channel analysis for vascularity and the L* comparison for pigmentation have not been validated across the full Fitzpatrick skin type spectrum. A production system would require extensive testing and calibration across diverse skin tones, particularly since keloid scars are significantly more prevalent in darker-skinned populations.

Not a Medical Device: ScarAnalyzer is explicitly informational. It does not diagnose conditions, prescribe treatments, or replace professional medical evaluation. All outputs include disclaimers. A production deployment in jurisdictions with medical device regulations (FDA in the US, CE marking in the EU) would require careful regulatory analysis to ensure the system does not cross the line into regulated medical device territory.

Measurement Limitations: Pliability and height cannot be determined from photographs. These dimensions rely on patient self-reporting, which introduces subjectivity. Future versions could explore structured light or stereo photography for height estimation.

Ground Truth Subjectivity: Clinical scar scales like VSS and POSAS have inter-rater reliability issues, two dermatologists scoring the same scar will disagree 30-40% of the time. The system's accuracy ceiling is bounded by this inherent subjectivity in the ground truth.

Cold Start: A new scar with a single observation has no trajectory data, limited classification accuracy, and no anomaly detection capability. The system becomes meaningfully useful after three or more observations spaced at least one to two weeks apart.

Future Enhancements
Deep Learning Classifier: Replace the rule-based classifier with a fine-tuned vision model trained on dermatological scar datasets (ISIC, DermNet) for improved classification accuracy.
Stereo/Depth Estimation: Use phone dual cameras or structured light to estimate scar height and three-dimensional surface topology from photographs.
Reference Object Detection: Automatic detection and calibration from coins, rulers, or standard reference cards placed next to the scar for accurate physical measurements.
Skin Tone Calibration: Fitzpatrick skin type detection and per-skin-type calibration curves for vascularity and pigmentation scoring.
Population Healing Curves: As the database grows, compute real population healing curves from anonymized aggregate data, enabling increasingly accurate population comparison.
Mobile Application: Native iOS/Android app with guided photo capture (consistent lighting, angle, and framing prompts) and push notification reminders for observation schedules.
FHIR Integration: Export scar tracking data in FHIR (Fast Healthcare Interoperability Resources) format for integration with electronic health record systems.
Muon Optimizer Integration: Apply the Muon optimizer to train the deep learning classifier's embedding matrices, leveraging its orthogonality enforcement via Newton-Schulz iteration to prevent embedding collapse, a known problem in medical image classification where subtle feature differences (week 4 vs week 8 scar texture) are lost during training.