# ScarAnalyzer – Coding Standards Document

**Software Engineering Laboratory – Assessment 5**

## 1. Document Purpose and Project Context

This document defines the coding standards and implementation practices used for ScarAnalyzer. It directly addresses the five requirements of Assessment 5: programming language and development environment, coding standards, project structure, version control, and initial module implementation.

ScarAnalyzer is a modular, Python-based computer-vision and data-analysis backend for longitudinal scar tracking and analysis. The repository separates data models, image feature extraction, classification, severity scoring, trajectory analysis, suggestions, persistence, API delivery, orchestration, and testing.

## 2. Programming Language and Development Environment

| Area | Implementation |
|---|---|
| Language | Python 3.9+ |
| Computer Vision | OpenCV and scikit-image |
| Numerical Computing | NumPy |
| API Framework | FastAPI with Uvicorn |
| Data Validation | Pydantic |
| Database | SQLite with WAL mode |
| Testing | Custom synthetic data generator and validation suite |
| Source Control | Git with GitHub repository |

### Development conventions

- Focused Python modules with clear interfaces.
- Dataclasses and enums for core data structures.
- Pydantic models for API request/response contracts.
- A reusable `ScarAnalysisPipeline` for business workflow orchestration.
- SQLite persistence for lightweight, reproducible storage.

## 3. Coding Standards Followed

### Naming and formatting

- Four-space indentation and readable line breaks.
- `snake_case` for functions and variables.
- `PascalCase` for classes.
- Descriptive names such as `ScarFeatureExtractor` and `ScarAnalysisPipeline`.

### Modularity and responsibility

Each module has a distinct responsibility: schema, feature extraction, classifier, severity, trajectory, suggestions, storage, API, pipeline, or testing.

### Documentation

Core classes and public methods use docstrings to explain purpose, inputs, outputs, and implementation assumptions. Algorithm-specific assumptions are documented where they affect interpretation, such as LAB color analysis and the limits of 2D images for measuring pliability and height.

### Explicit configuration

Thresholds and weights are kept as named constants, including `REVIEW_THRESHOLD`, `DIMENSION_WEIGHTS`, `TYPE_MODIFIERS`, and `GRADE_THRESHOLDS`.

### Error handling

The implementation validates image loading and scar identifiers and raises meaningful exceptions for missing images and unknown scar IDs.

### Testability

The repository includes synthetic longitudinal scenarios so the end-to-end pipeline can be exercised without real photographs during initial development.

## 4. Project Structure and Module Organization

```text
ScarAnalyzer/
├── api.py
├── classifier.py
├── feature_extractor.py
├── pipeline.py
├── schema.py
├── severity.py
├── storage.py
├── suggestions.py
├── trajectory.py
├── test_pipeline.py
├── PROJECT_DETAILS.md
├── .gitignore
└── uploads/
```

### Module responsibilities

- **`schema.py`** – Defines `ScarType`, `BodyRegion`, `ScarStateVector`, `ScarObservation`, `ScarProfile`, and `HealingTrajectory`.
- **`feature_extractor.py`** – Loads images, converts to LAB, segments scar regions, and computes vascularity, pigmentation, Delta E, texture regularity, and surface area.
- **`classifier.py`** – Uses weighted rule-based heuristics plus optional metadata/history to classify scar type and produce confidence/reasoning.
- **`severity.py`** – Produces normalized dimension scores, weighted composite severity, clinical grade, and severity change analysis.
- **`trajectory.py`** – Computes longitudinal trends, rates of change, acceleration, and anomaly patterns.
- **`suggestions.py`** – Ranks informational treatment categories using current severity, drivers, alerts, and trajectory context.
- **`storage.py`** – Persists patients, scar profiles, and observations in SQLite and supports timeline/statistical queries.
- **`pipeline.py`** – Orchestrates the full workflow from extraction through classification, scoring, trajectory analysis, alerts, persistence, and summary generation.
- **`api.py`** – Exposes the analysis workflow through FastAPI REST endpoints.
- **`test_pipeline.py`** – Generates synthetic longitudinal cases and validates classification, trend, and severity behavior.

## 5. Version Control Setup and Git Strategy

The project is maintained in the public GitHub repository `sai-140307/ScarAnalyzer`. The verified default branch is `main`.

### Branching strategy

- `main` – stable integration branch.
- `feature/<name>` – short-lived branches for isolated features or modules.
- `fix/<name>` – focused bug-fix branches.
- Merge changes back into `main` after validation/tests pass.

### Commit practice

The repository includes the descriptive initial commit message:

> add ScarAnalyzer project files and initial test harness

Future commits should remain concise, action-oriented, and limited to one logical change, for example `add trajectory stagnation detection` or `refine API observation response`.

### Repository hygiene

- Keep generated databases, runtime artifacts, virtual environments, caches, and other local files excluded through `.gitignore` where appropriate.
- Never commit secrets, credentials, API keys, or patient-identifying data.
- Keep commits reviewable and related to coherent changes.
- Use pull requests when formal review or collaboration is required.

## 6. Initial Module Implementation

The initial implementation establishes a common data contract and a complete modular processing pipeline.

### 6.1 Data model

`schema.py` defines `ScarStateVector` with seven tracked dimensions: vascularity, pigmentation, pliability, height, surface area, texture regularity, and CIE Delta E color deviation. `ScarObservation` additionally stores the timestamp, image path, notes, and reference-scale status.

### 6.2 Image analysis

`feature_extractor.py` uses OpenCV and scikit-image to convert images to LAB, perform a* based scar segmentation with morphological cleanup, create a surrounding-skin mask, and calculate image-derived features. Pliability and height default to zero because they cannot be reliably determined from a 2D photograph alone in the current implementation.

### 6.3 Classification

`classifier.py` provides weighted rule-based classification for hypertrophic, keloid, atrophic, contracture, stretch-mark, and flat/mature scar categories. It can use patient metadata and observation history and returns probabilities, confidence, reasoning, and a professional-review flag.

### 6.4 Severity scoring

`severity.py` normalizes the seven dimensions, applies explicit weights and scar-type modifiers, and maps the resulting composite score to minimal, mild, moderate, or severe grades. It also compares current and previous assessments.

### 6.5 End-to-end integration

`pipeline.py` provides the main entry points `process_observation()` and `generate_full_report()`. The observation path follows:

**extract features → classify → score severity → compare → persist → analyze trajectory → detect alerts → summarize**

### 6.6 Validation

`test_pipeline.py` provides synthetic longitudinal scenarios for controlled testing without real images. The project documentation reports a 6/6 pass rate across the defined scenarios for classification, trend detection, and severity direction.

## 7. Assessment 5 Coverage

| Requirement | Coverage |
|---|---|
| 1. Programming language and development environment | Python 3.9+, OpenCV, scikit-image, NumPy, FastAPI/Uvicorn, Pydantic, SQLite, Git/GitHub |
| 2. Coding standards followed | Naming, formatting, modularity, documentation, explicit interfaces, constants, errors, and testability |
| 3. Project structure and module organization | Current repository modules and responsibilities documented |
| 4. Version control setup | GitHub repository, `main` branch, branching strategy, commit practice, hygiene |
| 5. Initial module implementation | Core schema, computer vision, classification, severity, trajectory, suggestions, storage, API, pipeline, and test harness documented |

> **Note:** The assessment requirements are based on the uploaded QP5 sheet. Implementation details in this document are based on the current `sai-140307/ScarAnalyzer` repository.