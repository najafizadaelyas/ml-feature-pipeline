# ML Feature Pipeline

Production-grade, declarative ML feature engineering pipeline.

**Stack:** PySpark 3.5 · Airflow 2.9 (TaskFlow API) · AWS S3 (S3A) · Pydantic v2 · LocalStack

---

## Overview

New features are added via YAML only — no code changes required.
The pipeline runs daily at 02:00 UTC, reads raw events from S3, validates quality,
engineers features, and writes a partitioned Parquet feature store back to S3.

```
S3 Raw Zone
    │
    ▼
[Airflow DAG: feature_pipeline]
    ├─ load_config      → parse & validate pipeline_config.yaml
    ├─ ingest_raw       → read S3 raw data, assert row count
    ├─ validate         → schema check + null-rate gate (single agg scan)
    └─ compute_features → NullHandler → Agg → Encoding → TimeWindow → Write → Registry
```

---

## Quick Start

### Local development with LocalStack

```bash
# 1. Copy env and configure
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Create LocalStack S3 buckets
awslocal s3 mb s3://raw-data
awslocal s3 mb s3://feature-store

# 4. Upload sample data
awslocal s3 cp tests/fixtures/sample_events.parquet s3://raw-data/events/date=2024-01-01/

# 5. Open Airflow UI and trigger the DAG
open http://localhost:8080   # admin / admin
```

### Run tests locally

```bash
pip install -r requirements-dev.txt

# Unit tests (requires local JVM)
pytest tests/unit/ -v

# DAG integrity tests
AIRFLOW__CORE__LOAD_EXAMPLES=false airflow db migrate
pytest tests/dag_tests/ -v

# Lint
ruff check src/ dags/ tests/
```

---

## Adding New Features

1. Open (or create) a YAML file in `config/feature_definitions/`.
2. Add entries under `aggregations`, `encodings`, `time_windows`, or `null_handlers`.
3. Reference the file in `config/pipeline_config.yaml` under `feature_definition_files`.
4. Re-trigger the DAG — no code changes needed.

### Example: add a new aggregation

```yaml
# config/feature_definitions/customer_features.yaml
aggregations:
  - name: total_logins_7d
    source_column: login_id
    function: count
    window_days: 7
```

---

## Project Structure

```
ml-feature-pipeline/
├── dags/                    # Airflow DAG (TaskFlow API)
├── src/
│   ├── config/              # Pydantic models + YAML loader
│   ├── ingestion/           # S3 reader
│   ├── validation/          # Schema + null-rate quality gate
│   ├── features/            # Transformers (agg, encoding, time-window, null)
│   ├── output/              # Feature store writer + registry updater
│   └── spark/               # SparkSession factory
├── config/
│   ├── pipeline_config.yaml
│   └── feature_definitions/ # Declarative feature recipes
├── tests/
│   ├── unit/                # PySpark unit tests
│   └── dag_tests/           # Airflow DAG integrity tests
├── docker/airflow/          # Dockerfile (Airflow 2.9 + Java 17 + PySpark)
├── docker-compose.yml       # Airflow + Postgres + LocalStack
└── .github/workflows/ci.yml # lint → unit → dag-integrity → docker build
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Two-level YAML | Master config + per-entity recipes; add features without code |
| No Python UDFs | All logic uses native `F.*` / Spark ML for performance |
| Single `agg()` null check | One DataFrame scan for all columns |
| OLS slope via column arithmetic | Trend feature without `applyInPandas`; pure Spark Window |
| Left joins on entity spine | Never inflates row count |
| `write.mode("overwrite")` | Idempotent daily runs |
| LocalStack in docker-compose | Full S3 emulation; zero AWS cost for dev |
| Tasks share only XCom metadata | DataFrames never cross task boundaries (LocalExecutor-safe) |

---

## CI/CD

GitHub Actions pipeline: **lint → unit tests → DAG integrity → docker build**

See `.github/workflows/ci.yml`.
