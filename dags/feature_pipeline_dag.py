"""Airflow 2.9 DAG — ML Feature Engineering Pipeline (TaskFlow API)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="feature_pipeline",
    description="Daily feature engineering pipeline: ingest → validate → compute → store",
    schedule="0 2 * * *",  # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "feature-engineering"],
    max_active_runs=1,
)
def feature_pipeline():
    # ── Task 1: Load config ──────────────────────────────────────────────────
    @task()
    def load_config(run_date: str | None = None, **context) -> dict:
        """Parse and validate pipeline_config.yaml; return serialisable metadata."""
        from dags.utils.dag_helpers import get_config_dir, get_config_path
        from src.config.loader import load_feature_definitions, load_pipeline_config

        ds = run_date or context["ds"]  # Airflow logical date (YYYY-MM-DD)
        cfg = load_pipeline_config(get_config_path(), run_date=ds)
        defs = load_feature_definitions(cfg, base_dir=get_config_dir())

        logger.info(
            "Config loaded: pipeline=%s entity=%s run_date=%s definitions=%d",
            cfg.pipeline_name,
            cfg.entity_type,
            cfg.run_date,
            len(defs),
        )

        # Return small metadata dict — DataFrames must NOT travel via XCom
        return {
            "pipeline_name": cfg.pipeline_name,
            "entity_type": cfg.entity_type,
            "entity_key": cfg.entity_key,
            "date_column": cfg.date_column,
            "run_date": cfg.run_date,
            "definition_count": len(defs),
        }

    # ── Task 2: Ingest raw data ──────────────────────────────────────────────
    @task()
    def ingest_raw(config_meta: dict) -> dict:
        """Read raw data from S3; return row count metadata."""
        from dags.utils.dag_helpers import get_config_path
        from src.config.loader import load_pipeline_config
        from src.ingestion.s3_reader import read_raw
        from src.spark.session import create_spark_session, stop_spark_session

        cfg = load_pipeline_config(get_config_path(), run_date=config_meta["run_date"])
        spark = create_spark_session(cfg.spark, endpoint_url=cfg.s3.endpoint_url)
        try:
            df = read_raw(spark, cfg)
            row_count = df.count()
            logger.info("Ingested %d rows for run_date=%s", row_count, cfg.run_date)
            return {**config_meta, "row_count": row_count}
        finally:
            stop_spark_session(spark)

    # ── Task 3: Validate data quality ────────────────────────────────────────
    @task()
    def validate(ingest_meta: dict) -> dict:
        """Run schema + null-rate quality gate."""
        from dags.utils.dag_helpers import get_config_path
        from src.config.loader import load_pipeline_config
        from src.ingestion.s3_reader import read_raw
        from src.spark.session import create_spark_session, stop_spark_session
        from src.validation.quality_gate import run_quality_gate

        cfg = load_pipeline_config(get_config_path(), run_date=ingest_meta["run_date"])
        spark = create_spark_session(cfg.spark, endpoint_url=cfg.s3.endpoint_url)
        try:
            df = read_raw(spark, cfg)
            result = run_quality_gate(df, cfg)
            logger.info("Validation: %s", result.summary())
            return {**ingest_meta, "validation_passed": result.passed}
        finally:
            stop_spark_session(spark)

    # ── Task 4: Compute features + write store + update registry ─────────────
    @task()
    def compute_features(validated_meta: dict) -> dict:
        """Assemble features, write to feature store, update registry."""
        from dags.utils.dag_helpers import get_config_dir, get_config_path
        from src.config.loader import load_feature_definitions, load_pipeline_config
        from src.features.feature_assembler import assemble_features
        from src.ingestion.s3_reader import read_raw
        from src.output.feature_store_writer import write_feature_store
        from src.output.registry_updater import update_registry
        from src.spark.session import create_spark_session, stop_spark_session

        cfg = load_pipeline_config(get_config_path(), run_date=validated_meta["run_date"])
        defs = load_feature_definitions(cfg, base_dir=get_config_dir())
        spark = create_spark_session(cfg.spark, endpoint_url=cfg.s3.endpoint_url)

        try:
            df = read_raw(spark, cfg)
            feature_df = assemble_features(df, cfg, defs)
            output_path = write_feature_store(feature_df, cfg)
            feature_cols = [c for c in feature_df.columns if c != cfg.entity_key]
            update_registry(cfg, defs, feature_cols, output_path)

            logger.info(
                "Pipeline complete: %d features written to %s",
                len(feature_cols),
                output_path,
            )
            return {
                **validated_meta,
                "output_path": output_path,
                "feature_count": len(feature_cols),
            }
        finally:
            stop_spark_session(spark)

    # ── Wire tasks ───────────────────────────────────────────────────────────
    cfg_meta = load_config()
    ingest_meta = ingest_raw(cfg_meta)
    validated_meta = validate(ingest_meta)
    compute_features(validated_meta)


feature_pipeline_dag = feature_pipeline()
