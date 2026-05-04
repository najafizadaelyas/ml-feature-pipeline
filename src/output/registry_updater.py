"""Updates the feature registry JSON in S3 via boto3."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import boto3

from src.config.models import FeatureDefinition, PipelineConfig

logger = logging.getLogger(__name__)


def _build_registry_entry(
    config: PipelineConfig,
    definitions: list[FeatureDefinition],
    feature_columns: list[str],
    output_path: str,
) -> dict:
    all_features: list[dict] = []

    for defn in definitions:
        for feat in defn.aggregations:
            all_features.append({"name": feat.name, "type": "aggregation"})
        for feat in defn.encodings:
            all_features.append({"name": feat.name, "type": "encoding"})
        for feat in defn.time_windows:
            all_features.append({"name": feat.name, "type": "time_window"})

    return {
        "pipeline_name": config.pipeline_name,
        "entity_type": config.entity_type,
        "entity_key": config.entity_key,
        "run_date": config.run_date or str(date.today()),
        "output_path": output_path,
        "feature_count": len(feature_columns),
        "features": all_features,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def update_registry(
    config: PipelineConfig,
    definitions: list[FeatureDefinition],
    feature_columns: list[str],
    output_path: str,
) -> None:
    """Read current registry from S3, upsert the entry for this pipeline, write back."""
    s3_kwargs: dict = {}
    if config.s3.endpoint_url:
        s3_kwargs["endpoint_url"] = config.s3.endpoint_url

    s3 = boto3.client("s3", **s3_kwargs)
    bucket = config.s3.registry_bucket
    key = config.s3.registry_key

    # Load existing registry or start fresh
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        registry: dict = json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        registry = {"pipelines": {}}
    except Exception:
        registry = {"pipelines": {}}

    registry.setdefault("pipelines", {})
    registry["pipelines"][config.pipeline_name] = _build_registry_entry(
        config, definitions, feature_columns, output_path
    )

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(registry, indent=2).encode(),
        ContentType="application/json",
    )
    logger.info("Feature registry updated at s3://%s/%s", bucket, key)
