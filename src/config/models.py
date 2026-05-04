"""Pydantic v2 models for pipeline configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class S3Config(BaseModel):
    raw_bucket: str
    raw_prefix: str
    feature_store_bucket: str
    feature_store_prefix: str
    registry_bucket: str
    registry_key: str = "feature_registry.json"
    endpoint_url: str | None = None  # LocalStack override


class SparkConfig(BaseModel):
    app_name: str = "ml-feature-pipeline"
    master: str = "local[*]"
    executor_memory: str = "2g"
    driver_memory: str = "2g"
    shuffle_partitions: int = 50
    extra_conf: dict[str, str] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    max_null_rate: float = 0.05
    min_row_count: int = 1


class PipelineConfig(BaseModel):
    pipeline_name: str
    entity_type: str
    entity_key: str
    date_column: str
    input_format: Literal["csv", "parquet"] = "parquet"
    feature_definition_files: list[str] = Field(default_factory=list)
    s3: S3Config
    spark: SparkConfig = Field(default_factory=SparkConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    run_date: str | None = None  # injected at runtime by Airflow

    @model_validator(mode="after")
    def _check_entity_key_in_date_column(self) -> "PipelineConfig":
        if self.entity_key == self.date_column:
            raise ValueError("entity_key and date_column must be different columns")
        return self


# ── Feature definition models ───────────────────────────────────────────────

class AggregationFeature(BaseModel):
    name: str
    source_column: str
    function: Literal["sum", "mean", "count", "stddev", "min", "max"]
    window_days: int | None = None  # None → all-time aggregate


class EncodingFeature(BaseModel):
    name: str
    source_column: str
    method: Literal["frequency", "label", "onehot"]


class TimeWindowFeature(BaseModel):
    name: str
    source_column: str
    feature_type: Literal["days_diff", "rolling_mean", "rolling_slope"]
    window_days: int | None = None  # required for rolling features


class NullHandlerSpec(BaseModel):
    column: str
    strategy: Literal["fill", "median_impute", "drop"]
    fill_value: str | float | int | None = None  # used when strategy == "fill"
    add_indicator: bool = False


class FeatureDefinition(BaseModel):
    entity_type: str
    aggregations: list[AggregationFeature] = Field(default_factory=list)
    encodings: list[EncodingFeature] = Field(default_factory=list)
    time_windows: list[TimeWindowFeature] = Field(default_factory=list)
    null_handlers: list[NullHandlerSpec] = Field(default_factory=list)
