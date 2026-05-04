"""Unit tests for quality_gate."""

import pytest
from pyspark.sql import SparkSession

from src.config.models import PipelineConfig, S3Config, ValidationConfig
from src.validation.quality_gate import DataQualityError, run_quality_gate


@pytest.fixture()
def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        pipeline_name="test_pipeline",
        entity_type="customer",
        entity_key="customer_id",
        date_column="event_date",
        s3=S3Config(
            raw_bucket="b",
            raw_prefix="p/",
            feature_store_bucket="fb",
            feature_store_prefix="fp/",
            registry_bucket="rb",
        ),
        validation=ValidationConfig(max_null_rate=0.1, min_row_count=1),
    )


@pytest.fixture()
def clean_df(spark: SparkSession):
    data = [("c1", "2024-01-01"), ("c2", "2024-01-02"), ("c3", "2024-01-03")]
    return spark.createDataFrame(data, ["customer_id", "event_date"])


@pytest.fixture()
def dirty_df(spark: SparkSession):
    # >10% nulls in event_date
    data = [("c1", "2024-01-01"), ("c2", None), ("c3", None), ("c4", None)]
    return spark.createDataFrame(data, ["customer_id", "event_date"])


def test_passes_on_clean_data(spark, clean_df, pipeline_config):
    result = run_quality_gate(clean_df, pipeline_config)
    assert result.passed is True


def test_raises_on_missing_required_column(spark: SparkSession, pipeline_config):
    df = spark.createDataFrame([("c1",)], ["customer_id"])  # missing event_date
    with pytest.raises(DataQualityError, match="event_date"):
        run_quality_gate(df, pipeline_config)


def test_raises_on_high_null_rate(spark, dirty_df, pipeline_config):
    with pytest.raises(DataQualityError, match="null rate"):
        run_quality_gate(dirty_df, pipeline_config)


def test_custom_required_columns_checked(spark: SparkSession, pipeline_config):
    df = spark.createDataFrame([("c1", "2024-01-01")], ["customer_id", "event_date"])
    with pytest.raises(DataQualityError, match="extra_col"):
        run_quality_gate(df, pipeline_config, required_columns=["extra_col"])
