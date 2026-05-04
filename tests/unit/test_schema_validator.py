"""Unit tests for schema_validator."""

import pytest
from pyspark.sql import SparkSession

from src.validation.schema_validator import validate_schema


@pytest.fixture()
def sample_df(spark: SparkSession):
    data = [("c1", "2024-01-01", 100.0)]
    return spark.createDataFrame(data, ["customer_id", "event_date", "amount"])


def test_all_required_columns_present(spark, sample_df):
    result = validate_schema(sample_df, ["customer_id", "event_date", "amount"])
    assert result.passed is True
    assert result.failures() == []


def test_missing_column_fails(spark, sample_df):
    result = validate_schema(sample_df, ["customer_id", "missing_col"])
    assert result.passed is False
    failure_cols = [f.column for f in result.failures()]
    assert "missing_col" in failure_cols


def test_type_check_passes_for_correct_type(spark, sample_df):
    result = validate_schema(sample_df, ["customer_id"], {"amount": "double"})
    assert result.passed is True


def test_type_check_fails_for_wrong_type(spark, sample_df):
    result = validate_schema(sample_df, [], {"amount": "string"})
    assert result.passed is False


def test_type_alias_integer_accepts_long(spark: SparkSession):
    df = spark.createDataFrame([(1,)], ["count"]).selectExpr("cast(count as bigint) as count")
    result = validate_schema(df, [], {"count": "integer"})
    assert result.passed is True


def test_empty_required_columns(spark, sample_df):
    result = validate_schema(sample_df, [])
    assert result.passed is True
