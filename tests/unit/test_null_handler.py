"""Unit tests for NullHandler."""

import pytest
from pyspark.sql import SparkSession

from src.config.models import NullHandlerSpec
from src.features.null_handler import NullHandler


@pytest.fixture()
def df_with_nulls(spark: SparkSession):
    data = [("c1", 100.0, "basic"), ("c2", None, "premium"), ("c3", 50.0, None), ("c4", None, None)]
    return spark.createDataFrame(data, ["customer_id", "amount", "plan"])


def test_fill_strategy_replaces_nulls(spark, df_with_nulls):
    specs = [NullHandlerSpec(column="plan", strategy="fill", fill_value="unknown")]
    result = NullHandler(specs).transform(df_with_nulls)
    null_count = result.filter(result.plan.isNull()).count()
    assert null_count == 0
    values = {r["customer_id"]: r["plan"] for r in result.collect()}
    assert values["c3"] == "unknown"
    assert values["c4"] == "unknown"


def test_median_impute_replaces_nulls(spark, df_with_nulls):
    specs = [NullHandlerSpec(column="amount", strategy="median_impute")]
    result = NullHandler(specs).transform(df_with_nulls)
    null_count = result.filter(result.amount.isNull()).count()
    assert null_count == 0


def test_drop_strategy_removes_rows(spark, df_with_nulls):
    specs = [NullHandlerSpec(column="amount", strategy="drop")]
    result = NullHandler(specs).transform(df_with_nulls)
    assert result.count() == 2  # c1 and c3


def test_add_indicator_column(spark, df_with_nulls):
    specs = [NullHandlerSpec(column="amount", strategy="median_impute", add_indicator=True)]
    result = NullHandler(specs).transform(df_with_nulls)
    assert "amount_was_null" in result.columns
    indicators = {r["customer_id"]: r["amount_was_null"] for r in result.collect()}
    assert indicators["c2"] == 1
    assert indicators["c4"] == 1
    assert indicators["c1"] == 0


def test_missing_column_is_skipped(spark: SparkSession):
    df = spark.createDataFrame([("c1", 1.0)], ["customer_id", "x"])
    specs = [NullHandlerSpec(column="nonexistent", strategy="fill", fill_value=0)]
    result = NullHandler(specs).transform(df)
    assert result.columns == ["customer_id", "x"]
