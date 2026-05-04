"""Unit tests for AggregationTransformer."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config.models import AggregationFeature
from src.features.aggregation_transformer import AggregationTransformer


@pytest.fixture()
def sample_df(spark: SparkSession):
    data = [
        ("c1", "2024-01-10", 100.0, "s1"),
        ("c1", "2024-01-15", 200.0, "s2"),
        ("c1", "2023-11-01", 50.0, "s3"),  # >90 days before run_date
        ("c2", "2024-01-20", 300.0, "s4"),
        ("c2", "2024-01-25", 400.0, "s5"),
    ]
    return spark.createDataFrame(data, ["customer_id", "event_date", "amount", "session_id"]).withColumn(
        "event_date", F.col("event_date").cast("date")
    )


def test_all_time_count(spark, sample_df):
    feats = [AggregationFeature(name="total_sessions", source_column="session_id", function="count")]
    result = AggregationTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(sample_df)

    rows = {r["customer_id"]: r["total_sessions"] for r in result.collect()}
    assert rows["c1"] == 3
    assert rows["c2"] == 2


def test_window_sum(spark, sample_df):
    feats = [
        AggregationFeature(name="spend_30d", source_column="amount", function="sum", window_days=30)
    ]
    # run_date=2024-02-01, 30-day window → cutoff 2024-01-02
    result = AggregationTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(sample_df)

    rows = {r["customer_id"]: r["spend_30d"] for r in result.collect()}
    # c1: 100 + 200 = 300 (2023-11-01 excluded)
    assert rows["c1"] == pytest.approx(300.0)
    # c2: 300 + 400 = 700
    assert rows["c2"] == pytest.approx(700.0)


def test_mean_feature(spark, sample_df):
    feats = [AggregationFeature(name="avg_amount", source_column="amount", function="mean")]
    result = AggregationTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(sample_df)

    rows = {r["customer_id"]: r["avg_amount"] for r in result.collect()}
    assert rows["c1"] == pytest.approx((100 + 200 + 50) / 3, rel=1e-4)


def test_output_contains_only_entity_and_feature_cols(spark, sample_df):
    feats = [AggregationFeature(name="cnt", source_column="session_id", function="count")]
    result = AggregationTransformer(feats, "customer_id", "event_date").transform(sample_df)
    assert set(result.columns) == {"customer_id", "cnt"}


def test_entity_spine_preserved_when_no_matching_rows(spark: SparkSession):
    """Entity with zero rows in window should appear with null feature (left join)."""
    data = [("c1", "2020-01-01", 10.0)]
    df = spark.createDataFrame(data, ["customer_id", "event_date", "amount"]).withColumn(
        "event_date", F.col("event_date").cast("date")
    )
    feats = [
        AggregationFeature(name="spend_30d", source_column="amount", function="sum", window_days=30)
    ]
    result = AggregationTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(df)
    rows = {r["customer_id"]: r["spend_30d"] for r in result.collect()}
    assert rows["c1"] is None
