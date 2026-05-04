"""Unit tests for TimeWindowTransformer."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config.models import TimeWindowFeature
from src.features.time_window_transformer import TimeWindowTransformer


@pytest.fixture()
def sample_df(spark: SparkSession):
    data = [
        ("c1", "2024-01-10", 100.0),
        ("c1", "2024-01-15", 150.0),
        ("c1", "2024-01-20", 200.0),
        ("c2", "2024-01-05", 50.0),
        ("c2", "2024-01-25", 250.0),
    ]
    return spark.createDataFrame(data, ["customer_id", "event_date", "amount"]).withColumn(
        "event_date", F.col("event_date").cast("date")
    )


def test_days_diff(spark, sample_df):
    feats = [TimeWindowFeature(name="days_since_last", source_column="event_date", feature_type="days_diff")]
    result = TimeWindowTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(sample_df)

    rows = {r["customer_id"]: r["days_since_last"] for r in result.collect()}
    # c1 last event 2024-01-20 → 12 days; c2 last event 2024-01-25 → 7 days
    assert rows["c1"] == 12
    assert rows["c2"] == 7


def test_rolling_mean(spark, sample_df):
    feats = [
        TimeWindowFeature(name="rolling_mean_7d", source_column="amount", feature_type="rolling_mean", window_days=7)
    ]
    # run_date = 2024-02-01, cutoff = 2024-01-25
    # c1: only 2024-01-20 (200.0) is in window [2024-01-25, 2024-02-01]? No — 2024-01-20 < 2024-01-25
    # Actually cutoff = date_sub(2024-02-01, 7) = 2024-01-25
    # c2: 2024-01-25 (250.0) → mean = 250.0
    result = TimeWindowTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(sample_df)
    rows = {r["customer_id"]: r["rolling_mean_7d"] for r in result.collect()}
    assert rows["c2"] == pytest.approx(250.0)


def test_rolling_slope_positive_trend(spark: SparkSession):
    """Slope should be positive for monotonically increasing amounts."""
    data = [
        ("c1", "2024-01-01", 10.0),
        ("c1", "2024-01-11", 20.0),
        ("c1", "2024-01-21", 30.0),
    ]
    df = spark.createDataFrame(data, ["customer_id", "event_date", "amount"]).withColumn(
        "event_date", F.col("event_date").cast("date")
    )
    feats = [
        TimeWindowFeature(
            name="slope_30d", source_column="amount", feature_type="rolling_slope", window_days=30
        )
    ]
    result = TimeWindowTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(df)
    rows = {r["customer_id"]: r["slope_30d"] for r in result.collect()}
    assert rows["c1"] > 0


def test_rolling_slope_flat_returns_zero(spark: SparkSession):
    """Constant values → slope should be 0."""
    data = [("c1", "2024-01-01", 5.0), ("c1", "2024-01-15", 5.0), ("c1", "2024-01-29", 5.0)]
    df = spark.createDataFrame(data, ["customer_id", "event_date", "amount"]).withColumn(
        "event_date", F.col("event_date").cast("date")
    )
    feats = [
        TimeWindowFeature(
            name="slope_30d", source_column="amount", feature_type="rolling_slope", window_days=30
        )
    ]
    result = TimeWindowTransformer(feats, "customer_id", "event_date", "2024-02-01").transform(df)
    rows = {r["customer_id"]: r["slope_30d"] for r in result.collect()}
    assert rows["c1"] == pytest.approx(0.0, abs=1e-9)
