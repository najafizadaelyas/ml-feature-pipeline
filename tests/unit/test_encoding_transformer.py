"""Unit tests for EncodingTransformer."""

import pytest
from pyspark.sql import SparkSession

from src.config.models import EncodingFeature
from src.features.encoding_transformer import EncodingTransformer


@pytest.fixture()
def sample_df(spark: SparkSession):
    data = [
        ("c1", "premium"),
        ("c2", "basic"),
        ("c3", "premium"),
        ("c4", "basic"),
        ("c5", "enterprise"),
    ]
    return spark.createDataFrame(data, ["customer_id", "plan_type"])


def test_frequency_encoding_sums_to_one(spark, sample_df):
    feats = [EncodingFeature(name="plan_freq", source_column="plan_type", method="frequency")]
    result = EncodingTransformer(feats, "customer_id").transform(sample_df)

    # Each entity maps to its category's frequency; distinct category frequencies sum to 1.0
    joined = result.join(sample_df.select("customer_id", "plan_type"), on="customer_id")
    category_total = (
        joined.select("plan_type", "plan_freq").distinct()
        .selectExpr("sum(plan_freq)").collect()[0][0]
    )
    assert category_total == pytest.approx(1.0, rel=1e-4)


def test_frequency_encoding_higher_for_common_value(spark, sample_df):
    feats = [EncodingFeature(name="plan_freq", source_column="plan_type", method="frequency")]
    result = EncodingTransformer(feats, "customer_id").transform(sample_df)

    rows = {r["customer_id"]: r["plan_freq"] for r in result.collect()}
    # "premium" appears 2/5, "enterprise" 1/5
    assert rows["c1"] > rows["c5"]


def test_label_encoding_returns_numeric(spark, sample_df):
    feats = [EncodingFeature(name="plan_label", source_column="plan_type", method="label")]
    result = EncodingTransformer(feats, "customer_id").transform(sample_df)

    col_type = dict(result.dtypes)["plan_label"]
    assert "double" in col_type or "float" in col_type


def test_label_encoding_distinct_values(spark, sample_df):
    feats = [EncodingFeature(name="plan_label", source_column="plan_type", method="label")]
    result = EncodingTransformer(feats, "customer_id").transform(sample_df)
    distinct_values = result.select("plan_label").distinct().count()
    assert distinct_values == 3  # premium, basic, enterprise


def test_onehot_encoding_produces_vector(spark, sample_df):
    feats = [EncodingFeature(name="plan_ohe", source_column="plan_type", method="onehot")]
    result = EncodingTransformer(feats, "customer_id").transform(sample_df)

    col_type = dict(result.dtypes)["plan_ohe"]
    assert "vector" in col_type.lower()


def test_missing_source_column_skipped(spark: SparkSession):
    df = spark.createDataFrame([("c1",)], ["customer_id"])
    feats = [EncodingFeature(name="x", source_column="nonexistent", method="frequency")]
    result = EncodingTransformer(feats, "customer_id").transform(df)
    # Only entity key should remain
    assert result.columns == ["customer_id"]
