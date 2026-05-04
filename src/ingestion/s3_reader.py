"""Read raw CSV or Parquet data from S3 into a Spark DataFrame."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from src.config.models import PipelineConfig


def build_s3_path(bucket: str, prefix: str, run_date: str | None = None) -> str:
    base = f"s3a://{bucket}/{prefix.rstrip('/')}"
    if run_date:
        return f"{base}/date={run_date}"
    return base


def read_raw(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Read raw data from S3 and validate it is non-empty.

    Returns
    -------
    DataFrame
        The raw input DataFrame.

    Raises
    ------
    ValueError
        If the resulting DataFrame has zero rows.
    """
    path = build_s3_path(
        config.s3.raw_bucket,
        config.s3.raw_prefix,
        config.run_date,
    )

    if config.input_format == "csv":
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    else:
        df = spark.read.parquet(path)

    row_count = df.count()
    if row_count < config.validation.min_row_count:
        raise ValueError(
            f"Ingested {row_count} rows from {path!r}; "
            f"minimum required is {config.validation.min_row_count}."
        )

    return df
