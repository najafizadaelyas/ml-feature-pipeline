"""SparkSession factory with S3A connector configuration."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from src.config.models import SparkConfig


def create_spark_session(config: SparkConfig, endpoint_url: str | None = None) -> SparkSession:
    """Build and return a configured SparkSession.

    S3A credentials are sourced from environment variables:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    """
    builder = (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.executor.memory", config.executor_memory)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        # S3A connector
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
        )
        .config(
            "spark.hadoop.fs.s3a.access.key",
            os.environ.get("AWS_ACCESS_KEY_ID", ""),
        )
        .config(
            "spark.hadoop.fs.s3a.secret.key",
            os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        .config(
            "spark.hadoop.fs.s3a.endpoint.region",
            os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    )

    if endpoint_url:
        # LocalStack or custom S3-compatible endpoint
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", endpoint_url)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        )

    for key, value in config.extra_conf.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def stop_spark_session(spark: SparkSession) -> None:
    spark.stop()
