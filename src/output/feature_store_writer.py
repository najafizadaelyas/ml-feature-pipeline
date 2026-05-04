"""Writes assembled features as partitioned Parquet to S3 feature store."""

from __future__ import annotations

import logging
from datetime import date

from pyspark.sql import DataFrame

from src.config.models import PipelineConfig

logger = logging.getLogger(__name__)


def write_feature_store(
    df: DataFrame,
    config: PipelineConfig,
) -> str:
    """Write the feature DataFrame to S3 partitioned by entity_type and date.

    Partition path: s3a://<bucket>/<prefix>/entity_type=<X>/date=<Y>/

    Returns
    -------
    str
        The full S3 output path written.
    """
    run_date = config.run_date or str(date.today())
    output_path = (
        f"s3a://{config.s3.feature_store_bucket}/"
        f"{config.s3.feature_store_prefix.rstrip('/')}/"
        f"entity_type={config.entity_type}/"
        f"date={run_date}"
    )

    # Add partition columns as literals so they're stored in directory structure
    df_with_partitions = df.withColumn(
        "entity_type", _lit(config.entity_type)
    ).withColumn("date", _lit(run_date))

    (
        df_with_partitions.write.mode("overwrite")
        .partitionBy("entity_type", "date")
        .parquet(output_path)
    )

    logger.info("Feature store written to %s (%d rows)", output_path, df.count())
    return output_path


def _lit(value: str):
    from pyspark.sql import functions as F

    return F.lit(value)
