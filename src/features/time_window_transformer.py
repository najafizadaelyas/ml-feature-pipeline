"""Time-based feature transformers: days_diff, rolling_mean, rolling_slope (OLS via Spark)."""

from __future__ import annotations

from datetime import date

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config.models import TimeWindowFeature
from src.features.base_transformer import BaseTransformer


class TimeWindowTransformer(BaseTransformer):
    """Compute time-window features using pure Spark Window functions (no UDFs).

    Features
    --------
    days_diff    : days between most-recent event date and run_date
    rolling_mean : mean of source_column over last window_days days
    rolling_slope: OLS slope via column arithmetic over a row-based window
                   (avoids applyInPandas; uses Σxy/Σx² with centred indices)
    """

    def __init__(
        self,
        features: list[TimeWindowFeature],
        entity_key: str,
        date_column: str,
        run_date: str | None = None,
    ) -> None:
        self.features = features
        self.entity_key = entity_key
        self.date_column = date_column
        self.run_date = run_date or str(date.today())

    def transform(self, df: DataFrame) -> DataFrame:
        result = df.select(self.entity_key).distinct()

        for feat in self.features:
            col = feat.source_column
            if col not in df.columns and feat.feature_type != "days_diff":
                continue

            if feat.feature_type == "days_diff":
                latest = (
                    df.groupBy(self.entity_key)
                    .agg(F.max(self.date_column).alias("__max_date__"))
                    .withColumn(
                        feat.name,
                        F.datediff(F.lit(self.run_date).cast("date"), F.col("__max_date__")),
                    )
                    .select(self.entity_key, feat.name)
                )
                result = result.join(latest, on=self.entity_key, how="left")

            elif feat.feature_type == "rolling_mean":
                window_days = feat.window_days or 7
                cutoff = F.date_sub(F.lit(self.run_date).cast("date"), window_days)
                agg_df = (
                    df.filter(F.col(self.date_column) >= cutoff)
                    .groupBy(self.entity_key)
                    .agg(F.mean(col).alias(feat.name))
                )
                result = result.join(agg_df, on=self.entity_key, how="left")

            elif feat.feature_type == "rolling_slope":
                window_days = feat.window_days or 30
                cutoff = F.date_sub(F.lit(self.run_date).cast("date"), window_days)
                subset = df.filter(F.col(self.date_column) >= cutoff)

                # OLS slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
                # x = days since cutoff (integer), y = source_column value
                slope_df = (
                    subset.withColumn(
                        "__x__",
                        F.datediff(F.col(self.date_column), cutoff).cast("double"),
                    )
                    .withColumn("__y__", F.col(col).cast("double"))
                    .groupBy(self.entity_key)
                    .agg(
                        F.count("*").alias("__n__"),
                        F.sum(F.col("__x__") * F.col("__y__")).alias("__sum_xy__"),
                        F.sum("__x__").alias("__sum_x__"),
                        F.sum("__y__").alias("__sum_y__"),
                        F.sum(F.col("__x__") * F.col("__x__")).alias("__sum_x2__"),
                    )
                    .withColumn(
                        "__denom__",
                        F.col("__n__") * F.col("__sum_x2__")
                        - F.col("__sum_x__") * F.col("__sum_x__"),
                    )
                    .withColumn(
                        feat.name,
                        F.when(
                            F.col("__denom__") != 0,
                            (
                                F.col("__n__") * F.col("__sum_xy__")
                                - F.col("__sum_x__") * F.col("__sum_y__")
                            )
                            / F.col("__denom__"),
                        ).otherwise(F.lit(0.0)),
                    )
                    .select(self.entity_key, feat.name)
                )
                result = result.join(slope_df, on=self.entity_key, how="left")

        return result
