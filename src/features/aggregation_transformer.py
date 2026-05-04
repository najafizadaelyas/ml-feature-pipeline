"""Computes aggregation features (sum/mean/count/stddev/min/max) with optional time windows."""

from __future__ import annotations

from datetime import date

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config.models import AggregationFeature
from src.features.base_transformer import BaseTransformer

_AGG_FUNCS = {
    "sum": F.sum,
    "mean": F.mean,
    "count": F.count,
    "stddev": F.stddev,
    "min": F.min,
    "max": F.max,
}


class AggregationTransformer(BaseTransformer):
    """Group by entity key and aggregate source columns.

    If window_days is set, only rows within that many days before run_date
    are included in the aggregation.
    """

    def __init__(
        self,
        features: list[AggregationFeature],
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
            agg_fn = _AGG_FUNCS[feat.function]
            subset = df

            if feat.window_days is not None:
                cutoff = F.date_sub(F.lit(self.run_date).cast("date"), feat.window_days)
                subset = df.filter(F.col(self.date_column) >= cutoff)

            agg_df = subset.groupBy(self.entity_key).agg(
                agg_fn(feat.source_column).alias(feat.name)
            )

            result = result.join(agg_df, on=self.entity_key, how="left")

        return result
