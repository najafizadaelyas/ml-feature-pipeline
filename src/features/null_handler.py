"""Handles null values via fill, median imputation, or row drop."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.config.models import NullHandlerSpec
from src.features.base_transformer import BaseTransformer


class NullHandler(BaseTransformer):
    """Apply null handling strategies declared in the feature definition.

    Strategies
    ----------
    fill         : replace nulls with a constant fill_value
    median_impute: replace nulls with the column's approximate median
    drop         : drop rows where the column is null
    """

    def __init__(self, specs: list[NullHandlerSpec]) -> None:
        self.specs = specs

    def transform(self, df: DataFrame) -> DataFrame:
        for spec in self.specs:
            col = spec.column
            if col not in df.columns:
                continue

            if spec.add_indicator:
                indicator_col = f"{col}_was_null"
                df = df.withColumn(indicator_col, F.col(col).isNull().cast("integer"))

            if spec.strategy == "fill":
                df = df.fillna({col: spec.fill_value})

            elif spec.strategy == "median_impute":
                median_val = df.approxQuantile(col, [0.5], 0.01)[0]
                df = df.fillna({col: median_val})

            elif spec.strategy == "drop":
                df = df.filter(F.col(col).isNotNull())

        return df
