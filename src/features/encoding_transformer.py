"""Frequency, label, and one-hot encoding transformers."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import DataFrame

from src.config.models import EncodingFeature
from src.features.base_transformer import BaseTransformer


class EncodingTransformer(BaseTransformer):
    """Apply frequency, label, or one-hot encoding per feature spec.

    All encodings are computed on the full DataFrame passed to transform().
    The encoder produces entity-level encoded columns by joining back on
    entity_key after computing encoding maps.
    """

    def __init__(self, features: list[EncodingFeature], entity_key: str) -> None:
        self.features = features
        self.entity_key = entity_key

    def transform(self, df: DataFrame) -> DataFrame:
        result = df.select(self.entity_key).distinct()

        for feat in self.features:
            col = feat.source_column
            if col not in df.columns:
                continue

            if feat.method == "frequency":
                freq_map = (
                    df.groupBy(col)
                    .agg(F.count("*").alias("__freq__"))
                    .withColumn(
                        feat.name,
                        F.col("__freq__") / F.sum("__freq__").over(
                            _unbounded_window()
                        ),
                    )
                    .select(col, feat.name)
                )
                encoded = df.select(self.entity_key, col).join(freq_map, on=col, how="left")
                encoded = encoded.select(self.entity_key, feat.name)

            elif feat.method == "label":
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=feat.name,
                    handleInvalid="keep",
                )
                model = indexer.fit(df)
                encoded = (
                    model.transform(df.select(self.entity_key, col))
                    .select(self.entity_key, feat.name)
                )

            elif feat.method == "onehot":
                idx_col = f"__idx_{col}__"
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=idx_col,
                    handleInvalid="keep",
                )
                idx_model = indexer.fit(df)
                indexed = idx_model.transform(df.select(self.entity_key, col))
                encoder = OneHotEncoder(inputCol=idx_col, outputCol=feat.name, dropLast=False)
                enc_model = encoder.fit(indexed)
                encoded = enc_model.transform(indexed).select(self.entity_key, feat.name)

            else:
                continue

            result = result.join(encoded, on=self.entity_key, how="left")

        return result


def _unbounded_window():
    from pyspark.sql import Window

    return Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
