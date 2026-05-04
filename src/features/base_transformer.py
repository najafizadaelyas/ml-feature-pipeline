"""Abstract base class for all feature transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class BaseTransformer(ABC):
    """All feature transformers inherit from this class."""

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply the transformation and return a new DataFrame.

        The returned DataFrame must contain at least the entity key column
        plus the newly derived feature columns.  It is the responsibility of
        FeatureAssembler to join transformer outputs back to the entity spine.
        """
