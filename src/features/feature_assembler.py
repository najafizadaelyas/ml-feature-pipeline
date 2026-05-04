"""Joins all transformer outputs onto the entity spine via left joins."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame

from src.config.models import FeatureDefinition, PipelineConfig
from src.features.aggregation_transformer import AggregationTransformer
from src.features.encoding_transformer import EncodingTransformer
from src.features.null_handler import NullHandler
from src.features.time_window_transformer import TimeWindowTransformer

logger = logging.getLogger(__name__)


def assemble_features(
    df: DataFrame,
    config: PipelineConfig,
    definitions: list[FeatureDefinition],
) -> DataFrame:
    """Run NullHandler → Agg → Encoding → TimeWindow then join all onto entity spine.

    Parameters
    ----------
    df:
        Raw (validated) input DataFrame.
    config:
        Pipeline config (entity_key, date_column, run_date).
    definitions:
        List of parsed FeatureDefinition objects.

    Returns
    -------
    DataFrame
        Wide feature DataFrame with entity_key + all computed feature columns.
    """
    # Build entity spine (distinct entity keys)
    spine: DataFrame = df.select(config.entity_key).distinct()

    for defn in definitions:
        logger.info("Processing feature definition for entity_type=%s", defn.entity_type)

        # 1. Apply null handling to source data
        null_handler = NullHandler(defn.null_handlers)
        clean_df = null_handler.transform(df)

        # 2. Aggregation features
        if defn.aggregations:
            agg_df = AggregationTransformer(
                features=defn.aggregations,
                entity_key=config.entity_key,
                date_column=config.date_column,
                run_date=config.run_date,
            ).transform(clean_df)
            spine = spine.join(agg_df, on=config.entity_key, how="left")

        # 3. Encoding features
        if defn.encodings:
            enc_df = EncodingTransformer(
                features=defn.encodings,
                entity_key=config.entity_key,
            ).transform(clean_df)
            spine = spine.join(enc_df, on=config.entity_key, how="left")

        # 4. Time window features
        if defn.time_windows:
            tw_df = TimeWindowTransformer(
                features=defn.time_windows,
                entity_key=config.entity_key,
                date_column=config.date_column,
                run_date=config.run_date,
            ).transform(clean_df)
            spine = spine.join(tw_df, on=config.entity_key, how="left")

    logger.info(
        "Feature assembly complete: %d feature columns for %d entities",
        len(spine.columns) - 1,
        spine.count(),
    )
    return spine
