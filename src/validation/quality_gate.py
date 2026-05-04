"""Aggregates schema and null-rate validation results; raises on failure."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame

from src.config.models import PipelineConfig
from src.validation.models import ValidationResult
from src.validation.null_checker import check_null_rates
from src.validation.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def run_quality_gate(
    df: DataFrame,
    config: PipelineConfig,
    required_columns: list[str] | None = None,
    expected_types: dict[str, str] | None = None,
) -> ValidationResult:
    """Run schema validation and null rate checks; raise DataQualityError on failure.

    Parameters
    ----------
    df:
        Input DataFrame to validate.
    config:
        Pipeline config (provides entity_key, date_column, validation thresholds).
    required_columns:
        Extra columns beyond entity_key and date_column that must exist.
    expected_types:
        Optional column → type alias mapping for type checks.

    Raises
    ------
    DataQualityError
        If any check fails.
    """
    all_required = list({config.entity_key, config.date_column} | set(required_columns or []))

    schema_result = validate_schema(df, all_required, expected_types)
    null_result = check_null_rates(df, config.validation.max_null_rate)

    combined_results = schema_result.results + null_result.results
    overall_passed = schema_result.passed and null_result.passed

    result = ValidationResult(passed=overall_passed, results=combined_results)

    if not overall_passed:
        failure_details = "\n  ".join(f.detail for f in result.failures())
        raise DataQualityError(
            f"Data quality gate FAILED for pipeline '{config.pipeline_name}':\n  {failure_details}"
        )

    logger.info("Data quality gate PASSED: %s", result.summary())
    return result


class DataQualityError(Exception):
    """Raised when data quality checks do not pass."""
