"""Single-pass null rate check across all columns."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.validation.models import ColumnValidationResult, ValidationResult


def check_null_rates(
    df: DataFrame,
    max_null_rate: float = 0.05,
    columns: list[str] | None = None,
) -> ValidationResult:
    """Check null rates across columns in a single aggregation pass.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    max_null_rate:
        Fraction above which a column fails the check (0–1).
    columns:
        Subset of columns to check; defaults to all columns.
    """
    cols_to_check = columns or df.columns
    total_rows = df.count()

    if total_rows == 0:
        return ValidationResult(
            passed=False,
            results=[
                ColumnValidationResult(
                    column="*",
                    check="null_rate",
                    passed=False,
                    detail="DataFrame is empty; cannot compute null rates.",
                )
            ],
        )

    # Single agg() call — one scan for all columns
    null_counts = df.agg(
        *[F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in cols_to_check]
    ).collect()[0]

    results: list[ColumnValidationResult] = []
    for col in cols_to_check:
        null_count = null_counts[col]
        rate = null_count / total_rows
        passed = rate <= max_null_rate
        results.append(
            ColumnValidationResult(
                column=col,
                check="null_rate",
                passed=passed,
                detail=(
                    ""
                    if passed
                    else (
                        f"Column '{col}' null rate {rate:.2%} exceeds threshold {max_null_rate:.2%}"
                    )
                ),
            )
        )

    return ValidationResult(
        passed=all(r.passed for r in results),
        results=results,
    )
