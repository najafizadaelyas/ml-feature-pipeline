"""Validates that required columns are present and have expected types."""

from __future__ import annotations

from pyspark.sql import DataFrame

from src.validation.models import ColumnValidationResult, ValidationResult

# Mapping from user-friendly type aliases to Spark type strings (prefix match)
_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "string": ("StringType",),
    "integer": ("IntegerType", "LongType", "ShortType", "ByteType"),
    "long": ("LongType",),
    "double": ("DoubleType", "FloatType"),
    "float": ("FloatType", "DoubleType"),
    "date": ("DateType",),
    "timestamp": ("TimestampType",),
    "boolean": ("BooleanType",),
}


def validate_schema(
    df: DataFrame,
    required_columns: list[str],
    expected_types: dict[str, str] | None = None,
) -> ValidationResult:
    """Check column presence and optionally column types.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required_columns:
        Columns that must exist.
    expected_types:
        Optional mapping of column → expected type alias.
    """
    results: list[ColumnValidationResult] = []
    actual_fields = {f.name: type(f.dataType).__name__ for f in df.schema.fields}

    for col in required_columns:
        present = col in actual_fields
        results.append(
            ColumnValidationResult(
                column=col,
                check="presence",
                passed=present,
                detail=f"Column '{col}' missing from schema" if not present else "",
            )
        )

    if expected_types:
        for col, expected_alias in expected_types.items():
            if col not in actual_fields:
                continue  # already flagged above
            actual_type = actual_fields[col]
            allowed = _TYPE_ALIASES.get(expected_alias.lower(), (expected_alias,))
            match = any(actual_type.startswith(a) for a in allowed)
            results.append(
                ColumnValidationResult(
                    column=col,
                    check="type",
                    passed=match,
                    detail=(
                        ""
                        if match
                        else f"Column '{col}': expected {expected_alias}, got {actual_type}"
                    ),
                )
            )

    return ValidationResult(
        passed=all(r.passed for r in results),
        results=results,
    )
