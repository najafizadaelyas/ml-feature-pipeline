"""Shared pytest fixtures — local SparkSession for unit tests."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Provide a local[2] SparkSession for the entire test session."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ml-feature-pipeline-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
