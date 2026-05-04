"""Utility helpers shared across DAG tasks."""

from __future__ import annotations

import os
from pathlib import Path


def get_config_path() -> Path:
    """Return absolute path to pipeline_config.yaml.

    Checks PIPELINE_CONFIG_PATH env var first, then falls back to
    <repo_root>/config/pipeline_config.yaml.
    """
    env_path = os.environ.get("PIPELINE_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # Resolve relative to this file: dags/utils/ → ../../config/
    return Path(__file__).resolve().parents[2] / "config" / "pipeline_config.yaml"


def get_config_dir() -> Path:
    """Return the directory containing pipeline_config.yaml."""
    return get_config_path().parent
