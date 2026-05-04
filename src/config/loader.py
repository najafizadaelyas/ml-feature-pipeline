"""Loads and merges pipeline_config.yaml + per-entity feature definition YAMLs."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.config.models import FeatureDefinition, PipelineConfig


def _resolve_path(base_dir: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else base_dir / p


def load_pipeline_config(config_path: str | Path, run_date: str | None = None) -> PipelineConfig:
    """Parse pipeline_config.yaml into a validated PipelineConfig."""
    config_path = Path(config_path)
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    cfg = PipelineConfig(**raw)
    if run_date:
        cfg = cfg.model_copy(update={"run_date": run_date})
    return cfg


def load_feature_definitions(
    config: PipelineConfig,
    base_dir: str | Path | None = None,
) -> list[FeatureDefinition]:
    """Load all feature definition YAMLs referenced in the pipeline config."""
    if base_dir is None:
        base_dir = Path(os.environ.get("PIPELINE_CONFIG_DIR", "config"))

    base_dir = Path(base_dir)
    definitions: list[FeatureDefinition] = []

    for rel_path in config.feature_definition_files:
        full_path = _resolve_path(base_dir, rel_path)
        with full_path.open() as fh:
            raw = yaml.safe_load(fh)
        definitions.append(FeatureDefinition(**raw))

    return definitions
