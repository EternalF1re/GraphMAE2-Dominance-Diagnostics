"""Release-relative configuration path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_NAMES = {
    "ogbn-arxiv": "ogbn-arxiv_upstream_default.yaml",
}
PAPER_RECORD_KEYS = {
    "provenance",
    "execution",
    "model_and_optimizer",
    "linear_probe",
    "dataset_adaptation",
}


def resolve_config_path(dataset_name: str, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    else:
        candidate = CONFIG_DIR / DEFAULT_CONFIG_NAMES.get(dataset_name, f"{dataset_name}.yaml")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    return resolved


def load_runtime_config(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"GraphMAE2 runtime configuration must be a flat mapping: {resolved}")
    record_keys = PAPER_RECORD_KEYS.intersection(payload)
    if record_keys:
        raise ValueError(
            "This file is a paper execution/provenance record, not a runnable "
            "GraphMAE2 configuration. Use a flat runtime YAML with --config_path."
        )
    nested_keys = sorted(key for key, value in payload.items() if isinstance(value, Mapping))
    if nested_keys:
        raise ValueError(
            "GraphMAE2 runtime configuration must be a flat mapping; nested keys "
            f"are not accepted: {', '.join(str(key) for key in nested_keys)}"
        )
    return dict(payload)
