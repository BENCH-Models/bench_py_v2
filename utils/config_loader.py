"""Load and normalize BENCH model configuration files."""

import json
import os
from typing import Any, Dict, List

from utils.constants import (
    CASE_STUDIES,
    SCENARIOS,
    POLICIES,
    LEARNING_TYPES,
    DEFAULT_LEARNING_TYPE,
)

try:
    import yaml
except ImportError:
    yaml = None


def _read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _read_yaml(path: str) -> Any:
    if yaml is None:
        raise ImportError(
            'PyYAML is not installed. Install it to read YAML configuration files.'
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config_file(path: str) -> List[Dict[str, Any]]:
    """Load a JSON or YAML config file and return a list of run configurations."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    lower = path.lower()
    if lower.endswith('.yaml') or lower.endswith('.yml'):
        data = _read_yaml(path)
    else:
        data = _read_json(path)

    if isinstance(data, dict):
        configs = [data]
    elif isinstance(data, list):
        configs = data
    else:
        raise ValueError(
            "Configuration file must contain either an object or an array of objects."
        )

    return [normalize_run_config(config) for config in configs]


def normalize_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config values and fill defaults."""
    normalized = dict(config)

    if 'learning' in normalized and 'learning_type' not in normalized:
        normalized['learning_type'] = normalized.pop('learning')

    normalized['case_study'] = normalized.get('case_study')
    normalized['scenario'] = normalized.get('scenario')
    normalized['policy'] = normalized.get('policy')
    normalized['learning_type'] = normalized.get('learning_type',)
    normalized['run_label'] = normalized.get('run_label')
    normalized['debug'] = normalized.get('debug')

    # ADD THIS - Carbon price awareness flag (default to True)
    normalized['carbon_price_awareness'] = normalized.get('carbon_price_awareness', True)


    # Keep values even if not in known lists so experiments can use custom labels.
    if normalized['case_study'] not in CASE_STUDIES:
        print(f"Warning: using unknown case_study '{normalized['case_study']}'")
    if normalized['scenario'] not in SCENARIOS:
        print(f"Warning: using unknown scenario '{normalized['scenario']}'")
    if normalized['policy'] not in POLICIES:
        print(f"Warning: using unknown policy '{normalized['policy']}'")
    if normalized['learning_type'] not in LEARNING_TYPES:
        print(f"Warning: using unknown learning_type '{normalized['learning_type']}'")

    return normalized
