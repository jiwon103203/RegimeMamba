"""Shared I/O helpers for the rolling-window backtest scripts.

Logging setup, timestamped output directories, config dumps, YAML overrides and
JSON checkpoints used to be copy-pasted into every script under ``scripts/``.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def json_serializer(obj):
    """JSON 직렬화를 위한 변환 함수 - 순환 참조 처리 및 다양한 타입 지원

    Args:
        obj: 직렬화할 객체

    Returns:
        직렬화 가능한 객체
    """
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # 다른 유형의 경우 문자열로 변환 시도
        try:
            return str(obj)
        except Exception:
            return None


def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration

    Args:
        log_file: Path to log file (optional)
        log_level: Logging level

    Returns:
        logging.Logger: Logger instance
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def prepare_output_directory(output_dir: str, prefix: str = "train_backtest") -> Tuple[str, str]:
    """Create a timestamped result directory ``<output_dir>/<prefix>_<timestamp>``.

    Args:
        output_dir: Base output directory
        prefix: Directory / log file prefix (e.g. ``train_backtest``, ``e2e_backtest``)

    Returns:
        tuple: (result_dir, log_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"{prefix}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    log_file = os.path.join(result_dir, f"{prefix}.log")

    return result_dir, log_file


def config_to_dict(config) -> Dict[str, Any]:
    """Collect the public, non-callable attributes of a config object."""
    config_dict = {key: getattr(config, key) for key in dir(config)
                   if not key.startswith('__') and not callable(getattr(config, key))}
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    return config_dict


def save_config_files(config, output_dir: str, title: str,
                      sections: Optional[Iterable[Tuple[str, Iterable[str]]]] = None,
                      sort_keys: bool = False):
    """Save ``config.yaml`` and a human readable ``config.txt``.

    Args:
        config: Configuration object
        output_dir: Output directory
        title: Title written at the top of ``config.txt``
        sections: Optional ``(section_title, keys)`` pairs written before the remaining keys
        sort_keys: Whether to sort the remaining keys alphabetically
    """
    config_dict = config_to_dict(config)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    written = set()
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f"=== {title} Configuration ===\n\n")
        if sections:
            for section_title, keys in sections:
                f.write(f"--- {section_title} ---\n")
                for key in keys:
                    if key in config_dict:
                        f.write(f"{key}: {config_dict[key]}\n")
                        written.add(key)
                f.write("\n")
            f.write("--- Other Parameters ---\n")

        items = sorted(config_dict.items()) if sort_keys else config_dict.items()
        for key, value in items:
            if key not in written:
                f.write(f"{key}: {value}\n")


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file, returning an empty dict when ``path`` is missing."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def apply_overrides(config, values: Dict[str, Any], skip_none: bool = True):
    """Set every key of ``values`` that already exists as an attribute of ``config``."""
    for key, value in values.items():
        if skip_none and value is None:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict[str, Any]: Checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading checkpoint: {str(e)}")


def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_path: str):
    """Save checkpoint

    Args:
        checkpoint_data: Checkpoint data
        checkpoint_path: Path to save checkpoint
    """
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, default=json_serializer, indent=4)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
