"""
Configuration loader utility.

Loads YAML config files and provides dot-notation access
to configuration values throughout the project.
"""

from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str = "experiments/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class ConfigDict(dict):
    """
    Dictionary subclass that allows dot-notation access.

    Example:
        cfg = ConfigDict({'model': {'hidden_dim': 64}})
        cfg.model.hidden_dim  # 64
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
