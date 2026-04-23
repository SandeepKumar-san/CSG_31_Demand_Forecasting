"""
Configuration loader utility.

Loads YAML config files and provides dictionary-based access
to configuration values throughout the project.
"""

from typing import Any, Dict, Optional

import yaml


def update_dynamic_parameters(config: dict, info: dict, graph_data: Any) -> None:
    """
    Update model parameters that depend strictly on the loaded data dimensions.
    
    This handles input/output shapes, not hyperparameters.
    """
    # 1. Material types (Layer 3/4)
    actual_mat_types = info["metadata"].get("n_material_types", 2)
    config["model"]["fusion"]["num_material_types"] = max(actual_mat_types, 2)

    # 2. Time-series features (Layer 2a)
    config["model"]["tft"]["num_unknown_features"] = info["n_features"]

    # 3. Forecast horizons (Layer 3/4)
    horizons = config["data"]["forecast_horizons"]
    config["model"]["fusion"]["num_horizons"] = len(horizons)

    # 4. Graph structure (Layer 2b)
    if "num_edge_types" in info:
        config["model"]["gat"]["num_edge_types"] = info["num_edge_types"]
        config["model"]["gat"]["num_edge_features"] = info["num_edge_types"] + 1

    if graph_data is not None and hasattr(graph_data, "x"):
        config["model"]["gat"]["num_node_features"] = graph_data.x.shape[1]
        # Static features in TFT often match node features
        config["model"]["tft"]["num_static_features"] = graph_data.x.shape[1]

    print(f"\n  [Config] Dynamic parameters updated from data:")
    print(f"    - num_material_types: {config['model']['fusion']['num_material_types']}")
    print(f"    - num_unknown_feat:   {config['model']['tft']['num_unknown_features']} (Temporal)")
    print(f"    - num_horizons:       {config['model']['fusion']['num_horizons']}")
    print(f"    - num_node_features:  {config['model']['gat']['num_node_features']} (Static)")
    print(f"    - num_edge_types:     {config['model']['gat'].get('num_edge_types', 'N/A')} (Relational)")


def load_config(config_path: str = "experiments/config.yaml") -> Dict[str, Any]:
    """Load the raw, full configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_config(
    dataset_name: str, config_path: str = "experiments/config.yaml"
) -> Dict[str, Any]:
    """
    Load configuration and merge dataset-specific overrides.

    Args:
        dataset_name: Name of the dataset ("supplygraph" or "usgs").
        config_path: Path to the YAML file.

    Returns:
        Flattened configuration dictionary for the specific dataset.
    """
    full_config = load_config(config_path)

    # 1. Start with 'common' settings
    merged = full_config.get("common", {}).copy()

    # 2. Add dataset-specific settings
    ds_cfg = full_config.get("datasets", {}).get(dataset_name)
    if not ds_cfg:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in 'datasets' section of {config_path}"
        )

    # 3. Merge (Dataset-specific overrides common)
    for key, value in ds_cfg.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Shallow merge of top-level dicts (e.g., risk_scoring)
            merged[key].update(value)
        else:
            merged[key] = value

    # 4. Inject dataset name for legacy compatibility
    if "data" not in merged:
        merged["data"] = {}
    merged["data"]["dataset"] = dataset_name

    return merged

