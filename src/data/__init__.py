"""
Data module for Adaptive Temporal-Structural Fusion Model.

REAL DATA ONLY — This is for a research paper.
No synthetic data generation. If SupplyGraph data is missing,
the loader prints download instructions and exits.
"""

from src.data.supplygraph_loader import SupplyGraphLoader


def get_data_loader(config: dict):
    """
    Factory function to return the appropriate data loader.

    Currently supports:
      - "supplygraph": Real SupplyGraph dataset from Kaggle (PRIMARY)
      - "usgs": USGS mineral data (placeholder for future integration)

    If data is not found, prints download instructions and exits.
    NO SYNTHETIC DATA — this is for a RESEARCH PAPER.

    Args:
        config: Full configuration dictionary.

    Returns:
        An instance of the appropriate data loader.
    """
    source = config["data"]["primary_source"]

    if source == "supplygraph":
        return SupplyGraphLoader(config)
    elif source == "usgs":
        if config["data"]["usgs"].get("enabled", False):
            from src.data.usgs_loader import USGSLoader
            return USGSLoader(config)
        else:
            print("⚠️ USGS data not yet enabled. Using SupplyGraph.")
            return SupplyGraphLoader(config)
    else:
        raise ValueError(
            f"Unknown data source: '{source}'. "
            f"Valid options: 'supplygraph', 'usgs'"
        )
