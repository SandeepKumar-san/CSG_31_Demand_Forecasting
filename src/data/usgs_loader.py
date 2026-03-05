"""
USGS Mineral Data Loader (Placeholder).

Placeholder for future USGS mineral commodity data integration.
Currently returns None/mock data with the proper schema.

USGS integration planned for:
  - 90 mineral commodities
  - 20+ years of annual/monthly data
  - Production, imports, exports, consumption, prices
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd


class USGSLoader:
    """
    Placeholder for future USGS mineral data integration.

    When implemented, this will:
      1. Load USGS mineral commodity data (CSV or API)
      2. Process 90 commodities across 20+ years
      3. Build supply-chain graphs from trade relationships

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        usgs_cfg = config.get("data", {}).get("usgs", {})
        self.data_path = usgs_cfg.get("path", "data/raw/usgs/")
        self.n_commodities = usgs_cfg.get("n_commodities", 90)
        self.start_year = usgs_cfg.get("start_year", 2000)
        self.end_year = usgs_cfg.get("end_year", 2023)
        self.enabled = usgs_cfg.get("enabled", False)

    def load_data(self) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Load USGS mineral commodity data.

        Returns:
            None (placeholder). When implemented, returns
            (dataframe, metadata_dict) matching SupplyGraphLoader schema.
        """
        # TODO: Implement USGS API or CSV loading
        # Expected schema:
        #   - commodity_id, year, month
        #   - production, imports, exports, consumption
        #   - price_domestic, price_world
        #   - reserves, recycling_rate
        print(
            "[USGSLoader] USGS loader not yet implemented. "
            "Use SupplyGraph dataset for now."
        )
        print(
            f"[USGSLoader] Planned: {self.n_commodities} commodities, "
            f"{self.start_year}-{self.end_year}"
        )
        return None

    def prepare_datasets(self):
        """Placeholder — returns None."""
        print("[USGSLoader] Dataset preparation not yet implemented.")
        return None, None, None, None
