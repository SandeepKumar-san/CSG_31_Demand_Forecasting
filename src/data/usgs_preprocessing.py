"""
USGS Mineral Commodity Data Preprocessing Pipeline — EXPANDED.

Processes ALL raw USGS MCS2026 data files into the format required by the
Adaptive Temporal-Structural Fusion model:

  Stage 1: Node Mapping — unique commodities → integer IDs + categories
  Stage 2: Edge Processing — relationship CSV → edge index/type/weight files
  Stage 3: Feature Extraction — long-format statistics → wide-format CSVs (12 signals)
  Stage 4: Static Node Features — Fig2/3/10/12/13/T4/T6/T7 → StaticNodeFeatures.csv

Output directory: data/processed/usgs/

Usage:
    from src.data.usgs_preprocessing import USGSPreprocessor
    p = USGSPreprocessor()
    p.run()
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Edge type string → integer ID mapping (15 types)
EDGE_TYPE_MAP: Dict[str, int] = {
    "supply_chain_input": 0,
    "technology_cluster": 1,
    "alloy_components": 2,
    "construction_coproduction": 3,
    "electronics_coproduction": 4,
    "substitution": 5,
    "battery_coproduction": 6,
    "critical_mineral_designation": 7,
    "geopolitical_supply_risk": 8,
    "price_correlation": 9,
    "recycling_secondary_source": 10,
    "byproduct_coproduction": 11,
    "catalyst_role": 12,
    "refractory_industrial": 13,
    "functional_coating": 14,
}

EDGE_TYPE_NAMES: Dict[int, str] = {v: k for k, v in EDGE_TYPE_MAP.items()}

# Symmetric edge types (should be bidirectional)
SYMMETRIC_EDGE_TYPES = {
    "substitution",
    "price_correlation",
    "alloy_components",
    "technology_cluster",
    "battery_coproduction",
}

# Confidence string → edge weight mapping
CONFIDENCE_WEIGHT_MAP: Dict[str, float] = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
}

# Material category assignment rules (Category ID → keywords)
CATEGORY_RULES: Dict[int, Tuple[str, List[str]]] = {
    0: ("Battery Materials", [
        "Lithium", "Cobalt", "Nickel", "Graphite", "Manganese",
    ]),
    1: ("Construction", [
        "Iron", "Steel", "Aluminum", "Cement", "Gypsum", "Sand", "Gravel",
        "Sand And Gravel", "Crushed Stone", "Stone", "Clays",
    ]),
    2: ("Electronics", [
        "Copper", "Gold", "Silver", "Silicon", "Gallium", "Germanium",
        "Indium", "Tellurium", "Selenium",
    ]),
    3: ("Energy", [
        "Uranium", "Coal", "Natural Gas", "Petroleum", "Peat", "Helium",
    ]),
    4: ("Rare Earths", [
        "Neodymium", "Dysprosium", "Praseodymium", "Lanthanum", "Cerium",
        "Yttrium", "Scandium", "Terbium", "Samarium", "Rare Earths",
        "Rare Earth", "Europium", "Gadolinium", "Holmium", "Erbium",
        "Thulium", "Ytterbium", "Lutetium",
    ]),
    5: ("Alloys", [
        "Chromium", "Chromite", "Molybdenum", "Tungsten", "Vanadium",
        "Niobium", "Tantalum", "Rhenium", "Hafnium",
    ]),
    6: ("Industrial", [
        "Sulfur", "Phosphate", "Potash", "Nitrogen", "Salt", "Soda Ash",
        "Boron", "Bromine", "Chlorine", "Fluorine", "Fluorspar", "Iodine",
        "Lime", "Magnesium", "Mica", "Pumice", "Talc", "Vermiculite",
        "Wollastonite", "Zeolites", "Barite", "Feldspar", "Kaolin",
        "Diatomite", "Perlite", "Strontium", "Titanium", "Zirconium",
    ]),
    7: ("Precious Metals", [
        "Platinum", "Palladium", "Rhodium", "Iridium", "Ruthenium",
        "Osmium",
    ]),
}

# Feature extraction — EXPANDED: Statistics column keyword matching (12 signals)
FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "Production": ["production", "mine production", "produced", "smelter"],
    "Consumption": ["consumption", "consumed", "apparent consumption"],
    "Imports": ["import"],
    "Exports": ["export"],
    "Price": ["price", "value", "unit value", "average price"],
    "Stocks": ["stock", "inventory", "stockpile"],
    "Capacity": ["capacity", "production capacity"],
    "Sales": ["sales", "sold", "sold or used", "shipment"],
    "Reserves": ["reserves", "reserve"],
    "Supply": ["supply", "apparent supply"],
    "Employment": ["employment", "employees"],
}


# ==============================================================================
# USGSPreprocessor
# ==============================================================================

class USGSPreprocessor:
    """
    Complete USGS data preprocessing pipeline — EXPANDED version.

    Processes ALL raw USGS MCS2026 CSV files into the format expected by
    the model's data loader:
      - NodesIndex.csv (node ID ↔ commodity name)
      - Node Types.csv (node ID → category)
      - Edge files (15 individual + 1 unified, with bidirectional edges)
      - Feature CSVs in wide format (Year × NodeID) — 12 signals
      - StaticNodeFeatures.csv — 11 features per node from supplementary files
      - Processing report

    Args:
        raw_dir: Path to raw USGS data directory.
        output_dir: Path to output processed data directory.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/usgs/",
        output_dir: str = "data/processed/usgs/",
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.report_lines: List[str] = []

        # Will be populated during processing
        self.nodes_df: Optional[pd.DataFrame] = None
        self.node_types_df: Optional[pd.DataFrame] = None
        self.name_to_id: Dict[str, int] = {}
        self.edges_df: Optional[pd.DataFrame] = None
        self.feature_dfs: Dict[str, pd.DataFrame] = {}
        self.main_df: Optional[pd.DataFrame] = None  # Cleaned main CSV

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full preprocessing pipeline (all 4 stages)."""
        os.makedirs(self.output_dir, exist_ok=True)

        start_time = time.time()
        self._log(f"USGS Preprocessing Pipeline (EXPANDED) — Started at {datetime.now()}")
        self._log("=" * 60)

        print("\n" + "=" * 60)
        print("  USGS DATA PREPROCESSING PIPELINE (EXPANDED)")
        print("=" * 60)

        # Stage 1: Node mapping
        print("\n[STEP] Stage 1: Node Mapping...")
        self._stage1_node_mapping()

        # Stage 2: Edge processing (with bidirectional edges)
        print("\n[STEP] Stage 2: Edge Processing (+ Bidirectional)...")
        self._stage2_edge_processing()

        # Stage 3: Temporal feature extraction (12 signals + World Production)
        print("\n[STEP] Stage 3: Feature Extraction (12 signals + World Production)...")
        self._stage3_feature_extraction()

        # Stage 4: Static node features from supplementary files
        print("\n[STEP] Stage 4: Static Node Features (from Fig/T files)...")
        self._stage4_static_features()

        # Validation
        print("\n[DONE] Stage 5: Validation...")
        self._validate()

        elapsed = time.time() - start_time
        self._log(f"\nTotal processing time: {elapsed:.1f} seconds")
        self._save_report()

        print(f"\n[DONE] All files saved to: {self.output_dir}")
        print(f"   Processing time: {elapsed:.1f}s")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Node Mapping
    # ------------------------------------------------------------------

    def _stage1_node_mapping(self) -> None:
        """Extract unique commodities, assign IDs, classify categories."""
        csv_path = os.path.join(self.raw_dir, "MCS2026_Commodities_Data.csv")
        df = pd.read_csv(csv_path, encoding="latin-1")

        # Extract unique commodity names, sorted alphabetically
        commodities = sorted(df["Commodity"].dropna().unique().tolist())
        n = len(commodities)

        # Create NodesIndex: node_id, commodity_name
        nodes_df = pd.DataFrame({
            "node_id": list(range(n)),
            "commodity_name": commodities,
        })
        nodes_df.to_csv(
            os.path.join(self.output_dir, "NodesIndex.csv"), index=False
        )
        self.nodes_df = nodes_df
        self.name_to_id = dict(zip(commodities, range(n)))

        # Create Node Types: node_id, commodity_name, category, category_name
        categories = []
        category_names = []
        for name in commodities:
            cat_id, cat_name = self._classify_commodity(name)
            categories.append(cat_id)
            category_names.append(cat_name)

        node_types_df = pd.DataFrame({
            "node_id": list(range(n)),
            "commodity_name": commodities,
            "category": categories,
            "category_name": category_names,
        })
        node_types_df.to_csv(
            os.path.join(self.output_dir, "Node Types.csv"), index=False
        )
        self.node_types_df = node_types_df

        # Log statistics
        self._log(f"\n--- Stage 1: Node Mapping ---")
        self._log(f"Total commodities: {n}")
        self._log(f"Node ID range: 0 to {n - 1}")
        cat_dist = node_types_df["category_name"].value_counts()
        self._log(f"Category distribution:")
        for cat_name, count in cat_dist.items():
            self._log(f"  {cat_name}: {count}")

        print(f"  ✓ {n} commodities mapped to IDs 0–{n - 1}")
        print(f"  ✓ 9 categories assigned")
        print(f"  ✓ Saved NodesIndex.csv and Node Types.csv")

    def _classify_commodity(self, name: str) -> Tuple[int, str]:
        """Assign a category ID to a commodity name."""
        name_upper = name.strip().upper()
        for cat_id, (cat_name, keywords) in CATEGORY_RULES.items():
            for kw in keywords:
                if kw.upper() in name_upper or name_upper in kw.upper():
                    return cat_id, cat_name
        # Check for "Rare Earth" substring
        if "RARE EARTH" in name_upper:
            return 4, "Rare Earths"
        return 8, "Other"

    # ------------------------------------------------------------------
    # Stage 2: Edge Processing (with Bidirectional Edges)
    # ------------------------------------------------------------------

    def _stage2_edge_processing(self) -> None:
        """Process relationship CSV into edge files with bidirectional edges."""
        rel_path = os.path.join(self.raw_dir, "pdf_extracted_relationships.csv")
        rels_df = pd.read_csv(rel_path)

        total_raw = len(rels_df)
        self._log(f"\n--- Stage 2: Edge Processing ---")
        self._log(f"Raw relationships loaded: {total_raw}")

        # Map material names to node IDs
        rels_df["source_id"] = rels_df["material1"].map(self.name_to_id)
        rels_df["target_id"] = rels_df["material2"].map(self.name_to_id)

        # Drop unmapped edges
        unmapped_mask = rels_df["source_id"].isna() | rels_df["target_id"].isna()
        n_unmapped = unmapped_mask.sum()
        if n_unmapped > 0:
            unmapped_materials = set()
            unmapped_materials.update(
                rels_df.loc[rels_df["source_id"].isna(), "material1"].unique()
            )
            unmapped_materials.update(
                rels_df.loc[rels_df["target_id"].isna(), "material2"].unique()
            )
            self._log(f"Unmapped edges dropped: {n_unmapped}")
            self._log(f"Unmapped materials: {sorted(unmapped_materials)}")
            print(f"  ⚠ {n_unmapped} edges dropped (unmapped materials)")

        rels_df = rels_df[~unmapped_mask].copy()
        rels_df["source_id"] = rels_df["source_id"].astype(int)
        rels_df["target_id"] = rels_df["target_id"].astype(int)

        # Remove self-loops
        self_loops = rels_df["source_id"] == rels_df["target_id"]
        n_self_loops = self_loops.sum()
        if n_self_loops > 0:
            rels_df = rels_df[~self_loops].copy()
            self._log(f"Self-loops removed: {n_self_loops}")

        # Remove duplicates (same source, target, type)
        before_dedup = len(rels_df)
        rels_df = rels_df.drop_duplicates(
            subset=["source_id", "target_id", "relationship_type"]
        )
        n_dupes = before_dedup - len(rels_df)
        if n_dupes > 0:
            self._log(f"Duplicate edges removed: {n_dupes}")

        # Map edge type strings to integer IDs
        rels_df["edge_type_id"] = rels_df["relationship_type"].map(EDGE_TYPE_MAP)

        # Handle any relationship types not in our map
        unknown_types = rels_df[rels_df["edge_type_id"].isna()]["relationship_type"].unique()
        if len(unknown_types) > 0:
            self._log(f"Unknown relationship types (dropped): {unknown_types.tolist()}")
            rels_df = rels_df.dropna(subset=["edge_type_id"])

        rels_df["edge_type_id"] = rels_df["edge_type_id"].astype(int)

        # Map confidence to edge weight
        rels_df["edge_weight"] = rels_df["confidence"].str.strip().str.lower().map(
            CONFIDENCE_WEIGHT_MAP
        )
        rels_df["edge_weight"] = rels_df["edge_weight"].fillna(0.7)  # default medium

        # ------- NEW: Add bidirectional edges for symmetric relationship types -------
        symmetric_edges = rels_df[
            rels_df["relationship_type"].isin(SYMMETRIC_EDGE_TYPES)
        ].copy()
        if len(symmetric_edges) > 0:
            reverse_edges = symmetric_edges.copy()
            reverse_edges["source_id"] = symmetric_edges["target_id"]
            reverse_edges["target_id"] = symmetric_edges["source_id"]

            # Combine original + reversed, then deduplicate
            rels_df = pd.concat([rels_df, reverse_edges], ignore_index=True)
            before = len(rels_df)
            rels_df = rels_df.drop_duplicates(
                subset=["source_id", "target_id", "edge_type_id"]
            )
            n_added = len(rels_df) - (before - len(reverse_edges))
            self._log(f"Bidirectional edges added: {len(reverse_edges)} reversed, "
                       f"{before - len(rels_df)} were duplicates")
            print(f"  ✓ Added bidirectional edges for {len(SYMMETRIC_EDGE_TYPES)} "
                  f"symmetric relationship types")

        self.edges_df = rels_df

        # Save unified all_edges.csv
        all_edges = rels_df[["source_id", "target_id", "edge_type_id", "edge_weight"]].copy()
        all_edges.to_csv(
            os.path.join(self.output_dir, "all_edges.csv"), index=False
        )

        # Save 15 individual edge files (one per type)
        for type_name, type_id in EDGE_TYPE_MAP.items():
            type_edges = rels_df[rels_df["edge_type_id"] == type_id]
            edge_file = os.path.join(
                self.output_dir, f"Edges ({type_name}).csv"
            )
            type_edges[["source_id", "target_id"]].to_csv(
                edge_file, index=False, header=True
            )

        # Log edge statistics
        total_edges = len(rels_df)
        self._log(f"Total edges after processing: {total_edges}")
        self._log(f"Edges per type:")
        for type_name, type_id in sorted(EDGE_TYPE_MAP.items(), key=lambda x: x[1]):
            count = len(rels_df[rels_df["edge_type_id"] == type_id])
            self._log(f"  [{type_id:2d}] {type_name}: {count}")

        weight_dist = rels_df["confidence"].str.strip().str.lower().value_counts()
        self._log(f"Edge weight distribution:")
        for conf, count in weight_dist.items():
            self._log(f"  {conf} ({CONFIDENCE_WEIGHT_MAP.get(conf, 0.7)}): {count}")

        print(f"  ✓ {total_edges} edges processed ({n_unmapped} dropped)")
        print(f"  ✓ 15 edge type files saved")
        print(f"  ✓ all_edges.csv saved")

    # ------------------------------------------------------------------
    # Stage 3: Feature Extraction (Long → Wide) — EXPANDED TO 12 SIGNALS
    # ------------------------------------------------------------------

    def _stage3_feature_extraction(self) -> None:
        """Extract 12 temporal features + World Production from main CSV."""
        csv_path = os.path.join(self.raw_dir, "MCS2026_Commodities_Data.csv")
        df = pd.read_csv(csv_path, encoding="latin-1")

        self._log(f"\n--- Stage 3: Feature Extraction (EXPANDED) ---")
        self._log(f"Main CSV shape: {df.shape}")

        # Clean the Statistics column
        df["Statistics"] = df["Statistics"].astype(str).str.strip()

        # Clean Year column — extract first 4-digit year from messy values
        df["Year"] = (
            df["Year"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

        # Clean Value column — remove commas, handle non-numeric
        df["Value"] = pd.to_numeric(
            df["Value"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )

        # Get all years
        all_years = sorted(df["Year"].dropna().unique().astype(int).tolist())
        self._log(f"Year range: {min(all_years)} - {max(all_years)}")
        self._log(f"Total years: {len(all_years)}")

        # Commodity → node_id mapping
        df["node_id"] = df["Commodity"].map(self.name_to_id)
        df = df.dropna(subset=["node_id"])
        df["node_id"] = df["node_id"].astype(int)

        # Store cleaned df for later
        self.main_df = df

        # ------- US Salient Statistics (11 features) -------
        # Filter to US data only (exclude World, Import Sources sections)
        us_df = df[
            df["Section"].str.contains("Salient", case=False, na=False)
        ].copy()

        for feature_name, keywords in FEATURE_KEYWORDS.items():
            self._extract_feature(us_df, feature_name, keywords, all_years)

        # ------- World Production (12th feature) -------
        world_df = df[
            df["Section"].str.contains("World", case=False, na=False)
        ].copy()
        world_prod_keywords = ["production", "mine production"]
        self._extract_feature(
            world_df, "WorldProduction", world_prod_keywords, all_years
        )

        # ------- World Capacity (13th feature) -------
        world_cap_keywords = ["capacity"]
        self._extract_feature(
            world_df, "WorldCapacity", world_cap_keywords, all_years
        )

        # ------- World Reserves (14th feature) -------
        world_res_keywords = ["reserves", "reserve"]
        self._extract_feature(
            world_df, "WorldReserves", world_res_keywords, all_years
        )

        # Align all feature files to same years and columns
        self._align_feature_files(all_years)

        print(f"  ✓ {len(self.feature_dfs)} features extracted (including World features)")
        print(f"  ✓ Years: {min(all_years)}–{max(all_years)} ({len(all_years)} years)")

    def _extract_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        keywords: List[str],
        all_years: List[int],
    ) -> None:
        """Extract a single feature, pivot to wide format, save."""
        # Filter rows matching any keyword (case-insensitive)
        mask = pd.Series(False, index=df.index)
        stats_lower = df["Statistics"].str.lower()
        for kw in keywords:
            mask = mask | stats_lower.str.contains(kw.lower(), na=False)

        feature_df = df[mask].copy()

        if len(feature_df) == 0:
            self._log(f"  {feature_name}: No data found (skipped)")
            print(f"    ⚠ {feature_name}: no matching rows")
            return

        # For commodities with multiple matching statistics rows per year,
        # take the mean (e.g., multiple types of production)
        pivot = feature_df.groupby(["Year", "node_id"])["Value"].mean().reset_index()

        # Pivot: rows=Year, columns=node_id
        wide = pivot.pivot(index="Year", columns="node_id", values="Value")

        # Ensure all node IDs are present as columns
        n_nodes = len(self.name_to_id)
        for nid in range(n_nodes):
            if nid not in wide.columns:
                wide[nid] = np.nan

        # Sort columns numerically and convert to string names
        wide = wide.reindex(columns=sorted(wide.columns))
        wide.columns = [str(c) for c in wide.columns]

        # Ensure all years present
        wide = wide.reindex(sorted(wide.index))

        # Forward fill, then fill remaining NaN with 0
        wide = wide.ffill().fillna(0)

        # Reset index so Year is a column
        wide.index.name = "Year"
        wide = wide.reset_index()
        wide["Year"] = wide["Year"].astype(int)

        # Save
        out_path = os.path.join(self.output_dir, f"{feature_name}.csv")
        wide.to_csv(out_path, index=False)
        self.feature_dfs[feature_name] = wide

        n_nonzero = (wide.iloc[:, 1:] != 0).sum().sum()
        self._log(
            f"  {feature_name}: {wide.shape[0]} years × {wide.shape[1] - 1} nodes, "
            f"{n_nonzero} non-zero values"
        )
        print(f"    ✓ {feature_name}: {wide.shape[0]} years × {wide.shape[1] - 1} nodes")

    def _align_feature_files(self, all_years: List[int]) -> None:
        """Ensure all feature files have identical year rows and node columns."""
        if not self.feature_dfs:
            return

        n_nodes = len(self.name_to_id)
        expected_cols = ["Year"] + [str(i) for i in range(n_nodes)]

        for name, df in self.feature_dfs.items():
            # Ensure all expected columns exist
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Reorder columns
            df = df[expected_cols].copy()

            # Save aligned version
            out_path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(out_path, index=False)
            self.feature_dfs[name] = df

    # ------------------------------------------------------------------
    # Stage 4: Static Node Features from Supplementary Files — NEW
    # ------------------------------------------------------------------

    def _stage4_static_features(self) -> None:
        """
        Build rich static features per node from Fig/T supplementary files.

        Output: StaticNodeFeatures.csv with columns:
          node_id, commodity_name,
          net_import_reliance, price_cagr_5yr, consumption_trend_5yr,
          supply_concentration, n_end_uses, world_production,
          is_critical_mineral, scrap_recycling_rate, domestic_value_rank,
          export_control_risk, trade_agreement_coverage
        """
        n_nodes = len(self.name_to_id)
        id_to_name = {v: k for k, v in self.name_to_id.items()}

        # Initialize feature dataframe
        static = pd.DataFrame({
            "node_id": list(range(n_nodes)),
            "commodity_name": [id_to_name[i] for i in range(n_nodes)],
        })

        # Default values
        static["net_import_reliance"] = 0.0
        static["price_cagr_5yr"] = 0.0
        static["consumption_trend_5yr"] = 0.0
        static["supply_concentration"] = 0.0
        static["n_end_uses"] = 1.0
        static["world_production"] = 0.0
        static["is_critical_mineral"] = 0
        static["scrap_recycling_rate"] = 0.0
        static["domestic_value_rank"] = 0.0
        static["export_control_risk"] = 0.0
        static["trade_agreement_coverage"] = 0.0

        # ----- Fig2: Net Import Reliance -----
        fig2_path = os.path.join(self.raw_dir, "MCS2026_Fig2_Net_Import_Reliance.csv")
        if os.path.exists(fig2_path):
            fig2 = pd.read_csv(fig2_path)
            matched = 0
            for _, row in fig2.iterrows():
                commodity = str(row.get("Commodity", "")).strip()
                nir_raw = str(row.get("Net_Import_Reliance_pct_2025", "0")).strip()
                # Handle '>75', '<25', 'E', 'W' etc.
                nir_val = self._parse_percentage(nir_raw)
                nid = self._fuzzy_match_commodity(commodity)
                if nid is not None:
                    static.loc[static["node_id"] == nid, "net_import_reliance"] = nir_val / 100.0
                    static.loc[static["node_id"] == nid, "is_critical_mineral"] = 1
                    matched += 1
            print(f"    ✓ Fig2 Net Import Reliance: {matched} minerals matched")
            self._log(f"  Fig2: {matched} minerals matched")

        # ----- Fig10: Price Growth Rates -----
        fig10_path = os.path.join(self.raw_dir, "MCS2026_Fig10_Price_Growth_Rates.csv")
        if os.path.exists(fig10_path):
            fig10 = pd.read_csv(fig10_path)
            matched = 0
            for _, row in fig10.iterrows():
                commodity = str(row.get("critical_mineral_priced", "")).strip()
                cagr = pd.to_numeric(str(row.get("CAGR_2021_2025", "0")).strip(),
                                      errors="coerce")
                if pd.isna(cagr):
                    cagr = 0.0
                nid = self._fuzzy_match_commodity(commodity)
                if nid is not None:
                    static.loc[static["node_id"] == nid, "price_cagr_5yr"] = cagr / 100.0
                    matched += 1
            print(f"    ✓ Fig10 Price CAGR: {matched} minerals matched")
            self._log(f"  Fig10: {matched} minerals matched")

        # ----- Fig12: Consumption Trend 5yr -----
        fig12_path = os.path.join(self.raw_dir, "MCS2026_Fig12_Pch_Consump_2021_2025.csv")
        if os.path.exists(fig12_path):
            fig12 = pd.read_csv(fig12_path)
            matched = 0
            for _, row in fig12.iterrows():
                commodity = str(row.get("Mineral_Commodity", "")).strip()
                pch = pd.to_numeric(str(row.get("Pch_Consumption_2021_2025", "0")).strip(),
                                     errors="coerce")
                if pd.isna(pch):
                    pch = 0.0
                nid = self._fuzzy_match_commodity(commodity)
                if nid is not None:
                    static.loc[static["node_id"] == nid, "consumption_trend_5yr"] = pch / 100.0
                    matched += 1
            print(f"    ✓ Fig12 Consumption Trend: {matched} minerals matched")
            self._log(f"  Fig12: {matched} minerals matched")

        # ----- Fig3: Supply Concentration (Herfindahl Index) -----
        fig3_path = os.path.join(self.raw_dir, "MCS2026_Fig3_Major_Import_Sources.csv")
        if os.path.exists(fig3_path):
            fig3 = pd.read_csv(fig3_path)
            matched = 0
            for commodity, group in fig3.groupby("Commodity"):
                nid = self._fuzzy_match_commodity(str(commodity).strip())
                if nid is not None:
                    shares = pd.to_numeric(group["Percent"], errors="coerce").dropna()
                    if len(shares) > 0:
                        # Herfindahl index: sum of squared shares (0-1 scale)
                        hhi = ((shares / 100.0) ** 2).sum()
                        static.loc[static["node_id"] == nid, "supply_concentration"] = min(hhi, 1.0)
                        matched += 1
            print(f"    ✓ Fig3 Supply Concentration: {matched} minerals matched")
            self._log(f"  Fig3: {matched} minerals matched")

        # ----- T6: End-Use Applications Count -----
        t6_path = os.path.join(self.raw_dir, "MCS2026_T6_Critical_Minerals_End_Use.csv")
        if os.path.exists(t6_path):
            t6 = pd.read_csv(t6_path)
            matched = 0
            for _, row in t6.iterrows():
                commodity = str(row.get("Critical_Mineral", "")).strip()
                apps_text = str(row.get("Primary_Applications", ""))
                # Count comma-separated applications + 1
                n_apps = len([a for a in apps_text.split(",") if a.strip()]) if apps_text else 1
                nid = self._fuzzy_match_commodity(commodity)
                if nid is not None:
                    static.loc[static["node_id"] == nid, "n_end_uses"] = float(n_apps)
                    matched += 1
            print(f"    ✓ T6 End-Use Applications: {matched} minerals matched")
            self._log(f"  T6: {matched} minerals matched")

        # ----- T7: World Production + Criticality -----
        t7_path = os.path.join(self.raw_dir, "MCS2026_T7_Critical_Minerals_Salient.csv")
        if os.path.exists(t7_path):
            t7 = pd.read_csv(t7_path)
            matched = 0
            for _, row in t7.iterrows():
                commodity = str(row.get("Critical_mineral", "")).strip()
                world_prod = pd.to_numeric(
                    str(row.get("World_total_prod", "0")).replace(",", "").strip(),
                    errors="coerce"
                )
                if pd.isna(world_prod):
                    world_prod = 0.0
                nid = self._fuzzy_match_commodity(commodity)
                if nid is not None:
                    static.loc[static["node_id"] == nid, "world_production"] = world_prod
                    static.loc[static["node_id"] == nid, "is_critical_mineral"] = 1
                    matched += 1
            print(f"    ✓ T7 World Production: {matched} minerals matched")
            self._log(f"  T7: {matched} minerals matched")

        # ----- Fig13: Scrap/Recycling Rates -----
        fig13_path = os.path.join(self.raw_dir, "MCS2026_Fig13_Scrap.csv")
        if os.path.exists(fig13_path):
            try:
                fig13 = pd.read_csv(fig13_path)
                matched = 0
                for _, row in fig13.iterrows():
                    # Try to find commodity and recycling rate columns
                    commodity = None
                    recycle_rate = 0.0
                    for col in fig13.columns:
                        if "mineral" in col.lower() or "commodity" in col.lower():
                            commodity = str(row[col]).strip()
                        if "recycle" in col.lower() or "scrap" in col.lower() or "rate" in col.lower():
                            recycle_rate = pd.to_numeric(str(row[col]).strip(), errors="coerce")
                            if pd.isna(recycle_rate):
                                recycle_rate = 0.0
                    if commodity:
                        nid = self._fuzzy_match_commodity(commodity)
                        if nid is not None:
                            static.loc[static["node_id"] == nid, "scrap_recycling_rate"] = recycle_rate / 100.0
                            matched += 1
                print(f"    ✓ Fig13 Scrap/Recycling: {matched} minerals matched")
                self._log(f"  Fig13: {matched} minerals matched")
            except Exception as e:
                print(f"    ⚠ Fig13 Scrap: Error reading file ({e})")

        # ----- T4: Export Control Risk -----
        t4_path = os.path.join(self.raw_dir, "MCS2026_T4_Country_Export_Control.csv")
        if not os.path.exists(t4_path):
            t4_path = os.path.join(
                self.raw_dir,
                "MCS2026_Mineral_Industry_Trends_and_Salient_Statistics",
                "MCS2026_T4_Country_Export_Control.csv",
            )
        if os.path.exists(t4_path):
            t4 = pd.read_csv(t4_path)
            matched = 0
            # Extract mineral names from the Commodity_Year column
            for _, row in t4.iterrows():
                text = str(row.get("Commodity_Year", "")).lower()
                # Try to match each commodity to this export control
                for name, nid in self.name_to_id.items():
                    if name.lower() in text or name.lower().split()[0] in text:
                        # Increment risk score based on control type
                        control_type = str(row.get("Control_Type", "")).lower()
                        risk_score = 1.0 if "ban" in control_type else 0.7
                        current = static.loc[static["node_id"] == nid, "export_control_risk"].values[0]
                        static.loc[static["node_id"] == nid, "export_control_risk"] = min(current + risk_score, 1.0)
                        matched += 1
            print(f"    ✓ T4 Export Controls: {matched} mineral-country pairs matched")
            self._log(f"  T4: {matched} matched")

        # ----- T3: Domestic State Value Rank -----
        t3_path = os.path.join(self.raw_dir, "MCS2026_T3_State_Value_Rank.csv")
        if not os.path.exists(t3_path):
            t3_path = os.path.join(
                self.raw_dir,
                "MCS2026_Mineral_Industry_Trends_and_Salient_Statistics",
                "MCS2026_T3_State_Value_Rank.csv",
            )
        if os.path.exists(t3_path):
            t3 = pd.read_csv(t3_path)
            # Count how many states produce each commodity
            commodity_state_count: Dict[int, int] = {}
            for _, row in t3.iterrows():
                commodities_text = str(row.get("Principal_commodities", "")).lower()
                for name, nid in self.name_to_id.items():
                    name_lower = name.lower()
                    # Check if any word from the commodity name appears
                    if name_lower in commodities_text or name_lower.split()[0] in commodities_text:
                        commodity_state_count[nid] = commodity_state_count.get(nid, 0) + 1
            matched = 0
            if commodity_state_count:
                max_states = max(commodity_state_count.values())
                for nid, count in commodity_state_count.items():
                    static.loc[static["node_id"] == nid, "domestic_value_rank"] = count / max(max_states, 1)
                    matched += 1
            print(f"    ✓ T3 Domestic Production: {matched} minerals found across states")
            self._log(f"  T3: {matched} matched")

        # ----- Normalize numeric columns -----
        # NOTE FOR PAPER: world_production here is normalized by max() relative to all commodities.
        # In Stage 3, WorldProduction temporal feature is generated as raw values.
        # This implies GAT static feature scale != TFT temporal feature scale downstream.
        numeric_cols = [
            "net_import_reliance", "price_cagr_5yr", "consumption_trend_5yr",
            "supply_concentration", "n_end_uses", "world_production",
            "scrap_recycling_rate", "domestic_value_rank",
            "export_control_risk", "trade_agreement_coverage",
        ]
        for col in numeric_cols:
            vals = static[col].values.astype(float)
            max_val = np.abs(vals).max()
            if max_val > 0:
                static[col] = vals / max_val  # Normalize to [-1, 1]

        # Save
        out_path = os.path.join(self.output_dir, "StaticNodeFeatures.csv")
        static.to_csv(out_path, index=False)

        n_nonzero = (static[numeric_cols] != 0).sum().sum()
        self._log(f"\n--- Stage 4: Static Node Features ---")
        self._log(f"  Output shape: {static.shape}")
        self._log(f"  Non-zero feature values: {n_nonzero}")
        print(f"  ✓ StaticNodeFeatures.csv: {static.shape[0]} nodes × {len(numeric_cols) + 1} features")
        print(f"  ✓ {n_nonzero} non-zero feature values")

    def _fuzzy_match_commodity(self, name: str) -> Optional[int]:
        """
        Fuzzy match a commodity name from supplementary files to our node IDs.

        Handles common mismatches like:
          "Aluminum, bauxite" → "Aluminum"
          "ARSENIC, all forms" → "Arsenic"
        """
        if not name or name.lower() in ("nan", "none", ""):
            return None

        # Clean the name
        clean = name.strip()

        # Direct match
        if clean in self.name_to_id:
            return self.name_to_id[clean]

        # Case-insensitive match
        clean_upper = clean.upper()
        for known_name, nid in self.name_to_id.items():
            if known_name.upper() == clean_upper:
                return nid

        # Take first word before comma (e.g., "Aluminum, bauxite" → "Aluminum")
        first_part = clean.split(",")[0].strip()
        if first_part in self.name_to_id:
            return self.name_to_id[first_part]
        for known_name, nid in self.name_to_id.items():
            if known_name.upper() == first_part.upper():
                return nid

        # Substring match (commodity name contained in known name or vice versa)
        for known_name, nid in self.name_to_id.items():
            known_upper = known_name.upper()
            if (re.search(r'\b' + re.escape(known_upper) + r'\b', clean_upper) or
                re.search(r'\b' + re.escape(clean_upper) + r'\b', known_upper)):
                return nid

        return None

    def _parse_percentage(self, val_str: str) -> float:
        """Parse a percentage string that may contain >, <, E, W markers."""
        val_str = str(val_str).strip()
        if val_str in ("", "nan", "None", "W", "E", "NA", "—"):
            return 0.0
        # Handle >75, <25 etc.
        val_str = val_str.replace(">", "").replace("<", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run validation checks and print summary."""
        self._log(f"\n--- Validation ---")

        # Node validation
        nodes_path = os.path.join(self.output_dir, "NodesIndex.csv")
        nodes = pd.read_csv(nodes_path)
        assert nodes["node_id"].is_monotonic_increasing, "Node IDs not sequential"
        assert nodes["node_id"].iloc[0] == 0, "Node IDs don't start at 0"
        n_nodes = len(nodes)
        self._log(f"✓ Nodes: {n_nodes} (IDs 0–{n_nodes - 1})")

        # Edge validation
        edges_path = os.path.join(self.output_dir, "all_edges.csv")
        edges = pd.read_csv(edges_path)
        assert edges["source_id"].between(0, n_nodes - 1).all(), "Invalid source_id"
        assert edges["target_id"].between(0, n_nodes - 1).all(), "Invalid target_id"
        assert edges["edge_type_id"].between(0, 14).all(), "Invalid edge_type_id"
        assert edges["edge_weight"].between(0.4, 1.0).all(), "Invalid edge_weight"
        assert (edges["source_id"] != edges["target_id"]).all(), "Self-loops found"
        n_edges = len(edges)
        self._log(f"✓ Edges: {n_edges} (all valid)")

        # Graph statistics
        unique_nodes_in_edges = pd.concat(
            [edges["source_id"], edges["target_id"]]
        ).nunique()
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        self._log(f"  Nodes participating in edges: {unique_nodes_in_edges}")
        self._log(f"  Average degree: {avg_degree:.1f}")
        self._log(f"  Graph density: {density:.4f}")
        self._log(f"  Self-loops: 0")

        # Feature validation
        for name, df in self.feature_dfs.items():
            n_years, n_cols = df.shape
            self._log(f"✓ {name}: {n_years} years × {n_cols - 1} nodes")

        # Static features validation
        static_path = os.path.join(self.output_dir, "StaticNodeFeatures.csv")
        if os.path.exists(static_path):
            static_df = pd.read_csv(static_path)
            self._log(f"✓ StaticNodeFeatures: {static_df.shape}")

        print(f"  ✓ All validation checks passed")
        print(f"  ✓ {n_nodes} nodes, {n_edges} edges, avg degree {avg_degree:.1f}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Add a line to the processing report."""
        self.report_lines.append(msg)
        logger.info(msg)

    def _save_report(self) -> None:
        """Save the processing report to file."""
        report_path = os.path.join(self.output_dir, "processing_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))
        print(f"  ✓ Processing report saved to {report_path}")


# ==============================================================================
# CLI entry point
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = USGSPreprocessor()
    preprocessor.run()
