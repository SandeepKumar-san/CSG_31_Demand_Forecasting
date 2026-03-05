"""
Reproducible graph builder for supply-chain material dependency networks.

NOTE: For the SupplyGraph dataset, the pre-computed EdgesIndex/ CSVs
are loaded directly by SupplyGraphLoader. This module provides
ADDITIONAL utilities for custom graph construction and analysis.

Edge types: product_group, sub_group, plant, storage.
All operations are seeded for deterministic output.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


class ReproducibleGraphBuilder:
    """
    Build a supply-chain graph with deterministic edge ordering.

    For the SupplyGraph dataset, edges are loaded directly from
    EdgesIndex/ CSVs by SupplyGraphLoader. This class provides:
      - Custom graph construction from raw material properties
      - Connection strength computation
      - Graph analysis utilities

    Nodes represent materials/products. Edges represent relationships:
      - Same product_group (type 0)
      - Same sub_group (type 1)
      - Same plant (type 2)
      - Same storage location (type 3)

    Each edge type is encoded as a one-hot feature (4 dimensions).

    Args:
        seed: Random seed for reproducibility.
    """

    EDGE_TYPE_NAMES: List[str] = [
        "product_group",
        "sub_group",
        "plant",
        "storage",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def build_graph(
        self,
        material_properties: List[Dict[str, Any]],
        demand_data: Optional[np.ndarray] = None,
        price_data: Optional[np.ndarray] = None,
        production_data: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from material properties.

        This method is used for custom graph construction (e.g. when not
        loading from pre-computed EdgesIndex/ CSVs).

        Args:
            material_properties: List of dicts with keys: product_id,
                product_group, sub_group, plant, storage, lead_time, base_price.
            demand_data: [n_products, n_timepoints] demand array.
            price_data: [n_products, n_timepoints] price array.
            production_data: [n_products, n_timepoints] production array.

        Returns:
            torch_geometric.data.Data with x, edge_index, edge_attr, and
            edge_type tensors.
        """
        n_nodes = len(material_properties)

        # ---- Node features ----
        node_features = self._compute_node_features(
            material_properties, demand_data, price_data, production_data
        )

        # ---- Edges ----
        edges, edge_types = self._compute_edges(material_properties)

        # Sort edges for deterministic ordering (CRITICAL for reproducibility)
        sorted_idx = sorted(range(len(edges)), key=lambda i: (edges[i][0], edges[i][1]))
        edges = [edges[i] for i in sorted_idx]
        edge_types = [edge_types[i] for i in sorted_idx]

        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = self._encode_edge_types(edge_types)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(self.EDGE_TYPE_NAMES)), dtype=torch.float)

        # ---- Build Data object ----
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_nodes,
        )

        # Store additional metadata
        data.material_types = torch.tensor(
            [p.get("category_idx", p.get("sub_group", 0)) for p in material_properties],
            dtype=torch.long,
        )
        data.lead_times = torch.tensor(
            [p["lead_time"] for p in material_properties],
            dtype=torch.float,
        )

        print(
            f"[GraphBuilder] Built graph: {n_nodes} nodes, "
            f"{edge_index.shape[1]} edges, "
            f"{node_features.shape[1]} node features"
        )
        return data

    def _compute_node_features(
        self,
        props: List[Dict[str, Any]],
        demand_data: Optional[np.ndarray],
        price_data: Optional[np.ndarray],
        production_data: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Compute node features for each material.

        Features (4 per node):
          - recent_demand_avg (last 7 days average)
          - price_level (current price / average price)
          - production_capacity_util (production / max_production)
          - lead_time_days (normalized to 0-1, max 30 days)
        """
        n_nodes = len(props)
        features = np.zeros((n_nodes, 4), dtype=np.float32)

        for i, p in enumerate(props):
            # Recent demand average
            if demand_data is not None and i < demand_data.shape[0]:
                recent = demand_data[i, -7:] if demand_data.shape[1] >= 7 else demand_data[i]
                features[i, 0] = np.mean(recent)
            else:
                features[i, 0] = p.get("base_demand", 100.0)

            # Price level (current / average)
            if price_data is not None and i < price_data.shape[0]:
                avg_price = np.mean(price_data[i]) + 1e-8
                features[i, 1] = price_data[i, -1] / avg_price
            else:
                features[i, 1] = 1.0

            # Production capacity utilization
            if production_data is not None and i < production_data.shape[0]:
                max_prod = np.max(production_data[i]) + 1e-8
                features[i, 2] = np.mean(production_data[i, -7:]) / max_prod
            else:
                features[i, 2] = 0.7

            # Lead time (normalized to 0-1 range, max 30 days)
            features[i, 3] = p["lead_time"] / 30.0

        return torch.tensor(features, dtype=torch.float)

    def _compute_edges(
        self, props: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Compute edges based on shared attributes.
        Edge types: 0=product_group, 1=sub_group, 2=plant, 3=storage
        """
        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        n = len(props)

        for i in range(n):
            for j in range(i + 1, n):
                for k, attr in enumerate(self.EDGE_TYPE_NAMES):
                    if props[i].get(attr) == props[j].get(attr):
                        # Add bidirectional edges
                        edges.append((i, j))
                        edge_types.append(k)
                        edges.append((j, i))
                        edge_types.append(k)

        return edges, edge_types

    def _encode_edge_types(self, edge_types: List[int]) -> torch.Tensor:
        """One-hot encode edge types (4 dimensions)."""
        n_types = len(self.EDGE_TYPE_NAMES)
        one_hot = torch.zeros((len(edge_types), n_types), dtype=torch.float)
        for i, t in enumerate(edge_types):
            one_hot[i, t] = 1.0
        return one_hot

    def compute_connection_strength(
        self,
        material_properties: List[Dict[str, Any]],
        demand_data: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Compute edge connection strength from demand co-occurrence patterns.

        Products with correlated demand patterns have stronger connections.
        """
        n = len(material_properties)
        strength = torch.zeros((n, n), dtype=torch.float)

        if demand_data is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if i < demand_data.shape[0] and j < demand_data.shape[0]:
                        corr = np.corrcoef(demand_data[i], demand_data[j])[0, 1]
                        if not np.isnan(corr):
                            s = abs(corr)
                            strength[i, j] = s
                            strength[j, i] = s

        return strength
