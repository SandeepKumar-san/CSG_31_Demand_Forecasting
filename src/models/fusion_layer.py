"""
Layer 3: Adaptive Fusion Layer ⭐ (KEY INNOVATION).

Learns WHEN to trust temporal vs structural signals, conditioned
on material type and forecast horizon. This is the novel contribution.

Mechanism:
  - Material type embedding (8 categories → 16 dim)
  - Forecast horizon embedding (4 horizons → 16 dim)
  - Cross-attention between temporal and structural embeddings
  - Alpha predictor MLP: [h_temporal, h_structural, mat_emb, hor_emb] → alpha
  - Fusion: fused = alpha * h_temporal + (1 - alpha) * h_structural

Alpha values are stored for interpretability and visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class AdaptiveFusionLayer(nn.Module):
    """
    Adaptive Fusion Layer — Layer 3 (Novel Contribution).

    Dynamically learns fusion weights (alpha) that are specific to:
      1. Material type (8 categories with distinct supply-chain dynamics)
      2. Forecast horizon (short-term vs long-term prediction)
      3. Current input signals (temporal and structural embeddings)

    When alpha → 1.0: model trusts temporal (TFT) signal more.
    When alpha → 0.0: model trusts structural (GAT) signal more.

    Args:
        config: Configuration dictionary with 'model' and 'model.fusion' sections.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.hidden_dim = config.get("model", {}).get("hidden_dim", 64)
        fusion_cfg = config.get("model", {}).get("fusion", {})
        self.num_material_types = fusion_cfg.get("num_material_types", 8)
        self.num_horizons = fusion_cfg.get("num_horizons", 4)
        mat_embed_dim = fusion_cfg.get("material_embed_dim", 16)
        hor_embed_dim = fusion_cfg.get("horizon_embed_dim", 16)

        # ---- Context Embeddings ----
        self.material_embed = nn.Embedding(self.num_material_types, mat_embed_dim)
        self.horizon_embed = nn.Embedding(self.num_horizons, hor_embed_dim)

        # ---- Cross-Attention (optional enrichment) ----
        # Temporal attends to structural and vice versa
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # ---- Alpha Predictor MLP ----
        # Input: [h_temporal, h_structural, material_emb, horizon_emb]
        alpha_input_dim = self.hidden_dim * 2 + mat_embed_dim + hor_embed_dim
        self.alpha_mlp = nn.Sequential(
            nn.Linear(alpha_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),  # alpha ∈ [0, 1]
        )

        # ---- Post-fusion projection ----
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(
        self,
        h_temporal: torch.Tensor,
        h_structural: torch.Tensor,
        material_type: torch.Tensor,
        horizon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive fusion of temporal and structural embeddings.

        Args:
            h_temporal: [batch, hidden_dim] from TFT branch.
            h_structural: [batch, hidden_dim] from GAT branch.
            material_type: [batch] integer indices (0 to num_material_types-1).
            horizon: [batch] integer indices (0 to num_horizons-1).

        Returns:
            fused: [batch, hidden_dim] adaptively fused embedding.
            alpha: [batch, 1] fusion weight (1.0=temporal, 0.0=structural).
        """
        # ---- Embed context ----
        z_material = self.material_embed(material_type)   # [batch, mat_embed_dim]
        z_horizon = self.horizon_embed(horizon)            # [batch, hor_embed_dim]

        # ---- Cross-attention enrichment ----
        # Temporal embedding enriched by structural context
        h_t_enriched, _ = self.cross_attention(
            h_temporal.unsqueeze(1),    # query  [B, 1, H]
            h_structural.unsqueeze(1),  # key    [B, 1, H]
            h_structural.unsqueeze(1),  # value  [B, 1, H]
        )
        h_t_enriched = h_t_enriched.squeeze(1) + h_temporal  # Residual [B, H]

        # ---- Predict alpha ----
        fusion_input = torch.cat(
            [h_t_enriched, h_structural, z_material, z_horizon], dim=-1
        )  # [batch, 2*H + mat_dim + hor_dim]

        alpha = self.alpha_mlp(fusion_input)  # [batch, 1]

        # ---- Adaptive fusion ----
        # fused = alpha * temporal + (1 - alpha) * structural
        fused = alpha * h_temporal + (1.0 - alpha) * h_structural  # [batch, H]

        # Post-fusion projection with layer norm
        fused = self.fusion_proj(fused)

        return fused, alpha
