"""
Layer 2a: Temporal Fusion Transformer (TFT) Branch.

Captures temporal patterns from historical demand sequences using:
  1. Variable Selection Network (VSN) — learns which features matter
  2. LSTM Encoder — temporal sequence modeling
  3. Multi-Head Interpretable Attention — attends to important time steps
  4. Quantile Regression Heads — uncertainty estimation

Architecture follows Bryan Lim et al. (2021) "Temporal Fusion Transformers
for Interpretable Multi-horizon Time Series Forecasting".

Output: temporal embedding [batch, hidden_dim] + multi-horizon quantile forecasts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — core building block of TFT.

    GRN(x, c) = LayerNorm(x + GLU(W1 · ELU(W2 · x + W3 · c + b) + b2))

    Args:
        input_dim: Dimension of primary input.
        hidden_dim: Internal hidden dimension.
        output_dim: Output dimension.
        context_dim: Optional dimension of context vector (static covariates).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # *2 for GLU

        self.context_proj = None
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        self.skip_proj = None
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, ..., input_dim]
            context: Optional [batch, context_dim]

        Returns:
            [batch, ..., output_dim]
        """
        residual = x if self.skip_proj is None else self.skip_proj(x)

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            # Broadcast context to match x dimensions
            ctx = self.context_proj(context)
            if ctx.dim() < hidden.dim():
                ctx = ctx.unsqueeze(1).expand_as(hidden)
            hidden = hidden + ctx

        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)
        gated = self.fc2(hidden)

        # GLU activation: split into value and gate
        value, gate = gated.chunk(2, dim=-1)
        gated_output = torch.sigmoid(gate) * value

        return self.layer_norm(residual + gated_output)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) — learns feature importance.

    Applies GRN-based softmax gating to select which input features
    are most relevant for the forecasting task.

    Args:
        input_dim: Dimension of each input variable.
        num_vars: Number of input variables to select from.
        hidden_dim: Hidden dimension for GRN.
        context_dim: Optional static context dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Per-variable GRNs for feature transformation
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])

        # Joint GRN for variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_dim * num_vars, hidden_dim, num_vars,
            context_dim=context_dim, dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, num_vars, input_dim] or [batch, num_vars, input_dim]

        Returns:
            selected: [batch, (seq_len,) hidden_dim] weighted combination
            weights: [batch, (seq_len,) num_vars] selection weights
        """
        has_temporal = x.dim() == 4
        if has_temporal:
            batch, seq_len, num_vars, feat_dim = x.shape
        else:
            batch, num_vars, feat_dim = x.shape
            seq_len = 1
            x = x.unsqueeze(1)

        # Flatten variables for selection weight computation
        flat = x.reshape(batch, seq_len, -1)  # [B, T, num_vars * feat_dim]
        weights = self.selection_grn(flat, context)  # [B, T, num_vars]
        weights = F.softmax(weights, dim=-1)  # [B, T, num_vars]

        # Apply per-variable GRN and weight
        var_outputs = []
        for i in range(self.num_vars):
            var_outputs.append(self.var_grns[i](x[:, :, i, :]))  # [B, T, hidden]

        var_stack = torch.stack(var_outputs, dim=-1)  # [B, T, hidden, num_vars]
        selected = torch.sum(
            var_stack * weights.unsqueeze(2), dim=-1
        )  # [B, T, hidden]

        if not has_temporal:
            selected = selected.squeeze(1)
            weights = weights.squeeze(1)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT.

    Unlike standard multi-head attention, this variant shares values
    across heads, producing interpretable attention weights that can
    be directly analyzed.

    Args:
        hidden_dim: Model dimensionality (d_model).
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_q, hidden_dim]
            key: [batch, seq_k, hidden_dim]
            value: [batch, seq_v, hidden_dim]
            mask: Optional attention mask

        Returns:
            output: [batch, seq_q, hidden_dim]
            attn_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size = query.size(0)

        # Project and reshape for multi-head
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        output = self.out_proj(attn_output)

        return output, attn_weights


class TFTBranch(nn.Module):
    """
    Temporal Fusion Transformer (TFT) Branch — Layer 2a.

    Architecture:
      1. Input embeddings (continuous + categorical)
      2. Variable Selection Network
      3. LSTM Encoder (2 layers)
      4. Static covariate enrichment
      5. Interpretable Multi-Head Attention
      6. Temporal Fusion Decoder
      7. Quantile output heads

    Args:
        config: Configuration dictionary with 'model.tft' section.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        tft_cfg = config.get("model", {}).get("tft", {})
        self.hidden_dim = tft_cfg.get("hidden_dim", 64)
        self.num_layers = tft_cfg.get("num_layers", 2)
        self.num_heads = tft_cfg.get("num_heads", 4)
        self.dropout = tft_cfg.get("dropout", 0.1)
        self.num_unknown = tft_cfg.get("num_unknown_features", 4)
        self.num_static = tft_cfg.get("num_static_features", 2)

        data_cfg = config.get("data", {})
        self.num_horizons = len(data_cfg.get("forecast_horizons", [1, 3, 6, 12]))
        self.quantiles = data_cfg.get("quantiles", [0.1, 0.5, 0.9])

        # ---- Input projections ----
        self.input_proj = nn.Linear(self.num_unknown, self.hidden_dim)
        self.price_proj = nn.Linear(1, self.hidden_dim)
        self.static_proj = nn.Linear(self.num_static, self.hidden_dim)

        # ---- Variable Selection ----
        self.vsn = VariableSelectionNetwork(
            input_dim=self.hidden_dim,
            num_vars=2,  # main features + price
            hidden_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # ---- LSTM Encoder ----
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # ---- Static Covariate Encoders (4 context vectors) ----
        self.static_context_variable = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
        )
        self.static_context_enrichment = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
        )

        # ---- Multi-Head Attention ----
        self.attention = InterpretableMultiHeadAttention(
            self.hidden_dim, self.num_heads, self.dropout
        )
        self.attention_grn = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
        )
        self.attention_layer_norm = nn.LayerNorm(self.hidden_dim)

        # ---- Temporal Fusion Decoder ----
        self.decoder_grn = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
        )

        # ---- Output heads ----
        # Point forecast for each horizon
        self.forecast_head = nn.Linear(self.hidden_dim, self.num_horizons)

        # Quantile forecasts: [horizons, quantiles]
        self.quantile_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.num_horizons)
            for _ in self.quantiles
        ])

    def forward(
        self,
        time_series: torch.Tensor,
        price_history: torch.Tensor,
        static_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the TFT branch.

        Args:
            time_series: [batch, seq_len, num_unknown_features]
            price_history: [batch, seq_len]
            static_features: [batch, num_static]

        Returns:
            Dictionary with keys:
              - 'predictions': [batch, num_horizons, num_quantiles]
              - 'temporal_embedding': [batch, hidden_dim]
              - 'attention_weights': [batch, num_heads, 1, seq_len]
              - 'variable_weights': [batch, seq_len, num_vars]
        """
        batch_size = time_series.size(0)
        seq_len = time_series.size(1)

        # ---- 1. Input projections ----
        ts_embed = self.input_proj(time_series)         # [B, T, H]
        price_embed = self.price_proj(price_history.unsqueeze(-1))  # [B, T, H]
        static_embed = self.static_proj(static_features)  # [B, H]

        # ---- 2. Static covariate context ----
        cs = self.static_context_variable(static_embed)   # [B, H]
        ce = self.static_context_enrichment(static_embed)  # [B, H]

        # ---- 3. Variable Selection ----
        var_input = torch.stack([ts_embed, price_embed], dim=2)  # [B, T, 2, H]
        selected, var_weights = self.vsn(var_input, context=cs)   # [B, T, H], [B, T, 2]

        # ---- 4. LSTM Encoding ----
        # Initialize hidden state with static context
        h0 = cs.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm_encoder(selected, (h0, c0))  # [B, T, H]

        # ---- 5. Static enrichment ----
        enriched = lstm_out + ce.unsqueeze(1)  # [B, T, H]

        # ---- 6. Interpretable Multi-Head Attention ----
        # Use last timestep as query, full sequence as key/value
        query = enriched[:, -1:, :]  # [B, 1, H]
        attn_out, attn_weights = self.attention(
            query, enriched, enriched
        )  # [B, 1, H], [B, heads, 1, T]

        # Gated residual after attention
        attn_out = self.attention_layer_norm(
            query + self.attention_grn(attn_out)
        )  # [B, 1, H]

        # ---- 7. Temporal Fusion Decoder ----
        decoded = self.decoder_grn(attn_out)  # [B, 1, H]
        temporal_embedding = decoded.squeeze(1)  # [B, H]

        # ---- 8. Output heads ----
        point_forecast = self.forecast_head(temporal_embedding)  # [B, horizons]

        quantile_preds = []
        for qhead in self.quantile_heads:
            quantile_preds.append(qhead(temporal_embedding))  # [B, horizons]
        quantile_predictions = torch.stack(quantile_preds, dim=-1)  # [B, horizons, Q]

        return {
            "predictions": quantile_predictions,
            "point_forecast": point_forecast,
            "temporal_embedding": temporal_embedding,
            "attention_weights": attn_weights,
            "variable_weights": var_weights,
        }
