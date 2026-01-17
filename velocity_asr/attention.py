"""
Hierarchical Global Context Module for VELOCITY-ASR v2.

This module implements the hierarchical pooling and attention mechanisms
that provide global context awareness with sub-quadratic complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ssm import GlobalSSM


class AdaptivePool(nn.Module):
    """
    Adaptive pooling layer that adjusts pool size based on sequence length.

    Level 1: K1 = max(64, L/8)
    Level 2: K2 = min(64, max(16, K1/4))

    Args:
        level: Pooling level (1 or 2)
        d_model: Feature dimension
    """

    def __init__(self, level: int = 1, d_model: int = 192):
        super().__init__()
        self.level = level
        self.d_model = d_model

        # Learnable pooling weights
        self.pool_proj = nn.Linear(d_model, d_model)

    def _compute_pool_size(self, seq_len: int, prev_pool_size: Optional[int] = None) -> int:
        """Compute adaptive pool size based on level and sequence length."""
        if self.level == 1:
            return max(64, seq_len // 8)
        else:
            # Level 2
            k1 = prev_pool_size if prev_pool_size else max(64, seq_len // 8)
            return min(64, max(16, k1 // 4))

    def forward(
        self,
        x: torch.Tensor,
        prev_pool_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass with adaptive pooling.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            prev_pool_size: Previous pooling size (for level 2)

        Returns:
            Tuple of:
                - Pooled tensor (batch, pool_size, d_model)
                - Pool size used
        """
        batch, seq_len, _ = x.shape
        pool_size = self._compute_pool_size(seq_len, prev_pool_size)

        # Ensure pool_size doesn't exceed sequence length
        pool_size = min(pool_size, seq_len)

        # Apply adaptive average pooling
        # Reshape for pooling: (B, D, L)
        x_t = x.transpose(1, 2)
        x_pooled = F.adaptive_avg_pool1d(x_t, pool_size)
        x_pooled = x_pooled.transpose(1, 2)  # (B, pool_size, D)

        # Apply learnable projection
        x_pooled = self.pool_proj(x_pooled)

        return x_pooled, pool_size


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for global context aggregation.

    Uses smaller attention dimension for efficiency.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_dim: Dimension per head (default: 48)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 192,
        num_heads: int = 4,
        attention_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, attention_dim)
        self.k_proj = nn.Linear(d_model, attention_dim)
        self.v_proj = nn.Linear(d_model, attention_dim)
        self.out_proj = nn.Linear(attention_dim, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch, q_len, d_model)
            key: Key tensor (batch, kv_len, d_model)
            value: Value tensor (batch, kv_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, q_len, d_model)
        """
        batch, q_len, _ = query.shape
        kv_len = key.shape[1]

        # Project to attention space
        q = self.q_proj(query)  # (B, Q, A)
        k = self.k_proj(key)    # (B, K, A)
        v = self.v_proj(value)  # (B, K, A)

        # Reshape for multi-head attention
        q = q.view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, Q, K)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, Q, head_dim)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.attention_dim)
        out = self.out_proj(out)

        return out


class GatedFusion(nn.Module):
    """
    Gated fusion layer for combining local and global features.

    Uses a learned gate to adaptively combine the two feature streams.

    Args:
        d_model: Feature dimension
    """

    def __init__(self, d_model: int = 192):
        super().__init__()

        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Feature transformation
        self.local_proj = nn.Linear(d_model, d_model)
        self.global_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse local and global features with adaptive gating.

        Args:
            local_features: Local SSM output (batch, seq_len, d_model)
            global_features: Global context (batch, seq_len, d_model)

        Returns:
            Fused features (batch, seq_len, d_model)
        """
        # Compute adaptive gate
        concat = torch.cat([local_features, global_features], dim=-1)
        gate = self.gate_proj(concat)  # (B, L, D)

        # Transform features
        local_t = self.local_proj(local_features)
        global_t = self.global_proj(global_features)

        # Gated combination
        fused = gate * local_t + (1 - gate) * global_t

        # Output projection
        out = self.out_proj(fused)

        return out


class HierarchicalGlobalContext(nn.Module):
    """
    Hierarchical Global Context Module.

    This module provides global context awareness through:
    1. Level 1 adaptive pooling
    2. Small SSM processing on pooled features
    3. Level 2 adaptive pooling
    4. Multi-head attention back to original sequence
    5. Gated fusion with local features

    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        attention_dim: Attention dimension
        global_ssm_layers: Number of SSM layers for global processing
        global_ssm_state_dim: State dimension for global SSM
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 192,
        num_heads: int = 4,
        attention_dim: int = 48,
        global_ssm_layers: int = 2,
        global_ssm_state_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Level 1 pooling
        self.pool1 = AdaptivePool(level=1, d_model=d_model)

        # Global SSM for processing pooled features
        self.global_ssm = GlobalSSM(
            d_model=d_model,
            num_layers=global_ssm_layers,
            state_dim=global_ssm_state_dim,
            dropout=dropout,
        )

        # Level 2 pooling
        self.pool2 = AdaptivePool(level=2, d_model=d_model)

        # Cross-attention from original sequence to global features
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Gated fusion
        self.fusion = GatedFusion(d_model=d_model)

    def forward(
        self,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of hierarchical global context module.

        Args:
            local_features: Output from local SSM processor (batch, seq_len, d_model)

        Returns:
            Fused features with global context (batch, seq_len, d_model)
        """
        # Level 1 pooling
        x_pool1, pool_size1 = self.pool1(local_features)

        # Process with global SSM
        x_ssm = self.global_ssm(x_pool1)

        # Level 2 pooling
        x_pool2, _ = self.pool2(x_ssm, prev_pool_size=pool_size1)

        # Normalize for attention
        x_pool2 = self.norm1(x_pool2)
        query = self.norm2(local_features)

        # Cross-attention: original sequence attends to pooled global features
        global_context = self.cross_attention(
            query=query,
            key=x_pool2,
            value=x_pool2,
        )

        # Fuse local and global features
        output = self.fusion(local_features, global_context)

        return output
