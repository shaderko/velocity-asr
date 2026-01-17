"""
Selective State Space Model (SSM) components for VELOCITY-ASR v2.

This module implements Mamba-style selective SSM blocks that form the
core local acoustic processor of the architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with input-dependent parameters.

    Unlike fixed SSMs, the A, B, C parameters are computed dynamically
    based on the input, allowing content-dependent processing.

    Args:
        d_model: Input/output dimension
        state_dim: SSM state dimension (N)
        expand_ratio: Expansion ratio for inner dimension
    """

    def __init__(
        self,
        d_model: int = 192,
        state_dim: int = 64,
        expand_ratio: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.d_inner = d_model * expand_ratio

        # Input projection to expanded dimension
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameter projections (input-dependent)
        # B and C are projected from expanded dimension
        self.x_proj = nn.Linear(self.d_inner, state_dim * 2, bias=False)

        # Delta (timestep) projection
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize A as a diagonal matrix (log-spaced)
        # A is fixed but transformed via softplus(dt) to be input-dependent
        A = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # D is a skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of selective SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Compute input-dependent B and C
        x_dbl = self.x_proj(x_proj)  # (B, L, 2*state_dim)
        B, C = x_dbl.chunk(2, dim=-1)  # Each (B, L, state_dim)

        # Compute discretization step
        dt = F.softplus(self.dt_proj(x_proj))  # (B, L, d_inner)

        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (state_dim,)

        # Run selective scan
        y = self._selective_scan(x_proj, dt, A, B, C)

        # Apply gating and output projection
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient selective scan algorithm.

        Uses pre-allocated output tensor and avoids storing all intermediate
        states simultaneously to reduce memory footprint.

        Args:
            x: Input tensor (B, L, d_inner)
            dt: Discretization timestep (B, L, d_inner)
            A: State transition diagonal (state_dim,)
            B: Input projection (B, L, state_dim)
            C: Output projection (B, L, state_dim)

        Returns:
            Output tensor (B, L, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        state_dim = A.shape[0]

        # Pre-allocate output tensor (more memory efficient than list + stack)
        y = torch.empty_like(x)

        # State shape: (B, d_inner, state_dim)
        h = torch.zeros(batch, d_inner, state_dim, device=x.device, dtype=x.dtype)

        # A is reshaped once for broadcasting
        A_expanded = A.view(1, 1, -1)  # (1, 1, state_dim)

        # Process sequence step by step (memory efficient)
        for t in range(seq_len):
            dt_t = dt[:, t]  # (B, d_inner)
            B_t = B[:, t]    # (B, state_dim)
            C_t = C[:, t]    # (B, state_dim)
            x_t = x[:, t]    # (B, d_inner)

            # Compute dA and dB for this timestep only
            dA_t = torch.exp(dt_t.unsqueeze(-1) * A_expanded)  # (B, d_inner, state_dim)
            dB_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)       # (B, d_inner, state_dim)

            # Update state: h = dA * h + x * dB
            h = dA_t * h + x_t.unsqueeze(-1) * dB_t

            # Output: y = C @ h (einsum is memory efficient)
            y[:, t] = torch.einsum('bdn,bn->bd', h, C_t)

        # Add skip connection
        y = y + x * self.D

        return y


class SSMBlock(nn.Module):
    """
    Single SSM block with depthwise convolution and feed-forward network.

    Architecture:
        1. Depthwise 1D convolution for local context
        2. Selective SSM for sequence modeling
        3. Feed-forward network for additional capacity

    Args:
        d_model: Feature dimension
        state_dim: SSM state dimension
        expand_ratio: FFN expansion ratio
        kernel_size: Depthwise convolution kernel size
        dropout: Dropout probability
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        d_model: int = 192,
        state_dim: int = 64,
        expand_ratio: int = 2,
        kernel_size: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Depthwise convolution for local context
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,  # Depthwise
        )

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=d_model,
            state_dim=state_dim,
            expand_ratio=expand_ratio,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand_ratio, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation."""
        # Pre-norm + Conv + SSM + residual
        residual = x
        x = self.norm1(x)

        # Apply depthwise convolution
        x_conv = x.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :x.size(1)]  # Remove padding (causal)
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)

        # Apply SSM
        x = self.ssm(x_conv)
        x = self.dropout(x)
        x = x + residual

        # Pre-norm + FFN + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SSM block.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class LocalSSMProcessor(nn.Module):
    """
    Local SSM processor with stacked SSM blocks.

    This is the main workhorse of VELOCITY-ASR, processing acoustic
    features with linear complexity using selective state space models.

    Args:
        d_model: Feature dimension
        num_layers: Number of SSM blocks
        state_dim: SSM state dimension
        expand_ratio: Expansion ratio for SSM and FFN
        kernel_size: Depthwise convolution kernel size
        dropout: Dropout probability
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 8,
        state_dim: int = 64,
        expand_ratio: int = 2,
        kernel_size: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                state_dim=state_dim,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all SSM blocks.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class GlobalSSM(nn.Module):
    """
    Lightweight SSM for processing pooled global features.

    Uses smaller state dimension and fewer layers than local processor.

    Args:
        d_model: Feature dimension
        num_layers: Number of SSM blocks (default: 2)
        state_dim: SSM state dimension (default: 32)
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 2,
        state_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                state_dim=state_dim,
                expand_ratio=2,
                kernel_size=4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through global SSM blocks.

        Args:
            x: Pooled input tensor (batch, pool_len, d_model)

        Returns:
            Output tensor (batch, pool_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x
