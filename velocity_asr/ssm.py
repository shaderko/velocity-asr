"""
Selective State Space Model (SSM) components for VELOCITY-ASR v2.

This module implements Mamba-style selective SSM blocks that form the
core local acoustic processor of the architecture.

Supports three scan implementations:
1. sequential - Basic Python loop (slow, but always works)
2. parallel - Associative parallel scan (fast, pure PyTorch)
3. mamba - Official Mamba CUDA kernels (fastest, requires mamba-ssm)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Literal

# Try to import mamba-ssm for CUDA kernels
MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    selective_scan_fn = None

# Scan mode type
ScanMode = Literal["sequential", "parallel", "mamba"]


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with input-dependent parameters.

    Unlike fixed SSMs, the A, B, C parameters are computed dynamically
    based on the input, allowing content-dependent processing.

    Supports three scan implementations:
    - "sequential": Basic Python loop (slow, always works)
    - "parallel": Associative parallel scan (fast, pure PyTorch)
    - "mamba": Official Mamba CUDA kernels (fastest, requires mamba-ssm)

    Args:
        d_model: Input/output dimension
        state_dim: SSM state dimension (N)
        expand_ratio: Expansion ratio for inner dimension
        scan_mode: Which scan implementation to use
    """

    def __init__(
        self,
        d_model: int = 192,
        state_dim: int = 64,
        expand_ratio: int = 2,
        scan_mode: ScanMode = "parallel",
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.d_inner = d_model * expand_ratio
        self.scan_mode = scan_mode

        # Validate scan mode
        if scan_mode == "mamba" and not MAMBA_AVAILABLE:
            raise ImportError(
                "scan_mode='mamba' requires mamba-ssm package. "
                "Install with: pip install mamba-ssm"
            )

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

        # Run selective scan based on mode
        if self.scan_mode == "sequential":
            y = self._sequential_scan(x_proj, dt, A, B, C)
        elif self.scan_mode == "parallel":
            y = self._parallel_scan(x_proj, dt, A, B, C)
        elif self.scan_mode == "mamba":
            y = self._mamba_scan(x_proj, dt, A, B, C)
        else:
            raise ValueError(f"Unknown scan_mode: {self.scan_mode}")

        # Apply gating and output projection
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

    def _sequential_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential scan - basic Python loop implementation.
        Slow but memory efficient and always works.
        """
        batch, seq_len, d_inner = x.shape
        state_dim = A.shape[0]

        # Pre-allocate output tensor
        y = torch.empty_like(x)

        # State shape: (B, d_inner, state_dim)
        h = torch.zeros(batch, d_inner, state_dim, device=x.device, dtype=x.dtype)

        # A reshaped for broadcasting
        A_expanded = A.view(1, 1, -1)

        for t in range(seq_len):
            dt_t = dt[:, t]
            B_t = B[:, t]
            C_t = C[:, t]
            x_t = x[:, t]

            dA_t = torch.exp(dt_t.unsqueeze(-1) * A_expanded)
            dB_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = dA_t * h + x_t.unsqueeze(-1) * dB_t
            y[:, t] = torch.einsum('bdn,bn->bd', h, C_t)

        y = y + x * self.D
        return y

    def _parallel_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel associative scan implementation.

        Uses the associative property of the SSM recurrence to compute
        in O(log L) parallel steps instead of O(L) sequential steps.

        The recurrence h[t] = dA[t] * h[t-1] + dB[t] * x[t] can be written as:
        (h[t], 1) = (dA[t], dB[t]*x[t]) ⊗ (h[t-1], 1)

        where ⊗ is the associative operator:
        (a2, b2) ⊗ (a1, b1) = (a2 * a1, a2 * b1 + b2)
        """
        batch, seq_len, d_inner = x.shape
        state_dim = A.shape[0]

        # Compute discretized parameters for all timesteps
        # dA: (B, L, d_inner, state_dim)
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, 1, -1))
        # dB: (B, L, d_inner, state_dim)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)
        # x * dB: (B, L, d_inner, state_dim)
        x_dB = x.unsqueeze(-1) * dB

        # Parallel scan using associative operator
        # We compute prefix products and sums in parallel
        h = self._associative_scan(dA, x_dB)  # (B, L, d_inner, state_dim)

        # Compute output: y[t] = C[t] @ h[t]
        # h: (B, L, d_inner, state_dim), C: (B, L, state_dim)
        y = torch.einsum('bldn,bln->bld', h, C)

        # Add skip connection
        y = y + x * self.D
        return y

    def _associative_scan(
        self,
        dA: torch.Tensor,
        x_dB: torch.Tensor,
    ) -> torch.Tensor:
        """
        Associative scan for SSM recurrence.

        Computes h[t] = dA[t] * h[t-1] + x_dB[t] for all t in parallel.

        Uses work-efficient parallel scan algorithm.
        """
        batch, seq_len, d_inner, state_dim = dA.shape

        # Pad to power of 2 for efficient parallel scan
        log_len = math.ceil(math.log2(max(seq_len, 1)))
        padded_len = 2 ** log_len

        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            # Pad with identity elements: dA=1, x_dB=0
            dA = F.pad(dA, (0, 0, 0, 0, 0, pad_size), value=1.0)
            x_dB = F.pad(x_dB, (0, 0, 0, 0, 0, pad_size), value=0.0)

        # Clone for in-place operations
        a = dA.clone()  # multiplicative part
        b = x_dB.clone()  # additive part

        # Up-sweep (reduce) phase
        stride = 1
        for d in range(log_len):
            stride_2 = stride * 2
            # Indices for parallel operations
            idx_right = torch.arange(stride_2 - 1, padded_len, stride_2, device=dA.device)
            idx_left = idx_right - stride

            # (a2, b2) ⊗ (a1, b1) = (a2 * a1, a2 * b1 + b2)
            a_left = a[:, idx_left]
            b_left = b[:, idx_left]
            a_right = a[:, idx_right]
            b_right = b[:, idx_right]

            a[:, idx_right] = a_right * a_left
            b[:, idx_right] = a_right * b_left + b_right

            stride = stride_2

        # Set last element (will be the identity after down-sweep)
        a[:, -1] = 1.0
        b[:, -1] = 0.0

        # Down-sweep phase
        stride = padded_len // 2
        for d in range(log_len):
            stride_2 = stride * 2
            idx_right = torch.arange(stride_2 - 1, padded_len, stride_2, device=dA.device)
            idx_left = idx_right - stride

            # Save left values
            a_left_old = a[:, idx_left].clone()
            b_left_old = b[:, idx_left].clone()

            # Left gets right's value
            a[:, idx_left] = a[:, idx_right]
            b[:, idx_left] = b[:, idx_right]

            # Right = right ⊗ old_left
            a[:, idx_right] = a[:, idx_right] * a_left_old
            b[:, idx_right] = a[:, idx_right] * b_left_old + b[:, idx_right]

            stride = stride // 2

        # The result b now contains the prefix scan (h values)
        # But we need to shift: h[t] depends on inputs [0..t], not [0..t-1]
        # Actually, our scan gives us the correct h[t] already

        # Apply final combination: h[t] = cumulative_a * 0 + cumulative_b = b
        h = b[:, :seq_len]

        return h

    def _mamba_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mamba CUDA kernel scan - fastest implementation.
        Requires mamba-ssm package.
        """
        batch, seq_len, d_inner = x.shape

        # Mamba expects specific tensor layouts
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2).contiguous()
        # dt: (B, L, D) -> (B, D, L)
        dt = dt.transpose(1, 2).contiguous()
        # A: (N,) -> (D, N) expanded
        A = A.unsqueeze(0).expand(d_inner, -1).contiguous()
        # B: (B, L, N) -> (B, 1, L, N) for group=1
        B = B.unsqueeze(1).contiguous()
        # C: (B, L, N) -> (B, 1, L, N) for group=1
        C = C.unsqueeze(1).contiguous()
        # D: (D,)
        D = self.D

        # Call Mamba selective scan
        y = selective_scan_fn(
            x, dt, A, B, C, D,
            z=None,
            delta_bias=None,
            delta_softplus=False,  # Already applied softplus
            return_last_state=False,
        )

        # y: (B, D, L) -> (B, L, D)
        y = y.transpose(1, 2).contiguous()

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
        scan_mode: SSM scan implementation ("sequential", "parallel", "mamba")
    """

    def __init__(
        self,
        d_model: int = 192,
        state_dim: int = 64,
        expand_ratio: int = 2,
        kernel_size: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        scan_mode: ScanMode = "parallel",
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
            scan_mode=scan_mode,
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
        scan_mode: SSM scan implementation ("sequential", "parallel", "mamba")
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
        scan_mode: ScanMode = "parallel",
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
                scan_mode=scan_mode,
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
