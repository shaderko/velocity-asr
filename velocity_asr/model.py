"""
Main VELOCITY-ASR v2 Model Architecture.

This module implements the complete VELOCITY-ASR model including:
- Temporal Binding Layer
- Local SSM Processor
- Hierarchical Global Context
- CTC Output Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .ssm import LocalSSMProcessor
from .attention import HierarchicalGlobalContext


@dataclass
class VelocityASRConfig:
    """Configuration for VELOCITY-ASR model."""

    # Input dimensions
    mel_bins: int = 80

    # Model dimensions
    d_model: int = 192

    # SSM configuration
    ssm_layers: int = 8
    ssm_state_dim: int = 64
    ssm_expand_ratio: int = 2
    ssm_kernel_size: int = 4

    # Global context configuration
    global_ssm_layers: int = 2
    global_ssm_state_dim: int = 32
    attention_heads: int = 4
    attention_dim: int = 48

    # Output configuration
    # NOTE: 50k vocab creates 9.6M params in CTC head alone!
    # Use 1000 for character-level, 5000 for small subword
    vocab_size: int = 1000

    # Regularization
    dropout: float = 0.1

    # Memory optimization
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VelocityASRConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spectro-temporal features.

    Separately encodes time and frequency dimensions, providing
    richer inductive biases for speech processing.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        mel_bins: Number of mel frequency bins
    """

    def __init__(
        self,
        d_model: int = 192,
        max_len: int = 5000,
        mel_bins: int = 80,
    ):
        super().__init__()
        self.d_model = d_model

        # Temporal positional encoding
        pe_time = torch.zeros(max_len, d_model // 2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2))
        )
        pe_time[:, 0::2] = torch.sin(position * div_term)
        pe_time[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_time', pe_time)

        # Frequency positional encoding (learnable)
        self.pe_freq = nn.Parameter(torch.randn(1, 1, d_model // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)

        # Temporal encoding
        time_enc = self.pe_time[:seq_len].unsqueeze(0)  # (1, L, D/2)

        # Frequency encoding (broadcast across sequence)
        freq_enc = self.pe_freq.expand(-1, seq_len, -1)  # (1, L, D/2)

        # Concatenate time and frequency encodings
        pos_enc = torch.cat([time_enc, freq_enc], dim=-1)  # (1, L, D)

        return x + pos_enc


class TemporalBindingLayer(nn.Module):
    """
    Temporal Binding Layer for converting mel spectrograms to embeddings.

    Performs:
    1. 1D convolution with stride 2 (halves sequence length)
    2. 2D positional encoding
    3. Layer normalization

    Args:
        mel_bins: Number of mel frequency bins (input channels)
        d_model: Output dimension
        kernel_size: Convolution kernel size
        stride: Convolution stride
    """

    def __init__(
        self,
        mel_bins: int = 80,
        d_model: int = 192,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()

        # 1D convolution for temporal reduction and feature projection
        self.conv = nn.Conv1d(
            in_channels=mel_bins,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(
            d_model=d_model,
            mel_bins=mel_bins,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.GELU()

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to sequence of embeddings.

        Args:
            mel_spectrogram: Input tensor (batch, frames, mel_bins)

        Returns:
            Embedding tensor (batch, frames/2, d_model)
        """
        # Transpose for Conv1d: (B, mel_bins, frames)
        x = mel_spectrogram.transpose(1, 2)

        # Apply convolution
        x = self.conv(x)
        x = self.activation(x)

        # Transpose back: (B, frames/2, d_model)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Layer normalization
        x = self.norm(x)

        return x


class CTCOutputHead(nn.Module):
    """
    CTC Output Head for generating token logits.

    Args:
        d_model: Input dimension
        vocab_size: Vocabulary size (including blank token)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 50000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate CTC logits.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Logits tensor (batch, seq_len, vocab_size)
        """
        return self.proj(x)


class VELOCITYASR(nn.Module):
    """
    VELOCITY-ASR v2 Model.

    Edge-optimized speech recognition architecture combining:
    - Selective State Space Models for efficient local processing
    - Hierarchical global context for linguistic awareness
    - CTC output for transcription

    Args:
        config: VelocityASRConfig instance or None for defaults
    """

    def __init__(self, config: Optional[VelocityASRConfig] = None):
        super().__init__()

        if config is None:
            config = VelocityASRConfig()

        self.config = config

        # Temporal binding layer
        self.temporal_binding = TemporalBindingLayer(
            mel_bins=config.mel_bins,
            d_model=config.d_model,
        )

        # Local SSM processor
        self.local_ssm = LocalSSMProcessor(
            d_model=config.d_model,
            num_layers=config.ssm_layers,
            state_dim=config.ssm_state_dim,
            expand_ratio=config.ssm_expand_ratio,
            kernel_size=config.ssm_kernel_size,
            dropout=config.dropout,
            use_checkpoint=config.gradient_checkpointing,
        )

        # Hierarchical global context
        self.global_context = HierarchicalGlobalContext(
            d_model=config.d_model,
            num_heads=config.attention_heads,
            attention_dim=config.attention_dim,
            global_ssm_layers=config.global_ssm_layers,
            global_ssm_state_dim=config.global_ssm_state_dim,
            dropout=config.dropout,
        )

        # CTC output head
        self.ctc_head = CTCOutputHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of VELOCITY-ASR.

        Args:
            mel_spectrogram: Input mel spectrogram (batch, frames, mel_bins)
            return_features: If True, return intermediate features

        Returns:
            CTC logits (batch, frames/2, vocab_size)
            Or tuple of (logits, features) if return_features=True
        """
        # Temporal binding: (B, T, 80) -> (B, T/2, 192)
        x = self.temporal_binding(mel_spectrogram)

        # Local SSM processing
        local_features = self.local_ssm(x)

        # Hierarchical global context + fusion
        fused_features = self.global_context(local_features)

        # CTC output
        logits = self.ctc_head(fused_features)

        if return_features:
            return logits, {
                'temporal_binding': x,
                'local_features': local_features,
                'fused_features': fused_features,
            }

        return logits

    def get_output_length(self, input_length: int) -> int:
        """
        Compute output sequence length given input length.

        The temporal binding layer halves the sequence length.

        Args:
            input_length: Input sequence length (mel frames)

        Returns:
            Output sequence length
        """
        # Conv with stride 2 approximately halves length
        return (input_length + 1) // 2

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quantized: bool = False,
        **kwargs,
    ) -> "VELOCITYASR":
        """
        Load a pretrained VELOCITY-ASR model.

        Args:
            model_name_or_path: Model name or path to checkpoint
            quantized: Whether to load quantized weights
            **kwargs: Additional arguments

        Returns:
            VELOCITYASR model instance
        """
        import os

        # Check if it's a local path
        if os.path.exists(model_name_or_path):
            checkpoint_path = model_name_or_path
        else:
            # TODO: Implement model hub download
            raise NotImplementedError(
                f"Model hub download not yet implemented. "
                f"Please provide a local path to the checkpoint."
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config
        if 'config' in checkpoint:
            config = VelocityASRConfig.from_dict(checkpoint['config'])
        else:
            config = VelocityASRConfig()

        # Create model
        model = cls(config)

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def save_pretrained(self, save_path: str):
        """
        Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'config': {
                'mel_bins': self.config.mel_bins,
                'd_model': self.config.d_model,
                'ssm_layers': self.config.ssm_layers,
                'ssm_state_dim': self.config.ssm_state_dim,
                'ssm_expand_ratio': self.config.ssm_expand_ratio,
                'ssm_kernel_size': self.config.ssm_kernel_size,
                'global_ssm_layers': self.config.global_ssm_layers,
                'global_ssm_state_dim': self.config.global_ssm_state_dim,
                'attention_heads': self.config.attention_heads,
                'attention_dim': self.config.attention_dim,
                'vocab_size': self.config.vocab_size,
                'dropout': self.config.dropout,
                'gradient_checkpointing': self.config.gradient_checkpointing,
            },
            'model_state_dict': self.state_dict(),
        }

        torch.save(checkpoint, save_path)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
