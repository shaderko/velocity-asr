"""
Quantization utilities for VELOCITY-ASR v2.

This module provides Quantization-Aware Training (QAT) and
INT8 export utilities for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    # Bit widths
    weight_bits: int = 8
    activation_bits: int = 8

    # Per-channel quantization for weights
    per_channel_weights: bool = True

    # Keep SSM state in FP32 to avoid error accumulation
    ssm_state_fp32: bool = True

    # Calibration settings
    num_calibration_batches: int = 100

    # Quantization scheme
    symmetric_weights: bool = True
    symmetric_activations: bool = False


class FakeQuantize(nn.Module):
    """
    Fake quantization module for QAT.

    Simulates quantization during forward pass while maintaining
    gradients for backpropagation.

    Args:
        bits: Number of quantization bits
        symmetric: Whether to use symmetric quantization
        per_channel: Whether to use per-channel quantization
        channel_dim: Dimension for per-channel quantization
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        # Learnable scale and zero point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('calibrated', torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization."""
        if not self.training and not self.calibrated:
            # During inference without calibration, pass through
            return x

        # Compute scale and zero point if needed
        if self.training or not self.calibrated:
            self._update_scale_zp(x)

        # Quantize
        x_q = self._quantize(x)

        # Dequantize (for fake quantization)
        x_dq = self._dequantize(x_q)

        # Straight-through estimator for gradient
        return x + (x_dq - x).detach()

    def _update_scale_zp(self, x: torch.Tensor):
        """Update scale and zero point based on input statistics."""
        if self.per_channel:
            # Per-channel statistics
            dims = list(range(x.dim()))
            dims.remove(self.channel_dim)
            x_min = x.amin(dim=dims, keepdim=True)
            x_max = x.amax(dim=dims, keepdim=True)
        else:
            x_min = x.min()
            x_max = x.max()

        if self.symmetric:
            # Symmetric quantization
            x_abs_max = torch.maximum(x_min.abs(), x_max.abs())
            self.scale = x_abs_max / self.qmax
            self.zero_point = torch.zeros_like(self.scale)
        else:
            # Asymmetric quantization
            self.scale = (x_max - x_min) / (self.qmax - self.qmin)
            self.zero_point = self.qmin - x_min / self.scale

        self.scale = torch.clamp(self.scale, min=1e-10)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor."""
        return torch.clamp(
            torch.round(x / self.scale + self.zero_point),
            self.qmin,
            self.qmax,
        )

    def _dequantize(self, x_q: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor."""
        return (x_q - self.zero_point) * self.scale

    def calibrate(self, x: torch.Tensor):
        """Calibrate quantization parameters."""
        with torch.no_grad():
            self._update_scale_zp(x)
            self.calibrated.fill_(True)


class QuantizedLinear(nn.Module):
    """
    Linear layer with fake quantization for QAT.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        config: Quantization configuration
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        config = config or QuantizationConfig()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Weight quantizer
        self.weight_quantizer = FakeQuantize(
            bits=config.weight_bits,
            symmetric=config.symmetric_weights,
            per_channel=config.per_channel_weights,
            channel_dim=0,
        )

        # Activation quantizer
        self.activation_quantizer = FakeQuantize(
            bits=config.activation_bits,
            symmetric=config.symmetric_activations,
            per_channel=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        # Quantize weights
        weight_q = self.weight_quantizer(self.linear.weight)

        # Apply linear operation
        output = F.linear(x, weight_q, self.linear.bias)

        # Quantize activations
        output = self.activation_quantizer(output)

        return output


class QuantizedConv1d(nn.Module):
    """
    1D convolution with fake quantization for QAT.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        groups: Number of groups
        bias: Whether to include bias
        config: Quantization configuration
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        config = config or QuantizationConfig()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        # Weight quantizer
        self.weight_quantizer = FakeQuantize(
            bits=config.weight_bits,
            symmetric=config.symmetric_weights,
            per_channel=config.per_channel_weights,
            channel_dim=0,
        )

        # Activation quantizer
        self.activation_quantizer = FakeQuantize(
            bits=config.activation_bits,
            symmetric=config.symmetric_activations,
            per_channel=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        # Quantize weights
        weight_q = self.weight_quantizer(self.conv.weight)

        # Apply convolution
        output = F.conv1d(
            x,
            weight_q,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )

        # Quantize activations
        output = self.activation_quantizer(output)

        return output


def prepare_model_for_qat(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training.

    Replaces Linear and Conv1d layers with quantized versions.
    Note: SSM state computations are kept in FP32.

    Args:
        model: Original model
        config: Quantization configuration

    Returns:
        Model prepared for QAT
    """
    config = config or QuantizationConfig()

    def replace_module(module: nn.Module, name: str = "") -> nn.Module:
        """Recursively replace modules with quantized versions."""
        # Skip SSM internal computations if configured
        if config.ssm_state_fp32 and "ssm" in name.lower():
            # Keep SSM computations in FP32
            return module

        if isinstance(module, nn.Linear):
            return QuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                config=config,
            )

        if isinstance(module, nn.Conv1d):
            return QuantizedConv1d(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                groups=module.groups,
                bias=module.bias is not None,
                config=config,
            )

        # Recursively process children
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            setattr(module, child_name, replace_module(child, full_name))

        return module

    return replace_module(model)


def calibrate_model(
    model: nn.Module,
    calibration_dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    device: str = "cuda",
):
    """
    Calibrate quantization parameters using representative data.

    Args:
        model: QAT-prepared model
        calibration_dataloader: Data loader for calibration
        num_batches: Number of calibration batches
        device: Device to run calibration on
    """
    model.eval()
    model.to(device)

    # Collect all fake quantize modules
    fake_quant_modules = []
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            fake_quant_modules.append(module)

    logger.info(f"Calibrating {len(fake_quant_modules)} quantization nodes...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_dataloader):
            if batch_idx >= num_batches:
                break

            if isinstance(batch, dict):
                mel = batch['mel_spectrogram'].to(device)
            else:
                mel = batch[0].to(device)

            # Forward pass to update calibration statistics
            _ = model(mel)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Calibration progress: {batch_idx + 1}/{num_batches}")

    # Mark all fake quantize modules as calibrated
    for module in fake_quant_modules:
        module.calibrated.fill_(True)

    logger.info("Calibration complete.")


def export_quantized_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int] = (1, 500, 80),
    opset_version: int = 17,
):
    """
    Export quantized model to ONNX format.

    Args:
        model: Trained model (optionally QAT-prepared)
        output_path: Path for ONNX output
        input_shape: Input tensor shape (batch, frames, mel_bins)
        opset_version: ONNX opset version
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['logits'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch', 1: 'frames'},
            'logits': {0: 'batch', 1: 'output_frames'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Exported ONNX model to {output_path}")


def quantize_onnx_model(
    onnx_path: str,
    output_path: str,
    calibration_dataloader: Optional[torch.utils.data.DataLoader] = None,
):
    """
    Apply INT8 quantization to ONNX model.

    Args:
        onnx_path: Path to FP32 ONNX model
        output_path: Path for quantized ONNX model
        calibration_dataloader: Optional data loader for static quantization
    """
    try:
        import onnxruntime.quantization as ort_quant
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX quantization. "
            "Install with: pip install onnxruntime"
        )

    if calibration_dataloader is None:
        # Dynamic quantization (no calibration data needed)
        ort_quant.quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=ort_quant.QuantType.QInt8,
        )
    else:
        # Static quantization with calibration
        class CalibrationDataReader(ort_quant.CalibrationDataReader):
            def __init__(self, dataloader):
                self.dataloader = iter(dataloader)
                self.count = 0

            def get_next(self):
                try:
                    if self.count >= 100:
                        return None
                    batch = next(self.dataloader)
                    if isinstance(batch, dict):
                        mel = batch['mel_spectrogram'].numpy()
                    else:
                        mel = batch[0].numpy()
                    self.count += 1
                    return {'mel_spectrogram': mel}
                except StopIteration:
                    return None

        calibration_reader = CalibrationDataReader(calibration_dataloader)

        ort_quant.quantize_static(
            onnx_path,
            output_path,
            calibration_reader,
            quant_format=ort_quant.QuantFormat.QDQ,
            per_channel=True,
            weight_type=ort_quant.QuantType.QInt8,
            activation_type=ort_quant.QuantType.QInt8,
        )

    logger.info(f"Exported quantized ONNX model to {output_path}")


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / (1024 * 1024)
