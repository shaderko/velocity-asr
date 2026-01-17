#!/usr/bin/env python3
"""
ONNX export script for VELOCITY-ASR v2.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output velocity-asr.onnx
    python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output velocity-asr-int8.onnx --quantize
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from velocity_asr import VELOCITYASR
from velocity_asr.quantize import (
    export_quantized_onnx,
    quantize_onnx_model,
    get_model_size_mb,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def verify_onnx_model(onnx_path: str, input_shape: tuple = (1, 500, 80)):
    """
    Verify ONNX model by running inference.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape for verification
    """
    try:
        import onnxruntime as ort
        import numpy as np

        logger.info(f"Verifying ONNX model: {onnx_path}")

        # Create inference session
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider'],
        )

        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        logger.info(f"  Input: {input_info.name}, shape: {input_info.shape}")
        logger.info(f"  Output: {output_info.name}, shape: {output_info.shape}")

        # Run inference with dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = session.run(None, {input_info.name: dummy_input})

        logger.info(f"  Output shape: {outputs[0].shape}")
        logger.info("  Verification successful!")

        return True

    except Exception as e:
        logger.error(f"  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export VELOCITY-ASR to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="velocity-asr.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export",
    )
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VELOCITYASR.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()

    # Log model info
    num_params = model.count_parameters()
    model_size = get_model_size_mb(model)
    logger.info(f"Model loaded: {num_params:,} parameters, {model_size:.2f} MB")

    # Determine output paths
    output_path = args.output
    if args.quantize and not output_path.endswith('-int8.onnx'):
        base, ext = os.path.splitext(output_path)
        fp32_path = f"{base}-fp32{ext}"
        int8_path = f"{base}-int8{ext}"
    else:
        fp32_path = output_path
        int8_path = output_path.replace('.onnx', '-int8.onnx') if args.quantize else None

    # Export to ONNX
    logger.info(f"Exporting to ONNX: {fp32_path}")
    export_quantized_onnx(
        model,
        fp32_path,
        input_shape=(1, 500, model.config.mel_bins),
        opset_version=args.opset,
    )

    # Get FP32 model size
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    logger.info(f"FP32 ONNX model size: {fp32_size:.2f} MB")

    # Verify FP32 model
    if args.verify:
        verify_onnx_model(fp32_path)

    # Apply quantization if requested
    if args.quantize:
        logger.info(f"Applying INT8 quantization: {int8_path}")
        quantize_onnx_model(fp32_path, int8_path)

        # Get INT8 model size
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        logger.info(f"INT8 ONNX model size: {int8_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")

        # Verify INT8 model
        if args.verify:
            verify_onnx_model(int8_path)

    logger.info("Export complete!")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Model parameters: {num_params:,}")
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.2f} MB)")
    if args.quantize:
        print(f"INT8 ONNX: {int8_path} ({int8_size:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
