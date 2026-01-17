#!/usr/bin/env python3
"""
Training script for VELOCITY-ASR v2.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --resume checkpoints/last.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from velocity_asr import VELOCITYASR, VelocityASRConfig
from velocity_asr.training import Trainer, TrainingConfig
from velocity_asr.quantize import prepare_model_for_qat, QuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dummy_dataloader(batch_size: int, num_batches: int = 100):
    """
    Create a dummy dataloader for testing.

    Replace this with actual data loading for production training.
    """
    from torch.utils.data import DataLoader, Dataset

    class DummyASRDataset(Dataset):
        def __init__(self, num_samples: int = 1000):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random mel spectrogram (100-500 frames)
            num_frames = torch.randint(100, 500, (1,)).item()
            mel = torch.randn(num_frames, 80)

            # Random target sequence (10-50 tokens)
            target_len = torch.randint(10, 50, (1,)).item()
            targets = torch.randint(1, 100, (target_len,))  # Avoid blank (0)

            return {
                'mel_spectrogram': mel,
                'targets': targets,
                'input_lengths': torch.tensor(num_frames),
                'target_lengths': torch.tensor(target_len),
            }

    def collate_fn(batch):
        """Collate batch with padding."""
        max_mel_len = max(item['mel_spectrogram'].size(0) for item in batch)
        max_target_len = max(item['targets'].size(0) for item in batch)

        mel_batch = []
        target_batch = []
        input_lengths = []
        target_lengths = []

        for item in batch:
            # Pad mel spectrogram
            mel = item['mel_spectrogram']
            pad_len = max_mel_len - mel.size(0)
            mel_padded = torch.nn.functional.pad(mel, (0, 0, 0, pad_len))
            mel_batch.append(mel_padded)

            # Pad targets
            targets = item['targets']
            pad_len = max_target_len - targets.size(0)
            targets_padded = torch.nn.functional.pad(targets, (0, pad_len), value=0)
            target_batch.append(targets_padded)

            input_lengths.append(item['input_lengths'])
            target_lengths.append(item['target_lengths'])

        return {
            'mel_spectrogram': torch.stack(mel_batch),
            'targets': torch.stack(target_batch),
            'input_lengths': torch.stack(input_lengths),
            'target_lengths': torch.stack(target_lengths),
        }

    dataset = DummyASRDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Train VELOCITY-ASR v2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cpu, mps)",
    )
    args = parser.parse_args()

    # Load configurations
    train_config = load_config(args.config) if os.path.exists(args.config) else {}
    model_config = load_config(args.model_config) if os.path.exists(args.model_config) else {}

    # Determine device
    device = args.device
    if device is None:
        device = train_config.get('hardware', {}).get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logger.warning("CUDA not available, falling back to CPU")

    logger.info(f"Using device: {device}")

    # Create model config
    model_cfg = VelocityASRConfig(
        mel_bins=model_config.get('input', {}).get('mel_bins', 80),
        d_model=model_config.get('model', {}).get('d_model', 192),
        ssm_layers=model_config.get('ssm', {}).get('num_layers', 8),
        ssm_state_dim=model_config.get('ssm', {}).get('state_dim', 64),
        ssm_expand_ratio=model_config.get('ssm', {}).get('expand_ratio', 2),
        ssm_kernel_size=model_config.get('ssm', {}).get('kernel_size', 4),
        global_ssm_layers=model_config.get('global_context', {}).get('ssm_layers', 2),
        global_ssm_state_dim=model_config.get('global_context', {}).get('ssm_state_dim', 32),
        attention_heads=model_config.get('global_context', {}).get('attention_heads', 4),
        attention_dim=model_config.get('global_context', {}).get('attention_dim', 48),
        vocab_size=model_config.get('model', {}).get('vocab_size', 1000),
        dropout=model_config.get('model', {}).get('dropout', 0.1),
        gradient_checkpointing=model_config.get('memory', {}).get('gradient_checkpointing', False),
    )

    # Create model
    model = VELOCITYASR(model_cfg)
    logger.info(f"Model created with {model.count_parameters():,} parameters")

    # Apply QAT if configured
    quant_config = train_config.get('quantization', {})
    if quant_config.get('enabled', False):
        logger.info("Preparing model for Quantization-Aware Training")
        qat_config = QuantizationConfig(
            weight_bits=quant_config.get('weight_bits', 8),
            activation_bits=quant_config.get('activation_bits', 8),
            per_channel_weights=quant_config.get('per_channel_weights', True),
            ssm_state_fp32=quant_config.get('ssm_state_fp32', True),
        )
        model = prepare_model_for_qat(model, qat_config)

    # Create training config
    opt_config = train_config.get('optimization', {})
    training_cfg = TrainingConfig(
        learning_rate=opt_config.get('learning_rate', 1e-4),
        weight_decay=opt_config.get('weight_decay', 0.01),
        warmup_steps=opt_config.get('warmup_steps', 10000),
        max_steps=opt_config.get('total_steps', 80000),
        grad_clip_norm=opt_config.get('grad_clip_norm', 1.0),
        batch_size=opt_config.get('batch_size', 32),
        gradient_accumulation_steps=opt_config.get('gradient_accumulation_steps', 1),
        use_amp=train_config.get('precision', {}).get('use_amp', True),
        log_interval=train_config.get('logging', {}).get('log_interval', 100),
        eval_interval=train_config.get('logging', {}).get('eval_interval', 1000),
        save_interval=train_config.get('checkpoint', {}).get('save_interval', 5000),
        checkpoint_dir=train_config.get('checkpoint', {}).get('dir', './checkpoints'),
    )

    # Create data loaders
    # TODO: Replace with actual data loading
    logger.warning("Using dummy data loader - replace with actual data for training")
    train_dataloader = create_dummy_dataloader(training_cfg.batch_size)
    eval_dataloader = create_dummy_dataloader(training_cfg.batch_size, num_batches=10)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Save final checkpoint
    trainer.save_checkpoint(
        os.path.join(training_cfg.checkpoint_dir, "final_model.pt")
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
