"""
Training utilities for VELOCITY-ASR v2.

This module provides training loop components, loss computation,
and optimization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 80000
    grad_clip_norm: float = 1.0

    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


class CTCLoss(nn.Module):
    """
    CTC Loss wrapper with proper handling of input/target lengths.

    Args:
        blank_token: Index of blank token (default: 0)
        reduction: Loss reduction method ('mean', 'sum', 'none')
        zero_infinity: Whether to zero infinite losses
    """

    def __init__(
        self,
        blank_token: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ):
        super().__init__()
        self.blank_token = blank_token
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target sequences (batch, max_target_len)
            input_lengths: Length of each input sequence (batch,)
            target_lengths: Length of each target sequence (batch,)

        Returns:
            CTC loss value
        """
        # CTC expects log probabilities of shape (T, N, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, N, C)

        # Flatten targets for CTC
        targets_flat = targets[targets != self.blank_token]

        loss = self.ctc_loss(
            log_probs,
            targets_flat if targets.dim() == 1 else targets,
            input_lengths,
            target_lengths,
        )

        return loss


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr_mult = self._get_lr_multiplier()

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr * lr_mult

    def _get_lr_multiplier(self) -> float:
        """Compute learning rate multiplier."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / self.warmup_steps

        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay.item()

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class Trainer:
    """
    Training loop handler for VELOCITY-ASR.

    Args:
        model: VELOCITY-ASR model
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device

        # Loss function
        self.criterion = CTCLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp and device == 'cuda' else None

        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Dictionary containing:
                - mel_spectrogram: (batch, frames, mel_bins)
                - targets: (batch, max_target_len)
                - input_lengths: (batch,)
                - target_lengths: (batch,)

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        mel = batch['mel_spectrogram'].to(self.device)
        targets = batch['targets'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        target_lengths = batch['target_lengths'].to(self.device)

        # Compute output lengths after temporal binding (stride 2)
        output_lengths = (input_lengths + 1) // 2

        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.config.use_amp and self.device == 'cuda'):
            logits = self.model(mel)
            loss = self.criterion(logits, targets, output_lengths, target_lengths)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {'loss': loss.item() * self.config.gradient_accumulation_steps}

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

        metrics['lr'] = self.scheduler.get_lr()[0]
        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute single evaluation step.

        Args:
            batch: Same format as train_step

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        mel = batch['mel_spectrogram'].to(self.device)
        targets = batch['targets'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        target_lengths = batch['target_lengths'].to(self.device)

        output_lengths = (input_lengths + 1) // 2

        logits = self.model(mel)
        loss = self.criterion(logits, targets, output_lengths, target_lengths)

        return {'eval_loss': loss.item()}

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Training history
        """
        import os

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        history = {'train_loss': [], 'eval_loss': [], 'lr': []}
        running_loss = 0.0

        data_iter = iter(self.train_dataloader)

        for step in range(self.config.max_steps):
            self.global_step = step

            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Training step
            metrics = self.train_step(batch)
            running_loss += metrics['loss']

            # Logging
            if (step + 1) % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                logger.info(
                    f"Step {step + 1}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {metrics['lr']:.6f}"
                )
                history['train_loss'].append(avg_loss)
                history['lr'].append(metrics['lr'])
                running_loss = 0.0

            # Evaluation
            if self.eval_dataloader and (step + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                history['eval_loss'].append(eval_metrics['eval_loss'])
                logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")

                if eval_metrics['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['eval_loss']
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, "best_model.pt")
                    )

            # Checkpointing
            if (step + 1) % self.config.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.config.checkpoint_dir,
                        f"checkpoint_step_{step + 1}.pt"
                    )
                )

        return history

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            metrics = self.eval_step(batch)
            total_loss += metrics['eval_loss']
            num_batches += 1

        return {'eval_loss': total_loss / num_batches}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__,
        }

        if hasattr(self.model, 'config'):
            checkpoint['model_config'] = self.model.config.__dict__

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.global_step = checkpoint['global_step']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))

        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


def compute_wer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute Word Error Rate.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        Word Error Rate
    """
    total_errors = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()

        # Dynamic programming for edit distance
        d = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]

        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j

        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if pred_words[i - 1] == ref_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(
                        d[i - 1][j],      # Deletion
                        d[i][j - 1],      # Insertion
                        d[i - 1][j - 1],  # Substitution
                    )

        total_errors += d[len(pred_words)][len(ref_words)]
        total_words += len(ref_words)

    return total_errors / total_words if total_words > 0 else 0.0


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute Character Error Rate.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        Character Error Rate
    """
    total_errors = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.lower())
        ref_chars = list(ref.lower())

        # Dynamic programming for edit distance
        d = [[0] * (len(ref_chars) + 1) for _ in range(len(pred_chars) + 1)]

        for i in range(len(pred_chars) + 1):
            d[i][0] = i
        for j in range(len(ref_chars) + 1):
            d[0][j] = j

        for i in range(1, len(pred_chars) + 1):
            for j in range(1, len(ref_chars) + 1):
                if pred_chars[i - 1] == ref_chars[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(
                        d[i - 1][j],
                        d[i][j - 1],
                        d[i - 1][j - 1],
                    )

        total_errors += d[len(pred_chars)][len(ref_chars)]
        total_chars += len(ref_chars)

    return total_errors / total_chars if total_chars > 0 else 0.0
