"""
Data loading utilities for VELOCITY-ASR v2.

Provides Dataset and DataLoader classes for LibriSpeech and other ASR datasets.
"""

import json
import os
import random
from typing import Optional, List, Dict, Any, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .audio import load_audio, compute_mel_spectrogram, SAMPLE_RATE


class ASRDataset(Dataset):
    """
    ASR Dataset that reads from JSON manifest files.

    Manifest format (JSON lines):
    {"audio_path": "...", "text": "...", "duration": ..., ...}

    Args:
        manifest_path: Path to manifest file (JSON lines)
        tokenizer: Optional tokenizer for text (if None, uses character-level)
        max_duration: Maximum audio duration in seconds (None = no limit)
        min_duration: Minimum audio duration in seconds
        sample_rate: Target sample rate
        normalize_audio: Whether to normalize audio
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: Optional[Any] = None,
        max_duration: Optional[float] = 30.0,
        min_duration: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
        normalize_audio: bool = True,
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

        # Load manifest
        self.samples = self._load_manifest()

        # Build character vocabulary if no tokenizer provided
        if self.tokenizer is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = None

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load and filter manifest entries."""
        samples = []

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                entry = json.loads(line)

                # Filter by duration
                duration = entry.get('duration', 0)
                if duration < self.min_duration:
                    continue
                if self.max_duration and duration > self.max_duration:
                    continue

                # Check audio file exists
                if not os.path.exists(entry['audio_path']):
                    continue

                samples.append(entry)

        return samples

    def _build_vocab(self) -> Dict[str, int]:
        """Build character vocabulary from manifest."""
        chars = set()
        for sample in self.samples:
            chars.update(sample['text'])

        # Sort for reproducibility
        chars = sorted(chars)

        # Build vocab with special tokens
        vocab = {
            '<blank>': 0,
            '<unk>': 1,
            '<pad>': 2,
        }

        for i, char in enumerate(chars):
            vocab[char] = i + 3

        return vocab

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)

        # Character-level tokenization
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])
        return tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio = load_audio(sample['audio_path'], sample_rate=self.sample_rate)

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(audio, normalize=self.normalize_audio)

        # Tokenize text
        tokens = self.text_to_tokens(sample['text'])

        return {
            'mel_spectrogram': mel,  # (frames, mel_bins)
            'targets': torch.tensor(tokens, dtype=torch.long),
            'input_lengths': torch.tensor(mel.shape[0], dtype=torch.long),
            'target_lengths': torch.tensor(len(tokens), dtype=torch.long),
            'text': sample['text'],
        }


class ASRCollator:
    """
    Collator for batching ASR samples with padding.

    Args:
        pad_token_id: Token ID for padding targets
        mel_pad_value: Value for padding mel spectrograms
    """

    def __init__(
        self,
        pad_token_id: int = 2,  # <pad>
        mel_pad_value: float = 0.0,
    ):
        self.pad_token_id = pad_token_id
        self.mel_pad_value = mel_pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        # Get max lengths
        max_mel_len = max(item['mel_spectrogram'].shape[0] for item in batch)
        max_target_len = max(item['targets'].shape[0] for item in batch)

        mel_batch = []
        target_batch = []
        input_lengths = []
        target_lengths = []
        texts = []

        for item in batch:
            # Pad mel spectrogram
            mel = item['mel_spectrogram']
            pad_len = max_mel_len - mel.shape[0]
            if pad_len > 0:
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, pad_len), value=self.mel_pad_value
                )
            mel_batch.append(mel)

            # Pad targets
            targets = item['targets']
            pad_len = max_target_len - targets.shape[0]
            if pad_len > 0:
                targets = torch.nn.functional.pad(
                    targets, (0, pad_len), value=self.pad_token_id
                )
            target_batch.append(targets)

            input_lengths.append(item['input_lengths'])
            target_lengths.append(item['target_lengths'])
            texts.append(item['text'])

        return {
            'mel_spectrogram': torch.stack(mel_batch),
            'targets': torch.stack(target_batch),
            'input_lengths': torch.stack(input_lengths),
            'target_lengths': torch.stack(target_lengths),
            'texts': texts,
        }


def create_dataloader(
    manifest_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_duration: Optional[float] = 30.0,
    min_duration: float = 0.5,
    tokenizer: Optional[Any] = None,
) -> Tuple[DataLoader, ASRDataset]:
    """
    Create a DataLoader for ASR training/evaluation.

    Args:
        manifest_path: Path to manifest file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        max_duration: Maximum audio duration
        min_duration: Minimum audio duration
        tokenizer: Optional tokenizer

    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = ASRDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        max_duration=max_duration,
        min_duration=min_duration,
    )

    collator = ASRCollator()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=shuffle,  # Drop last incomplete batch during training
    )

    return dataloader, dataset


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset using torchaudio directly (no manifest needed).

    Args:
        root: Root directory containing LibriSpeech data
        split: Dataset split (e.g., "train-clean-100", "dev-clean")
        tokenizer: Optional tokenizer
        max_duration: Maximum audio duration in seconds
        download: Whether to download if not present
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train-clean-100",
        tokenizer: Optional[Any] = None,
        max_duration: Optional[float] = 30.0,
        download: bool = False,
    ):
        try:
            from torchaudio.datasets import LIBRISPEECH
        except ImportError:
            raise ImportError("torchaudio is required for LibriSpeechDataset")

        self.dataset = LIBRISPEECH(
            root=root,
            url=split,
            download=download,
        )
        self.tokenizer = tokenizer
        self.max_duration = max_duration

        # Build character vocabulary if no tokenizer
        if self.tokenizer is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = None

    def _build_vocab(self) -> Dict[str, int]:
        """Build character vocabulary."""
        # Standard English characters + space + common punctuation
        chars = list(" abcdefghijklmnopqrstuvwxyz'")

        vocab = {
            '<blank>': 0,
            '<unk>': 1,
            '<pad>': 2,
        }

        for i, char in enumerate(chars):
            vocab[char] = i + 3

        return vocab

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)

        # Character-level tokenization
        text = text.lower()
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])
        return tokens

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        # Check duration
        duration = waveform.shape[0] / SAMPLE_RATE
        if self.max_duration and duration > self.max_duration:
            # Truncate
            max_samples = int(self.max_duration * SAMPLE_RATE)
            waveform = waveform[:max_samples]

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(waveform, normalize=True)

        # Tokenize text
        tokens = self.text_to_tokens(transcript)

        return {
            'mel_spectrogram': mel,
            'targets': torch.tensor(tokens, dtype=torch.long),
            'input_lengths': torch.tensor(mel.shape[0], dtype=torch.long),
            'target_lengths': torch.tensor(len(tokens), dtype=torch.long),
            'text': transcript.lower(),
        }


def create_librispeech_dataloaders(
    root: str = "./data",
    train_splits: List[str] = ["train-clean-100"],
    val_splits: List[str] = ["dev-clean"],
    batch_size: int = 8,
    num_workers: int = 4,
    max_duration: float = 30.0,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation DataLoaders for LibriSpeech.

    Args:
        root: Data root directory
        train_splits: List of training splits
        val_splits: List of validation splits
        batch_size: Batch size
        num_workers: Number of workers
        max_duration: Maximum audio duration
        download: Whether to download data

    Returns:
        Tuple of (train_loader, val_loader, vocab)
    """
    from torch.utils.data import ConcatDataset

    # Create training datasets
    train_datasets = []
    for split in train_splits:
        ds = LibriSpeechDataset(
            root=root,
            split=split,
            max_duration=max_duration,
            download=download,
        )
        train_datasets.append(ds)

    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        vocab = train_datasets[0].vocab  # Use first dataset's vocab
    else:
        train_dataset = train_datasets[0]
        vocab = train_dataset.vocab

    # Create validation datasets
    val_datasets = []
    for split in val_splits:
        ds = LibriSpeechDataset(
            root=root,
            split=split,
            max_duration=max_duration,
            download=download,
        )
        ds.vocab = vocab  # Share vocabulary
        val_datasets.append(ds)

    if len(val_datasets) > 1:
        val_dataset = ConcatDataset(val_datasets)
    else:
        val_dataset = val_datasets[0]

    # Create collator
    collator = ASRCollator()

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    return train_loader, val_loader, vocab
