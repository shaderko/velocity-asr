"""
Audio processing utilities for VELOCITY-ASR v2.

This module handles audio loading and mel spectrogram computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import numpy as np


# Default audio parameters
SAMPLE_RATE = 16000
N_FFT = 400  # 25ms at 16kHz
HOP_LENGTH = 160  # 10ms at 16kHz
N_MELS = 80
WINDOW_FN = torch.hann_window


def load_audio(
    path: str,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (default: 16000)
        mono: Convert to mono (default: True)

    Returns:
        Audio tensor of shape (samples,) for mono or (channels, samples) for stereo
    """
    try:
        import torchaudio
    except ImportError:
        raise ImportError(
            "torchaudio is required for audio loading. "
            "Install with: pip install torchaudio"
        )

    # Load audio
    waveform, sr = torchaudio.load(path)

    # Convert to mono if needed
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Remove channel dimension for mono
    if mono:
        waveform = waveform.squeeze(0)

    return waveform


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute mel spectrogram from audio waveform.

    Args:
        audio: Audio tensor of shape (samples,) or (batch, samples)
        sample_rate: Audio sample rate
        n_fft: FFT window size (default: 400 = 25ms at 16kHz)
        hop_length: Hop length (default: 160 = 10ms at 16kHz)
        n_mels: Number of mel bins (default: 80)
        normalize: Whether to normalize the spectrogram

    Returns:
        Mel spectrogram of shape (frames, n_mels) or (batch, frames, n_mels)
    """
    # Handle batched input
    squeeze_output = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True

    batch_size = audio.size(0)
    device = audio.device

    # Compute STFT
    window = WINDOW_FN(n_fft).to(device)

    # Pad audio for STFT
    pad_length = n_fft // 2
    audio_padded = F.pad(audio, (pad_length, pad_length), mode='reflect')

    # STFT
    stft = torch.stft(
        audio_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=False,
    )

    # Compute magnitude spectrogram
    magnitudes = stft.abs() ** 2

    # Create mel filterbank
    mel_filters = _create_mel_filterbank(
        n_fft=n_fft,
        n_mels=n_mels,
        sample_rate=sample_rate,
        device=device,
    )

    # Apply mel filterbank
    mel_spec = torch.matmul(mel_filters, magnitudes)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-10)

    # Normalize if requested
    if normalize:
        mel_spec = (mel_spec - mel_spec.mean(dim=-1, keepdim=True)) / (
            mel_spec.std(dim=-1, keepdim=True) + 1e-10
        )

    # Transpose to (batch, frames, n_mels)
    mel_spec = mel_spec.transpose(1, 2)

    if squeeze_output:
        mel_spec = mel_spec.squeeze(0)

    return mel_spec


def _create_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create mel filterbank matrix.

    Args:
        n_fft: FFT window size
        n_mels: Number of mel bins
        sample_rate: Audio sample rate
        device: Target device

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    # Frequency bins
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs, device=device)

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Mel frequencies
    mel_min = hz_to_mel(torch.tensor(0.0))
    mel_max = hz_to_mel(torch.tensor(sample_rate / 2.0))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)

    # Create filterbank
    filterbank = torch.zeros(n_mels, n_freqs, device=device)

    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        # Rising slope
        lower_slope = (freqs - lower) / (center - lower + 1e-10)
        # Falling slope
        upper_slope = (upper - freqs) / (upper - center + 1e-10)

        filterbank[i] = torch.maximum(
            torch.zeros_like(freqs),
            torch.minimum(lower_slope, upper_slope),
        )

    return filterbank


class MelSpectrogramTransform(nn.Module):
    """
    PyTorch module for mel spectrogram computation.

    Useful for including spectrogram computation in the model graph
    for ONNX export.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins
        normalize: Whether to normalize output
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize

        # Register window as buffer
        self.register_buffer('window', WINDOW_FN(n_fft))

        # Pre-compute mel filterbank
        mel_filters = _create_mel_filterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            device=torch.device('cpu'),
        )
        self.register_buffer('mel_filters', mel_filters)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.

        Args:
            audio: Audio tensor (samples,) or (batch, samples)

        Returns:
            Mel spectrogram (frames, n_mels) or (batch, frames, n_mels)
        """
        return compute_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalize=self.normalize,
        )


def audio_to_frames(
    audio_length: int,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> int:
    """
    Calculate number of mel frames from audio length.

    Args:
        audio_length: Number of audio samples
        hop_length: Hop length
        n_fft: FFT window size

    Returns:
        Number of mel frames
    """
    return (audio_length + n_fft) // hop_length


def frames_to_audio(
    num_frames: int,
    hop_length: int = HOP_LENGTH,
) -> int:
    """
    Calculate approximate audio length from number of frames.

    Args:
        num_frames: Number of mel frames
        hop_length: Hop length

    Returns:
        Approximate number of audio samples
    """
    return num_frames * hop_length


def pad_or_trim(
    audio: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """
    Pad or trim audio to target length.

    Args:
        audio: Audio tensor (samples,) or (batch, samples)
        target_length: Target number of samples

    Returns:
        Audio tensor of target length
    """
    current_length = audio.shape[-1]

    if current_length > target_length:
        # Trim
        audio = audio[..., :target_length]
    elif current_length < target_length:
        # Pad with zeros
        pad_length = target_length - current_length
        audio = F.pad(audio, (0, pad_length))

    return audio
