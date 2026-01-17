"""
VELOCITY-ASR v2: Edge-Optimized Speech Recognition

A lightweight ASR architecture combining selective state space models
with hierarchical global context for efficient edge deployment.

Example usage:
    >>> from velocity_asr import VELOCITYASR, load_audio, compute_mel_spectrogram
    >>>
    >>> # Load model
    >>> model = VELOCITYASR()
    >>> model.eval()
    >>>
    >>> # Process audio
    >>> audio = load_audio("speech.wav")
    >>> mel = compute_mel_spectrogram(audio)
    >>>
    >>> # Transcribe
    >>> with torch.no_grad():
    ...     logits = model(mel.unsqueeze(0))
    ...     transcription = ctc_greedy_decode(logits)
"""

__version__ = "2.0.0"
__author__ = "VELOCITY Research Team"

from .model import (
    VELOCITYASR,
    VelocityASRConfig,
    TemporalBindingLayer,
    CTCOutputHead,
)

from .ssm import (
    SelectiveSSM,
    SSMBlock,
    LocalSSMProcessor,
    GlobalSSM,
)

from .attention import (
    HierarchicalGlobalContext,
    AdaptivePool,
    MultiHeadAttention,
    GatedFusion,
)

from .audio import (
    load_audio,
    compute_mel_spectrogram,
    MelSpectrogramTransform,
    audio_to_frames,
    frames_to_audio,
    pad_or_trim,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)

from .decode import (
    ctc_greedy_decode,
    ctc_greedy_decode_with_timestamps,
    ctc_beam_search,
    CTCDecoder,
    DecodingResult,
    create_default_vocabulary,
)

# Convenience function for loading pretrained models
def from_pretrained(model_name_or_path: str, **kwargs) -> VELOCITYASR:
    """
    Load a pretrained VELOCITY-ASR model.

    Args:
        model_name_or_path: Model name or path to checkpoint
        **kwargs: Additional arguments passed to VELOCITYASR.from_pretrained

    Returns:
        VELOCITYASR model instance
    """
    return VELOCITYASR.from_pretrained(model_name_or_path, **kwargs)


__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Main model
    "VELOCITYASR",
    "VelocityASRConfig",
    "from_pretrained",

    # Model components
    "TemporalBindingLayer",
    "CTCOutputHead",
    "SelectiveSSM",
    "SSMBlock",
    "LocalSSMProcessor",
    "GlobalSSM",
    "HierarchicalGlobalContext",
    "AdaptivePool",
    "MultiHeadAttention",
    "GatedFusion",

    # Audio processing
    "load_audio",
    "compute_mel_spectrogram",
    "MelSpectrogramTransform",
    "audio_to_frames",
    "frames_to_audio",
    "pad_or_trim",
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",

    # Decoding
    "ctc_greedy_decode",
    "ctc_greedy_decode_with_timestamps",
    "ctc_beam_search",
    "CTCDecoder",
    "DecodingResult",
    "create_default_vocabulary",
]
