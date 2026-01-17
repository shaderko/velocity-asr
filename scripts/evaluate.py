#!/usr/bin/env python3
"""
Evaluation script for VELOCITY-ASR v2.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-set librispeech_test_clean
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --audio-dir ./test_audio
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from velocity_asr import (
    VELOCITYASR,
    load_audio,
    compute_mel_spectrogram,
    ctc_greedy_decode,
    CTCDecoder,
    create_default_vocabulary,
)
from velocity_asr.training import compute_wer, compute_cer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_test_data(test_set: str) -> List[Tuple[str, str]]:
    """
    Load test dataset.

    Args:
        test_set: Name or path of test set

    Returns:
        List of (audio_path, reference_text) tuples
    """
    # TODO: Implement actual dataset loading
    # For now, return empty list (placeholder)
    logger.warning(
        f"Dataset loading not implemented for '{test_set}'. "
        "Please implement actual data loading."
    )
    return []


def evaluate_directory(
    model: VELOCITYASR,
    audio_dir: str,
    decoder: CTCDecoder,
    device: str,
) -> List[Tuple[str, str]]:
    """
    Transcribe all audio files in a directory.

    Args:
        model: VELOCITY-ASR model
        audio_dir: Directory containing audio files
        decoder: CTC decoder
        device: Device to run inference on

    Returns:
        List of (filename, transcription) tuples
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []

    for path in Path(audio_dir).rglob('*'):
        if path.suffix.lower() in audio_extensions:
            audio_files.append(path)

    results = []
    model.eval()

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        try:
            # Load and preprocess audio
            audio = load_audio(str(audio_path))
            mel = compute_mel_spectrogram(audio)

            # Run inference
            with torch.no_grad():
                mel_tensor = mel.unsqueeze(0).to(device)
                logits = model(mel_tensor)
                transcription = decoder.decode_greedy(logits)[0]

            results.append((audio_path.name, transcription))

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            results.append((audio_path.name, f"[ERROR: {e}]"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VELOCITY-ASR v2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Test set name (e.g., librispeech_test_clean)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory containing audio files to transcribe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width for decoding (1 = greedy)",
    )
    args = parser.parse_args()

    if args.test_set is None and args.audio_dir is None:
        parser.error("Either --test-set or --audio-dir must be specified")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VELOCITYASR.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()

    logger.info(f"Model loaded with {model.count_parameters():,} parameters")

    # Create decoder
    vocabulary = create_default_vocabulary(model.config.vocab_size)
    decoder = CTCDecoder(vocabulary)

    if args.audio_dir:
        # Transcribe directory
        results = evaluate_directory(model, args.audio_dir, decoder, args.device)

        # Print results
        print("\n" + "=" * 60)
        print("TRANSCRIPTION RESULTS")
        print("=" * 60)
        for filename, transcription in results:
            print(f"\n{filename}:")
            print(f"  {transcription}")

        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                for filename, transcription in results:
                    f.write(f"{filename}\t{transcription}\n")
            logger.info(f"Results saved to {args.output}")

    elif args.test_set:
        # Evaluate on benchmark
        test_data = load_test_data(args.test_set)

        if not test_data:
            logger.error("No test data loaded. Exiting.")
            return

        predictions = []
        references = []

        for audio_path, reference in tqdm(test_data, desc="Evaluating"):
            try:
                audio = load_audio(audio_path)
                mel = compute_mel_spectrogram(audio)

                with torch.no_grad():
                    mel_tensor = mel.unsqueeze(0).to(args.device)
                    logits = model(mel_tensor)
                    transcription = decoder.decode_greedy(logits)[0]

                predictions.append(transcription)
                references.append(reference)

            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")

        # Compute metrics
        wer = compute_wer(predictions, references)
        cer = compute_cer(predictions, references)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Set: {args.test_set}")
        print(f"Samples: {len(predictions)}")
        print(f"WER: {wer * 100:.2f}%")
        print(f"CER: {cer * 100:.2f}%")
        print("=" * 60)

        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"Test Set: {args.test_set}\n")
                f.write(f"Samples: {len(predictions)}\n")
                f.write(f"WER: {wer * 100:.2f}%\n")
                f.write(f"CER: {cer * 100:.2f}%\n")
            logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
