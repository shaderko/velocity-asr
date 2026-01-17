#!/usr/bin/env python3
"""
CLI transcription script for VELOCITY-ASR v2.

Usage:
    python scripts/transcribe.py audio.wav --checkpoint checkpoints/best_model.pt
    python scripts/transcribe.py --input-dir ./audio --output-dir ./transcripts
    velocity-asr transcribe audio.wav --model velocity-asr-v2
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from velocity_asr import (
    VELOCITYASR,
    load_audio,
    compute_mel_spectrogram,
    CTCDecoder,
    create_default_vocabulary,
    ctc_greedy_decode_with_timestamps,
)
from velocity_asr.audio import SAMPLE_RATE, HOP_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def frames_to_seconds(frame_idx: int, hop_length: int = HOP_LENGTH, sample_rate: int = SAMPLE_RATE) -> float:
    """Convert frame index to seconds."""
    # Account for temporal binding stride of 2
    return (frame_idx * 2 * hop_length) / sample_rate


def transcribe_file(
    model: VELOCITYASR,
    audio_path: str,
    decoder: CTCDecoder,
    device: str,
    include_timestamps: bool = False,
) -> dict:
    """
    Transcribe a single audio file.

    Args:
        model: VELOCITY-ASR model
        audio_path: Path to audio file
        decoder: CTC decoder
        device: Device to run inference on
        include_timestamps: Whether to include word timestamps

    Returns:
        Dictionary with transcription results
    """
    # Load audio
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio)

    # Run inference
    with torch.no_grad():
        mel_tensor = mel.unsqueeze(0).to(device)
        logits = model(mel_tensor)

        if include_timestamps:
            # Decode with timestamps
            results = ctc_greedy_decode_with_timestamps(logits)
            tokens, timestamps = results[0]

            # Convert to text with word timestamps
            words = []
            current_word = []
            current_start = None

            for token, (start_frame, end_frame) in zip(tokens, timestamps):
                char = decoder.vocabulary[token] if 0 <= token < len(decoder.vocabulary) else "<unk>"

                if char == " " or char == "▁":
                    if current_word:
                        word_text = "".join(current_word).replace("▁", "")
                        if word_text:
                            words.append({
                                "word": word_text,
                                "start": frames_to_seconds(current_start),
                                "end": frames_to_seconds(end_frame),
                            })
                        current_word = []
                        current_start = None
                else:
                    if current_start is None:
                        current_start = start_frame
                    current_word.append(char)

            # Handle last word
            if current_word and timestamps:
                word_text = "".join(current_word).replace("▁", "")
                if word_text:
                    words.append({
                        "word": word_text,
                        "start": frames_to_seconds(current_start),
                        "end": frames_to_seconds(timestamps[-1][1]),
                    })

            transcription = " ".join(w["word"] for w in words)

            return {
                "file": audio_path,
                "duration": duration,
                "transcription": transcription,
                "words": words,
            }
        else:
            # Simple transcription
            transcription = decoder.decode_greedy(logits)[0]

            return {
                "file": audio_path,
                "duration": duration,
                "transcription": transcription,
            }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with VELOCITY-ASR v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Transcribe a single file
    python scripts/transcribe.py audio.wav --checkpoint model.pt

    # Transcribe with timestamps
    python scripts/transcribe.py audio.wav --checkpoint model.pt --timestamps

    # Batch transcribe a directory
    python scripts/transcribe.py --input-dir ./audio --output-dir ./transcripts --checkpoint model.pt

    # Output as JSON
    python scripts/transcribe.py audio.wav --checkpoint model.pt --format json
        """,
    )
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        default=None,
        help="Path to audio file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing audio files for batch transcription",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include word-level timestamps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress logging output",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.audio is None and args.input_dir is None:
        parser.error("Either audio file or --input-dir must be specified")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VELOCITYASR.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()

    # Create decoder
    vocabulary = create_default_vocabulary(model.config.vocab_size)
    decoder = CTCDecoder(vocabulary)

    if args.input_dir:
        # Batch transcription
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = []

        for path in Path(args.input_dir).rglob('*'):
            if path.suffix.lower() in audio_extensions:
                audio_files.append(path)

        if not audio_files:
            logger.error(f"No audio files found in {args.input_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files")

        # Create output directory
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        results = []
        for audio_path in audio_files:
            try:
                result = transcribe_file(
                    model,
                    str(audio_path),
                    decoder,
                    args.device,
                    include_timestamps=args.timestamps,
                )
                results.append(result)

                # Save individual transcript if output_dir specified
                if args.output_dir:
                    output_name = audio_path.stem + (".json" if args.format == "json" else ".txt")
                    output_path = Path(args.output_dir) / output_name

                    if args.format == "json":
                        with open(output_path, 'w') as f:
                            json.dump(result, f, indent=2)
                    else:
                        with open(output_path, 'w') as f:
                            f.write(result["transcription"])

                if not args.quiet:
                    print(f"\n{audio_path.name}:")
                    print(f"  {result['transcription']}")

            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")

        # Save combined results if output specified
        if args.output:
            if args.format == "json":
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                with open(args.output, 'w') as f:
                    for result in results:
                        f.write(f"{result['file']}\t{result['transcription']}\n")

        logger.info(f"Processed {len(results)} files")

    else:
        # Single file transcription
        try:
            result = transcribe_file(
                model,
                args.audio,
                decoder,
                args.device,
                include_timestamps=args.timestamps,
            )

            if args.format == "json":
                output = json.dumps(result, indent=2)
            else:
                output = result["transcription"]

            # Output to file or stdout
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                logger.info(f"Transcript saved to {args.output}")
            else:
                print(output)

        except Exception as e:
            logger.error(f"Error processing {args.audio}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
