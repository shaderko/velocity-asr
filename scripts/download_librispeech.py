#!/usr/bin/env python3
"""
Download LibriSpeech dataset using torchaudio.

Usage:
    # Download train-clean-100 and validation sets
    python scripts/download_librispeech.py --train 100

    # Download train-clean-360 and validation sets
    python scripts/download_librispeech.py --train 360

    # Download train-other-500 and validation sets
    python scripts/download_librispeech.py --train 500

    # Download multiple training sets
    python scripts/download_librispeech.py --train 100 360

    # Download all training sets
    python scripts/download_librispeech.py --train 100 360 500

    # Also download test sets
    python scripts/download_librispeech.py --train 100 --include-test

    # Specify custom data directory
    python scripts/download_librispeech.py --train 100 --data-dir /path/to/data

    # Generate manifest files for training
    python scripts/download_librispeech.py --train 100 --create-manifests
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# LibriSpeech split mappings
TRAIN_SPLITS = {
    100: "train-clean-100",
    360: "train-clean-360",
    500: "train-other-500",
}

VAL_SPLITS = ["dev-clean", "dev-other"]
TEST_SPLITS = ["test-clean", "test-other"]


def download_split(
    data_dir: str,
    split: str,
    show_progress: bool = True,
) -> LIBRISPEECH:
    """
    Download a single LibriSpeech split.

    Args:
        data_dir: Root directory for data
        split: Split name (e.g., "train-clean-100")
        show_progress: Whether to show download progress

    Returns:
        LIBRISPEECH dataset object
    """
    logger.info(f"Downloading/loading split: {split}")

    try:
        dataset = LIBRISPEECH(
            root=data_dir,
            url=split,
            download=True,
        )
        logger.info(f"  ✓ {split}: {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"  ✗ Failed to download {split}: {e}")
        raise


def create_manifest(
    dataset: LIBRISPEECH,
    output_path: str,
    split_name: str,
) -> int:
    """
    Create a JSON manifest file from a LibriSpeech dataset.

    Manifest format (JSON lines):
    {"audio_path": "...", "text": "...", "duration": ..., "speaker_id": ..., "chapter_id": ...}

    Args:
        dataset: LIBRISPEECH dataset object
        output_path: Path to output manifest file
        split_name: Name of the split for logging

    Returns:
        Number of samples written
    """
    logger.info(f"Creating manifest for {split_name}: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[i]

            # Get audio file path
            # LibriSpeech structure: {root}/LibriSpeech/{split}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.flac
            audio_path = os.path.join(
                dataset._path,
                str(speaker_id),
                str(chapter_id),
                f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
            )

            # Calculate duration
            duration = waveform.shape[1] / sample_rate

            entry = {
                "audio_path": audio_path,
                "text": transcript.lower(),  # Lowercase for training
                "duration": round(duration, 3),
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "sample_rate": sample_rate,
            }

            f.write(json.dumps(entry) + '\n')
            count += 1

    logger.info(f"  ✓ Written {count} entries to {output_path}")
    return count


def get_dataset_stats(dataset: LIBRISPEECH) -> dict:
    """
    Compute basic statistics for a dataset.

    Args:
        dataset: LIBRISPEECH dataset object

    Returns:
        Dictionary with statistics
    """
    total_duration = 0.0
    num_speakers = set()

    for i in range(min(len(dataset), 1000)):  # Sample first 1000 for speed
        waveform, sample_rate, _, speaker_id, _, _ = dataset[i]
        total_duration += waveform.shape[1] / sample_rate
        num_speakers.add(speaker_id)

    # Extrapolate if we sampled
    if len(dataset) > 1000:
        total_duration = total_duration * len(dataset) / 1000

    return {
        "num_samples": len(dataset),
        "estimated_hours": round(total_duration / 3600, 1),
        "num_speakers_sampled": len(num_speakers),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 100-hour training set + validation
    python scripts/download_librispeech.py --train 100

    # Download all training data (960 hours total)
    python scripts/download_librispeech.py --train 100 360 500

    # Download with manifest generation
    python scripts/download_librispeech.py --train 100 --create-manifests
        """,
    )
    parser.add_argument(
        "--train",
        type=int,
        nargs="+",
        choices=[100, 360, 500],
        required=True,
        help="Training splits to download (100=train-clean-100, 360=train-clean-360, 500=train-other-500)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for downloaded data (default: ./data)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download test sets (test-clean, test-other)",
    )
    parser.add_argument(
        "--create-manifests",
        action="store_true",
        help="Create JSON manifest files for training",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="./manifests",
        help="Directory for manifest files (default: ./manifests)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip downloading validation sets",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    manifest_dir = os.path.abspath(args.manifest_dir)

    logger.info("=" * 60)
    logger.info("LibriSpeech Download Script")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Training splits: {args.train}")

    # Collect all splits to download
    splits_to_download = []

    # Training splits
    for hours in args.train:
        splits_to_download.append(TRAIN_SPLITS[hours])

    # Validation splits (always included unless skipped)
    if not args.skip_validation:
        splits_to_download.extend(VAL_SPLITS)

    # Test splits (optional)
    if args.include_test:
        splits_to_download.extend(TEST_SPLITS)

    logger.info(f"Splits to download: {splits_to_download}")
    logger.info("=" * 60)

    # Download each split
    datasets = {}
    for split in splits_to_download:
        try:
            datasets[split] = download_split(data_dir, split)
        except Exception as e:
            logger.error(f"Failed to download {split}: {e}")
            continue

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    total_samples = 0
    total_hours = 0.0

    for split, dataset in datasets.items():
        stats = get_dataset_stats(dataset)
        total_samples += stats["num_samples"]
        total_hours += stats["estimated_hours"]
        logger.info(
            f"  {split}: {stats['num_samples']} samples, "
            f"~{stats['estimated_hours']} hours"
        )

    logger.info("-" * 60)
    logger.info(f"  Total: {total_samples} samples, ~{total_hours:.1f} hours")
    logger.info("=" * 60)

    # Create manifests if requested
    if args.create_manifests:
        logger.info("")
        logger.info("Creating manifest files...")
        os.makedirs(manifest_dir, exist_ok=True)

        # Group training manifests
        train_entries = []
        for hours in args.train:
            split = TRAIN_SPLITS[hours]
            if split in datasets:
                manifest_path = os.path.join(manifest_dir, f"{split}.jsonl")
                create_manifest(datasets[split], manifest_path, split)

        # Validation manifests
        if not args.skip_validation:
            for split in VAL_SPLITS:
                if split in datasets:
                    manifest_path = os.path.join(manifest_dir, f"{split}.jsonl")
                    create_manifest(datasets[split], manifest_path, split)

            # Combined validation manifest
            combined_val_path = os.path.join(manifest_dir, "dev-all.jsonl")
            logger.info(f"Creating combined validation manifest: {combined_val_path}")
            with open(combined_val_path, 'w') as out_f:
                for split in VAL_SPLITS:
                    split_path = os.path.join(manifest_dir, f"{split}.jsonl")
                    if os.path.exists(split_path):
                        with open(split_path, 'r') as in_f:
                            out_f.write(in_f.read())

        # Test manifests
        if args.include_test:
            for split in TEST_SPLITS:
                if split in datasets:
                    manifest_path = os.path.join(manifest_dir, f"{split}.jsonl")
                    create_manifest(datasets[split], manifest_path, split)

        # Create combined training manifest
        combined_train_path = os.path.join(manifest_dir, "train-all.jsonl")
        logger.info(f"Creating combined training manifest: {combined_train_path}")
        with open(combined_train_path, 'w') as out_f:
            for hours in args.train:
                split = TRAIN_SPLITS[hours]
                split_path = os.path.join(manifest_dir, f"{split}.jsonl")
                if os.path.exists(split_path):
                    with open(split_path, 'r') as in_f:
                        out_f.write(in_f.read())

        logger.info("")
        logger.info(f"Manifests saved to: {manifest_dir}")

    logger.info("")
    logger.info("Done!")

    # Print next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"Data location: {data_dir}/LibriSpeech/")
    if args.create_manifests:
        print(f"Manifests: {manifest_dir}/")
        print("\nTo train with this data, update configs/train.yaml:")
        print(f"  train_manifest: {manifest_dir}/train-all.jsonl")
        print(f"  val_manifest: {manifest_dir}/dev-all.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
