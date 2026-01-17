"""
CTC Decoding utilities for VELOCITY-ASR v2.

This module implements greedy and beam search decoding for CTC output.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


# Default blank token index (usually 0 in CTC)
BLANK_TOKEN = 0


@dataclass
class DecodingResult:
    """Result of CTC decoding."""

    text: str
    tokens: List[int]
    score: float
    timestamps: Optional[List[Tuple[int, int]]] = None


def ctc_greedy_decode(
    logits: torch.Tensor,
    blank_token: int = BLANK_TOKEN,
    collapse_repeated: bool = True,
) -> List[List[int]]:
    """
    Greedy CTC decoding.

    Takes argmax at each timestep and removes blanks and repeated tokens.

    Args:
        logits: CTC logits of shape (batch, seq_len, vocab_size)
        blank_token: Index of blank token
        collapse_repeated: Whether to collapse repeated tokens

    Returns:
        List of decoded token sequences (one per batch item)
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)

    batch_size = predictions.size(0)
    decoded = []

    for b in range(batch_size):
        pred = predictions[b].tolist()
        tokens = []
        prev_token = None

        for token in pred:
            # Skip blank tokens
            if token == blank_token:
                prev_token = None
                continue

            # Skip repeated tokens if requested
            if collapse_repeated and token == prev_token:
                continue

            tokens.append(token)
            prev_token = token

        decoded.append(tokens)

    return decoded


def ctc_greedy_decode_with_timestamps(
    logits: torch.Tensor,
    blank_token: int = BLANK_TOKEN,
) -> List[Tuple[List[int], List[Tuple[int, int]]]]:
    """
    Greedy CTC decoding with token timestamps.

    Args:
        logits: CTC logits of shape (batch, seq_len, vocab_size)
        blank_token: Index of blank token

    Returns:
        List of (tokens, timestamps) tuples for each batch item.
        Timestamps are (start_frame, end_frame) tuples.
    """
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)

    batch_size = predictions.size(0)
    results = []

    for b in range(batch_size):
        pred = predictions[b].tolist()
        tokens = []
        timestamps = []
        prev_token = None
        start_frame = 0

        for frame_idx, token in enumerate(pred):
            if token == blank_token:
                if prev_token is not None and prev_token != blank_token:
                    # End of previous token
                    timestamps.append((start_frame, frame_idx))
                prev_token = token
                continue

            if token != prev_token:
                if prev_token is not None and prev_token != blank_token:
                    # End of previous token
                    timestamps.append((start_frame, frame_idx))
                # Start of new token
                tokens.append(token)
                start_frame = frame_idx

            prev_token = token

        # Handle last token
        if prev_token is not None and prev_token != blank_token:
            timestamps.append((start_frame, len(pred)))

        results.append((tokens, timestamps))

    return results


def ctc_beam_search(
    logits: torch.Tensor,
    beam_width: int = 10,
    blank_token: int = BLANK_TOKEN,
    lm_weight: float = 0.0,
    lm_scorer: Optional[Any] = None,
) -> List[List[DecodingResult]]:
    """
    Beam search CTC decoding.

    Maintains multiple hypotheses and scores them by CTC probability.
    Optionally integrates language model scores.

    Args:
        logits: CTC logits of shape (batch, seq_len, vocab_size)
        beam_width: Number of beams to maintain
        blank_token: Index of blank token
        lm_weight: Weight for language model scores
        lm_scorer: Optional language model scorer

    Returns:
        List of beam results for each batch item.
        Each beam result is a list of DecodingResult sorted by score.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = log_probs.shape

    all_results = []

    for b in range(batch_size):
        # Initialize beams: (prefix_tuple, score, last_token)
        # prefix_tuple is the sequence without blanks/repeats
        beams: Dict[Tuple[int, ...], Tuple[float, Optional[int]]] = {
            (): (0.0, None)
        }

        for t in range(seq_len):
            new_beams: Dict[Tuple[int, ...], Tuple[float, Optional[int]]] = {}

            for prefix, (score, last_token) in beams.items():
                # Extend with blank
                blank_score = score + log_probs[b, t, blank_token].item()
                key = prefix
                if key not in new_beams or new_beams[key][0] < blank_score:
                    new_beams[key] = (blank_score, blank_token)

                # Extend with non-blank tokens
                for token in range(vocab_size):
                    if token == blank_token:
                        continue

                    token_score = score + log_probs[b, t, token].item()

                    # If same as last token, don't extend prefix
                    if last_token == token:
                        key = prefix
                    else:
                        key = prefix + (token,)

                    # Add LM score if available
                    if lm_scorer is not None and lm_weight > 0:
                        lm_score = lm_scorer.score(list(key))
                        token_score += lm_weight * lm_score

                    if key not in new_beams or new_beams[key][0] < token_score:
                        new_beams[key] = (token_score, token)

            # Prune to beam width
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: x[1][0],
                reverse=True,
            )[:beam_width]

            beams = dict(sorted_beams)

        # Convert beams to results
        results = []
        for prefix, (score, _) in sorted(
            beams.items(), key=lambda x: x[1][0], reverse=True
        ):
            results.append(DecodingResult(
                text="",  # Will be filled by tokenizer
                tokens=list(prefix),
                score=score,
            ))

        all_results.append(results)

    return all_results


class CTCDecoder:
    """
    CTC Decoder with vocabulary support.

    Wraps decoding functions and handles token-to-text conversion.

    Args:
        vocabulary: List of tokens (vocab[i] = token string for index i)
        blank_token: Index of blank token
    """

    def __init__(
        self,
        vocabulary: List[str],
        blank_token: int = BLANK_TOKEN,
    ):
        self.vocabulary = vocabulary
        self.blank_token = blank_token
        self.vocab_size = len(vocabulary)

        # Build reverse mapping
        self.token_to_idx = {token: idx for idx, token in enumerate(vocabulary)}

    def decode_greedy(
        self,
        logits: torch.Tensor,
        collapse_repeated: bool = True,
    ) -> List[str]:
        """
        Greedy decode and convert to text.

        Args:
            logits: CTC logits (batch, seq_len, vocab_size)
            collapse_repeated: Whether to collapse repeated tokens

        Returns:
            List of decoded strings
        """
        token_sequences = ctc_greedy_decode(
            logits,
            blank_token=self.blank_token,
            collapse_repeated=collapse_repeated,
        )

        return [self._tokens_to_text(tokens) for tokens in token_sequences]

    def decode_beam_search(
        self,
        logits: torch.Tensor,
        beam_width: int = 10,
        return_all_beams: bool = False,
    ) -> List[str]:
        """
        Beam search decode and convert to text.

        Args:
            logits: CTC logits (batch, seq_len, vocab_size)
            beam_width: Number of beams
            return_all_beams: If True, return all beam results

        Returns:
            List of decoded strings (best beam per batch item)
        """
        beam_results = ctc_beam_search(
            logits,
            beam_width=beam_width,
            blank_token=self.blank_token,
        )

        if return_all_beams:
            # Convert all beams to text
            for batch_results in beam_results:
                for result in batch_results:
                    result.text = self._tokens_to_text(result.tokens)
            return beam_results

        # Return best beam text
        return [
            self._tokens_to_text(results[0].tokens) if results else ""
            for results in beam_results
        ]

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token indices to text string."""
        chars = []
        for token in tokens:
            if 0 <= token < self.vocab_size:
                chars.append(self.vocabulary[token])
            else:
                chars.append("<unk>")

        # Join tokens (handle subword tokens starting with special markers)
        text = "".join(chars)

        # Clean up common subword markers
        text = text.replace("▁", " ").strip()

        return text

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        tokens = []
        for char in text:
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            elif "<unk>" in self.token_to_idx:
                tokens.append(self.token_to_idx["<unk>"])
        return tokens


def create_default_vocabulary(vocab_size: int = 50000) -> List[str]:
    """
    Create a default character-level vocabulary.

    For production use, replace with a proper subword vocabulary
    (e.g., SentencePiece, BPE).

    Args:
        vocab_size: Target vocabulary size

    Returns:
        List of vocabulary tokens
    """
    vocab = ["<blank>", "<unk>", "<pad>", " "]

    # Add lowercase letters
    vocab.extend(list("abcdefghijklmnopqrstuvwxyz"))

    # Add uppercase letters
    vocab.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    # Add digits
    vocab.extend(list("0123456789"))

    # Add common punctuation
    vocab.extend(list(".,!?;:'\"()-"))

    # Add placeholder tokens for remaining vocabulary
    current_size = len(vocab)
    for i in range(current_size, vocab_size):
        vocab.append(f"<token_{i}>")

    return vocab
