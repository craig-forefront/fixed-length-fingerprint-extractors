"""
Multi-finger fusion module for enhanced biometric verification and identification.

This module provides strategies to combine embeddings or scores from multiple fingers
captured in a single slap to improve verification accuracy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch

from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.embedding_loader import EmbeddingLoader
from flx.benchmarks.matchers import BiometricMatcher, CosineSimilarityMatcher


class FusionStrategy(Enum):
    """Available fusion strategies for multi-finger comparison."""

    # Score-level fusion
    SCORE_SUM = "score_sum"
    SCORE_MAX = "score_max"
    SCORE_MEAN = "score_mean"
    SCORE_WEIGHTED = "score_weighted"

    # Embedding-level fusion
    EMBEDDING_CONCAT = "embedding_concat"
    EMBEDDING_MEAN = "embedding_mean"
    EMBEDDING_ATTENTION = "embedding_attention"


@dataclass
class MultiFingerIdentifier:
    """
    Identifier for a multi-finger capture (slap).

    Groups multiple finger identifiers that belong to the same capture session.
    """

    person_id: int
    slap_id: int
    finger_ids: list[Identifier]  # Individual finger identifiers

    def __hash__(self):
        return hash((self.person_id, self.slap_id))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def num_fingers(self) -> int:
        return len(self.finger_ids)


class MultiFingerEmbeddingLoader(DataLoader):
    """
    Loader that groups individual finger embeddings by slap/person.

    Provides access to embeddings organized by multi-finger capture sessions.
    """

    def __init__(
        self,
        embedding_loader: EmbeddingLoader,
        num_fingers: int = 4,
        encode_finger_in_subject: bool = True,
    ):
        """
        Initialize the multi-finger embedding loader.

        Args:
            embedding_loader: Underlying single-finger embedding loader
            num_fingers: Number of fingers per slap
            encode_finger_in_subject: If True, assumes subject = person * num_fingers + finger_pos
        """
        self._embeddings = embedding_loader
        self._num_fingers = num_fingers
        self._encode_finger_in_subject = encode_finger_in_subject

        # Build multi-finger identifier set
        self._multi_finger_ids = self._build_multi_finger_ids()

    def _build_multi_finger_ids(self) -> dict[tuple[int, int], MultiFingerIdentifier]:
        """Group individual finger IDs into multi-finger identifiers."""
        # Group by (person, impression/slap)
        groups: dict[tuple[int, int], list[Identifier]] = {}

        for finger_id in self._embeddings.ids.identifiers:
            if self._encode_finger_in_subject:
                person = finger_id.subject // self._num_fingers
                slap = finger_id.impression
            else:
                # Assume SlapIdentifier-style encoding
                person = finger_id.subject
                slap = finger_id.impression

            key = (person, slap)
            if key not in groups:
                groups[key] = []
            groups[key].append(finger_id)

        # Convert to MultiFingerIdentifiers
        result = {}
        for (person, slap), finger_ids in groups.items():
            # Sort by finger position for consistency
            if self._encode_finger_in_subject:
                finger_ids.sort(key=lambda x: x.subject % self._num_fingers)
            result[(person, slap)] = MultiFingerIdentifier(
                person_id=person, slap_id=slap, finger_ids=finger_ids
            )

        return result

    @property
    def multi_finger_ids(self) -> list[MultiFingerIdentifier]:
        return list(self._multi_finger_ids.values())

    @property
    def num_persons(self) -> int:
        return len(set(mf.person_id for mf in self._multi_finger_ids.values()))

    def get(self, identifier: MultiFingerIdentifier) -> list[np.ndarray]:
        """Get embeddings for all fingers in a multi-finger capture."""
        return [self._embeddings.get(fid) for fid in identifier.finger_ids]

    def get_stacked(self, identifier: MultiFingerIdentifier) -> np.ndarray:
        """Get embeddings as a stacked array (num_fingers x embedding_dim)."""
        return np.stack(self.get(identifier))

    def get_by_person_slap(self, person: int, slap: int) -> list[np.ndarray]:
        """Get embeddings by person and slap ID."""
        key = (person, slap)
        if key not in self._multi_finger_ids:
            raise KeyError(f"No multi-finger capture found for person={person}, slap={slap}")
        return self.get(self._multi_finger_ids[key])


class EmbeddingFuser(ABC):
    """Base class for embedding-level fusion strategies."""

    @abstractmethod
    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Fuse multiple finger embeddings into a single embedding."""
        pass


class ConcatEmbeddingFuser(EmbeddingFuser):
    """Concatenate embeddings from all fingers."""

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        concatenated = np.concatenate(embeddings)
        # Normalize the concatenated embedding
        return concatenated / np.linalg.norm(concatenated)


class MeanEmbeddingFuser(EmbeddingFuser):
    """Average embeddings from all fingers."""

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        mean_emb = np.mean(embeddings, axis=0)
        # Normalize the mean embedding
        return mean_emb / np.linalg.norm(mean_emb)


class WeightedEmbeddingFuser(EmbeddingFuser):
    """Weighted average of embeddings with learnable or fixed weights."""

    def __init__(self, weights: Union[list[float], np.ndarray] = None, num_fingers: int = 4):
        if weights is None:
            # Default: equal weights
            self.weights = np.ones(num_fingers) / num_fingers
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # Normalize

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        weights = self.weights[: len(embeddings)]
        weights = weights / np.sum(weights)  # Re-normalize if fewer fingers
        weighted = np.sum([w * e for w, e in zip(weights, embeddings)], axis=0)
        return weighted / np.linalg.norm(weighted)


class AttentionEmbeddingFuser(EmbeddingFuser):
    """
    Attention-based fusion that learns to weight embeddings based on quality.

    Uses embedding magnitude as a proxy for quality/confidence.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        # Use L2 norm as quality indicator (before normalization)
        # Higher norm = more distinctive features = higher quality
        stacked = np.stack(embeddings)

        # Compute attention weights based on embedding characteristics
        # Use variance across dimensions as quality proxy
        quality_scores = np.var(stacked, axis=1)
        attention_weights = np.exp(quality_scores / self.temperature)
        attention_weights = attention_weights / np.sum(attention_weights)

        # Weighted combination
        fused = np.sum(attention_weights[:, np.newaxis] * stacked, axis=0)
        return fused / np.linalg.norm(fused)


class ScoreFuser(ABC):
    """Base class for score-level fusion strategies."""

    @abstractmethod
    def fuse(self, scores: list[float]) -> float:
        """Fuse multiple comparison scores into a single score."""
        pass


class SumScoreFuser(ScoreFuser):
    """Sum of all finger comparison scores."""

    def fuse(self, scores: list[float]) -> float:
        return sum(scores)


class MaxScoreFuser(ScoreFuser):
    """Maximum score across all finger comparisons."""

    def fuse(self, scores: list[float]) -> float:
        return max(scores)


class MeanScoreFuser(ScoreFuser):
    """Mean score across all finger comparisons."""

    def fuse(self, scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0


class WeightedScoreFuser(ScoreFuser):
    """Weighted score fusion with configurable weights per finger position."""

    def __init__(self, weights: Union[list[float], np.ndarray] = None, num_fingers: int = 4):
        if weights is None:
            self.weights = [1.0] * num_fingers
        else:
            self.weights = list(weights)

    def fuse(self, scores: list[float]) -> float:
        weights = self.weights[: len(scores)]
        total_weight = sum(weights)
        return sum(w * s for w, s in zip(weights, scores)) / total_weight


class MultiFingerMatcher(BiometricMatcher):
    """
    Matcher that compares multi-finger captures using fusion strategies.

    Supports both score-level and embedding-level fusion.
    """

    def __init__(
        self,
        embedding_loader: EmbeddingLoader,
        strategy: FusionStrategy = FusionStrategy.SCORE_MEAN,
        num_fingers: int = 4,
        encode_finger_in_subject: bool = True,
        weights: list[float] = None,
    ):
        """
        Initialize the multi-finger matcher.

        Args:
            embedding_loader: Single-finger embedding loader
            strategy: Fusion strategy to use
            num_fingers: Number of fingers per slap
            encode_finger_in_subject: If True, subject = person * num_fingers + finger_pos
            weights: Optional weights for weighted fusion strategies
        """
        self._embeddings = embedding_loader
        self._num_fingers = num_fingers
        self._encode_finger_in_subject = encode_finger_in_subject
        self._strategy = strategy

        # Initialize appropriate fuser
        self._score_fuser: ScoreFuser = None
        self._embedding_fuser: EmbeddingFuser = None

        if strategy == FusionStrategy.SCORE_SUM:
            self._score_fuser = SumScoreFuser()
        elif strategy == FusionStrategy.SCORE_MAX:
            self._score_fuser = MaxScoreFuser()
        elif strategy == FusionStrategy.SCORE_MEAN:
            self._score_fuser = MeanScoreFuser()
        elif strategy == FusionStrategy.SCORE_WEIGHTED:
            self._score_fuser = WeightedScoreFuser(weights, num_fingers)
        elif strategy == FusionStrategy.EMBEDDING_CONCAT:
            self._embedding_fuser = ConcatEmbeddingFuser()
        elif strategy == FusionStrategy.EMBEDDING_MEAN:
            self._embedding_fuser = MeanEmbeddingFuser()
        elif strategy == FusionStrategy.EMBEDDING_ATTENTION:
            self._embedding_fuser = AttentionEmbeddingFuser()

        # Build multi-finger loader
        self._mf_loader = MultiFingerEmbeddingLoader(
            embedding_loader, num_fingers, encode_finger_in_subject
        )

        # Cache for fused embeddings (only for embedding-level fusion)
        self._fused_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_finger_ids(self, person: int, slap: int) -> list[Identifier]:
        """Get individual finger identifiers for a person/slap."""
        finger_ids = []
        for finger_pos in range(self._num_fingers):
            if self._encode_finger_in_subject:
                subject = person * self._num_fingers + finger_pos
                finger_id = Identifier(subject, slap)
            else:
                finger_id = Identifier(person, slap)

            # Check if this finger exists
            try:
                self._embeddings.get(finger_id)
                finger_ids.append(finger_id)
            except (KeyError, IndexError):
                continue  # Finger not available

        return finger_ids

    def _get_fused_embedding(self, person: int, slap: int) -> np.ndarray:
        """Get or compute fused embedding for a multi-finger capture."""
        key = (person, slap)
        if key not in self._fused_cache:
            finger_ids = self._get_finger_ids(person, slap)
            embeddings = [self._embeddings.get(fid) for fid in finger_ids]
            self._fused_cache[key] = self._embedding_fuser.fuse(embeddings)
        return self._fused_cache[key]

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        """
        Compare two multi-finger captures.

        For score-level fusion: compares corresponding fingers and fuses scores.
        For embedding-level fusion: fuses embeddings first, then compares.

        Note: sample1 and sample2 should be identifiers where:
        - subject = person_id (not encoded with finger)
        - impression = slap_id
        """
        person1, slap1 = sample1.subject, sample1.impression
        person2, slap2 = sample2.subject, sample2.impression

        if self._embedding_fuser is not None:
            # Embedding-level fusion
            emb1 = self._get_fused_embedding(person1, slap1)
            emb2 = self._get_fused_embedding(person2, slap2)
            return float(np.dot(emb1, emb2))

        else:
            # Score-level fusion
            finger_ids1 = self._get_finger_ids(person1, slap1)
            finger_ids2 = self._get_finger_ids(person2, slap2)

            # Compare corresponding fingers
            scores = []
            for fid1 in finger_ids1:
                for fid2 in finger_ids2:
                    # Only compare same finger positions
                    if self._encode_finger_in_subject:
                        pos1 = fid1.subject % self._num_fingers
                        pos2 = fid2.subject % self._num_fingers
                        if pos1 != pos2:
                            continue

                    emb1 = self._embeddings.get(fid1)
                    emb2 = self._embeddings.get(fid2)
                    score = float(np.dot(emb1, emb2))
                    scores.append(max(0, score))  # Clip negative scores

            if not scores:
                return 0.0

            return self._score_fuser.fuse(scores)

    def clear_cache(self):
        """Clear the fused embedding cache."""
        self._fused_cache.clear()


def create_multi_finger_comparisons(
    multi_finger_ids: list[MultiFingerIdentifier],
    num_mated: int = None,
    num_non_mated: int = None,
) -> tuple[list[tuple[MultiFingerIdentifier, MultiFingerIdentifier]], list[tuple[MultiFingerIdentifier, MultiFingerIdentifier]]]:
    """
    Create mated and non-mated comparison pairs for multi-finger verification.

    Args:
        multi_finger_ids: List of multi-finger identifiers
        num_mated: Number of mated comparisons to generate (None = all possible)
        num_non_mated: Number of non-mated comparisons to generate (None = all possible)

    Returns:
        Tuple of (mated_pairs, non_mated_pairs)
    """
    # Group by person
    by_person: dict[int, list[MultiFingerIdentifier]] = {}
    for mfid in multi_finger_ids:
        if mfid.person_id not in by_person:
            by_person[mfid.person_id] = []
        by_person[mfid.person_id].append(mfid)

    # Generate mated pairs (same person, different slaps)
    mated_pairs = []
    for person, slaps in by_person.items():
        for i, slap1 in enumerate(slaps):
            for slap2 in slaps[i + 1 :]:
                mated_pairs.append((slap1, slap2))

    # Generate non-mated pairs (different persons)
    non_mated_pairs = []
    persons = list(by_person.keys())
    for i, p1 in enumerate(persons):
        for p2 in persons[i + 1 :]:
            for slap1 in by_person[p1]:
                for slap2 in by_person[p2]:
                    non_mated_pairs.append((slap1, slap2))

    # Subsample if requested
    if num_mated is not None and num_mated < len(mated_pairs):
        indices = np.random.choice(len(mated_pairs), num_mated, replace=False)
        mated_pairs = [mated_pairs[i] for i in indices]

    if num_non_mated is not None and num_non_mated < len(non_mated_pairs):
        indices = np.random.choice(len(non_mated_pairs), num_non_mated, replace=False)
        non_mated_pairs = [non_mated_pairs[i] for i in indices]

    return mated_pairs, non_mated_pairs
