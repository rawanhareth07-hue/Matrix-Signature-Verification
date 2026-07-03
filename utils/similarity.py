"""
Similarity engine module for AI Signature Verification & Anti-Forgery System.

Provides comparative analysis of feature vectors using multiple metrics:
Cosine Similarity, Euclidean Distance, Manhattan Distance, and Correlation.
Calculates weighted overall similarity and determines authentication verdicts.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from config import SIMILARITY_THRESHOLD, DISTANCE_WEIGHTS, logger


class SimilarityEngine:
    """Computes similarity scores and authentication decisions between signature feature vectors.

    Uses a weighted combination of Cosine Similarity, Euclidean Distance,
    Manhattan Distance, and Pearson Correlation to produce a single robust similarity score.
    """

    def __init__(self, threshold: float = None) -> None:
        """Initialize the SimilarityEngine.

        Args:
            threshold: Similarity threshold to decide between Genuine and Forged.
                       Defaults to config.SIMILARITY_THRESHOLD.
        """
        self.threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
        self.weights = DISTANCE_WEIGHTS
        logger.debug("SimilarityEngine initialized with threshold %.2f", self.threshold)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate the cosine similarity between two vectors.

        Formula: (v1 . v2) / (||v1|| * ||v2||)

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            Cosine similarity score (0.0 to 1.0 for normalized feature vectors).
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity = float(dot_product / (norm_v1 * norm_v2))
        # Clip to ensure valid float range [0.0, 1.0] since feature vectors are positive-only
        return float(np.clip(similarity, 0.0, 1.0))

    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate the Euclidean L2 distance between two vectors.

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            L2 distance.
        """
        return float(np.linalg.norm(v1 - v2))

    def manhattan_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate the Manhattan L1 distance between two vectors.

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            L1 distance.
        """
        return float(np.sum(np.abs(v1 - v2)))

    def correlation(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate the Pearson correlation coefficient between two vectors.

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            Correlation coefficient (-1.0 to 1.0).
        """
        if np.std(v1) == 0 or np.std(v2) == 0:
            return 0.0
        corr_matrix = np.corrcoef(v1, v2)
        return float(corr_matrix[0, 1])

    def compare_signatures(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, Any]:
        """Perform a full comparative analysis between two feature vectors.

        Computes individual metrics, maps distances to similarity percentages,
        and computes the weighted overall similarity and confidence scores.

        Args:
            vec1: Query feature vector.
            vec2: Stored reference feature vector.

        Returns:
            Dict containing:
                - cosine_similarity (float, 0-1)
                - euclidean_distance (float)
                - manhattan_distance (float)
                - correlation (float, -1 to 1)
                - overall_similarity (float, 0-1)
                - confidence (float, 0-1)
                - decision (str, 'Genuine' or 'Forged')
        """
        try:
            # Calculate raw metrics
            cos_sim = self.cosine_similarity(vec1, vec2)
            euc_dist = self.euclidean_distance(vec1, vec2)
            man_dist = self.manhattan_distance(vec1, vec2)
            corr = self.correlation(vec1, vec2)

            # Map distance metrics to similarity metrics in range [0, 1]
            # 1 / (1 + distance) is a standard mapping
            euc_sim = 1.0 / (1.0 + euc_dist)
            man_sim = 1.0 / (1.0 + man_dist)
            # Map correlation from [-1, 1] to [0, 1]
            corr_sim = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

            # Weighted sum of similarities
            overall_sim = (
                self.weights.get("cosine", 0.4) * cos_sim +
                self.weights.get("euclidean", 0.25) * euc_sim +
                self.weights.get("manhattan", 0.15) * man_sim +
                self.weights.get("correlation", 0.2) * corr_sim
            )

            # Safety clip
            overall_sim = float(np.clip(overall_sim, 0.0, 1.0))

            # Decision based on threshold
            decision = "Genuine" if overall_sim >= self.threshold else "Forged"

            # Confidence is high if overall similarity is far from the decision boundary
            # e.g., if threshold is 0.75:
            # - if similarity is 0.95 (distance = 0.20), confidence is very high
            # - if similarity is 0.76 (distance = 0.01), confidence is low (borderline decision)
            margin = abs(overall_sim - self.threshold)
            # Normalize confidence based on maximum possible margin
            max_margin = self.threshold if overall_sim < self.threshold else (1.0 - self.threshold)
            confidence = float(np.clip(0.5 + (margin / max_margin) * 0.5, 0.5, 1.0)) if max_margin > 0 else 0.5

            return {
                "cosine_similarity": cos_sim,
                "euclidean_distance": euc_dist,
                "manhattan_distance": man_dist,
                "correlation": corr,
                "overall_similarity": overall_sim,
                "confidence": confidence,
                "decision": decision,
            }

        except Exception as e:
            logger.error("Failed to compare signature vectors: %s", e)
            raise

    def find_most_similar(
        self, query_vector: np.ndarray, stored_signatures: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Find the best matching signature from a list of stored signatures.

        Args:
            query_vector: Feature vector of the signature to verify.
            stored_signatures: List of signature records from the database.

        Returns:
            Tuple of (best_matching_signature_dict, comparison_result_dict)
            or (None, None) if stored_signatures is empty.
        """
        if not stored_signatures:
            return None, None

        best_match = None
        best_result = None
        highest_similarity = -1.0

        for sig in stored_signatures:
            stored_vector = sig["feature_vector"]
            # Verify feature vector format
            if not isinstance(stored_vector, np.ndarray):
                continue

            result = self.compare_signatures(query_vector, stored_vector)
            similarity = result["overall_similarity"]

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = sig
                best_result = result

        return best_match, best_result
