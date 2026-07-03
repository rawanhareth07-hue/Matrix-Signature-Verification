"""
Linear algebra demonstration module for AI Signature Verification.

Computes mathematical operations on image matrices and feature vectors, and
returns step-by-step calculations, LaTeX formulas, and visualization-ready
structures for educational demonstration.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, List
from config import FEATURE_VECTOR_SIZE, logger


class LinearAlgebraDemo:
    """Calculates linear algebra concepts on signature matrices and feature vectors.

    Generates results, LaTeX formulas, textual explanations, and plotting data
    for educational screens.
    """

    def __init__(self) -> None:
        """Initialize the LinearAlgebraDemo."""
        logger.debug("LinearAlgebraDemo initialized.")

    def get_all_demonstrations(self, image: np.ndarray, feature_vector: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run all linear algebra operations on the provided signature image and feature vector.

        Args:
            image: Preprocessed signature image (single-channel 2D numpy array).
            feature_vector: Computed signature feature vector (1D numpy array of size 256).

        Returns:
            Dict where keys are mathematical concepts and values are dictionaries with:
                - result (numpy array, scalar, or tuple)
                - formula (str, LaTeX format)
                - explanation (str, educational description)
                - visualization_data (dict, optional data structures for plotting)
        """
        try:
            # Check image dimension
            if len(image.shape) > 2:
                img_2d = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img_2d = image.copy()

            # Ensure image is resized to standard size to ensure consistent operations
            img_resized = cv2.resize(img_2d, (128, 128))
            _, img_binary = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)

            # Sample 8x8 block for matrix-specific demonstrations (e.g., from center of the image)
            h, w = img_resized.shape
            sample_8x8 = img_resized[h//2-4:h//2+4, w//2-4:w//2+4].astype(np.float32)
            sample_8x8_bin = (img_binary[h//2-4:h//2+4, w//2-4:w//2+4] > 0).astype(np.float32)

            demos = {}

            # 1. Original Matrix
            demos["original_matrix"] = {
                "result": sample_8x8,
                "formula": r"\mathbf{A} \in \mathbb{R}^{8 \times 8}",
                "explanation": "A signature image is stored in a computer as a matrix of intensity values. Here is a small 8x8 pixel sample from the center of your image.",
                "visualization_data": {"matrix": sample_8x8.tolist()}
            }

            # 2. Binary Matrix
            demos["binary_matrix"] = {
                "result": sample_8x8_bin,
                "formula": r"\mathbf{B}_{ij} = \begin{cases} 1 & \text{if } A_{ij} \ge \tau \\ 0 & \text{if } A_{ij} < \tau \end{cases}",
                "explanation": "Binarization transforms grayscale intensity values into zeros (background) and ones (signature stroke) based on a threshold parameter.",
                "visualization_data": {"matrix": sample_8x8_bin.tolist()}
            }

            # 3. Matrix Dimensions
            demos["matrix_dimensions"] = {
                "result": img_resized.shape,
                "formula": r"\text{dim}(\mathbf{A}) = m \times n",
                "explanation": f"Matrix dimensions represent the resolution of the image. The signature is resized to a standardized space of {img_resized.shape[0]} rows by {img_resized.shape[1]} columns.",
                "visualization_data": {"rows": img_resized.shape[0], "cols": img_resized.shape[1]}
            }

            # 4. Feature Vector
            demos["feature_vector"] = {
                "result": feature_vector,
                "formula": r"\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^{256}",
                "explanation": "Feature extraction converts the raw pixel matrix into a compact 1D vector (descriptor) containing pixel densities, projection profiles, and structural features.",
                "visualization_data": {"vector": feature_vector.tolist()}
            }

            # 5. Vector Length / L2 Norm
            l2_norm = float(np.linalg.norm(feature_vector))
            demos["vector_length"] = {
                "result": l2_norm,
                "formula": r"\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{d} x_i^2}",
                "explanation": "The Euclidean length (L2 norm) represents the magnitude of the feature vector. Normalizing this vector to unit length (length = 1.0) ensures comparisons are scale-invariant.",
                "visualization_data": {"norm": l2_norm}
            }

            # 6. Matrix Multiplication
            # Multiply two 3x3 submatrices from the sample
            sub_m1 = sample_8x8[0:3, 0:3]
            sub_m2 = sample_8x8[3:6, 3:6]
            mult_res = np.matmul(sub_m1, sub_m2)
            demos["matrix_multiplication"] = {
                "result": mult_res,
                "formula": r"\mathbf{C} = \mathbf{X}\mathbf{Y}, \quad c_{ij} = \sum_{k=1}^{p} x_{ik} y_{kj}",
                "explanation": "Matrix multiplication represents composition of linear maps. It is used during filtering operations, transformations, and projections.",
                "visualization_data": {"matrix_x": sub_m1.tolist(), "matrix_y": sub_m2.tolist(), "result_matrix": mult_res.tolist()}
            }

            # 7. Transformation Matrix & Linear Transformation
            # Define a 2D rotation matrix (rotation of 30 degrees)
            theta = np.radians(30)
            c, s = np.cos(theta), np.sin(theta)
            transform_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
            
            # Create a simple box outline to transform
            points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=np.float32)
            transformed_points = np.dot(points, transform_mat.T)

            demos["transformation_matrix"] = {
                "result": transform_mat,
                "formula": r"\mathbf{T} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}",
                "explanation": "A 2D transformation matrix rotates, scales, or shears coordinates. This rotation matrix is used to align signatures or rotate them for orientation-invariant matching.",
                "visualization_data": {"matrix": transform_mat.tolist()}
            }

            demos["linear_transformation"] = {
                "result": transformed_points,
                "formula": r"\mathbf{v}' = \mathbf{T}\mathbf{v}",
                "explanation": "Applying a transformation matrix to coordinates maps points from one linear space to another. Here we show how a square is rotated by 30 degrees.",
                "visualization_data": {
                    "original_x": points[:, 0].tolist(),
                    "original_y": points[:, 1].tolist(),
                    "transformed_x": transformed_points[:, 0].tolist(),
                    "transformed_y": transformed_points[:, 1].tolist()
                }
            }

            # 8. Dot Product
            # Take two rows of the image block
            r1 = sample_8x8[0, :]
            r2 = sample_8x8[1, :]
            dot_val = float(np.dot(r1, r2))
            demos["dot_product"] = {
                "result": dot_val,
                "formula": r"\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i",
                "explanation": "The dot product measures the projection of one vector onto another. It is the fundamental operation behind cosine similarity and linear projection.",
                "visualization_data": {"v1": r1.tolist(), "v2": r2.tolist(), "dot_product": dot_val}
            }

            # 9. Vector Norms
            l1_norm = float(np.sum(np.abs(feature_vector)))
            l2_norm = float(np.linalg.norm(feature_vector))
            linf_norm = float(np.max(np.abs(feature_vector)))
            demos["vector_norm"] = {
                "result": (l1_norm, l2_norm, linf_norm),
                "formula": r"\|\mathbf{x}\|_p = \left( \sum |x_i|^p \right)^{1/p}, \quad \|\mathbf{x}\|_\infty = \max |x_i|",
                "explanation": "Different norms measure vector magnitude differently: L1 (sum of absolute values), L2 (Euclidean distance), and L-infinity (maximum absolute value). We use L2 norm for standardizing profiles.",
                "visualization_data": {"l1": l1_norm, "l2": l2_norm, "linf": linf_norm}
            }

            # 10. Cosine Similarity
            denom = np.linalg.norm(r1) * np.linalg.norm(r2)
            cos_sim_val = float(np.dot(r1, r2) / denom) if denom > 0 else 0.0
            demos["cosine_similarity"] = {
                "result": cos_sim_val,
                "formula": r"\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}",
                "explanation": "Cosine similarity measures the angle between two vectors, regardless of their magnitude. An angle of 0 degrees means the vectors are perfectly aligned (similarity = 1.0).",
                "visualization_data": {"cos_sim": cos_sim_val}
            }

            # 11. Matrix Rank
            rank = int(np.linalg.matrix_rank(img_resized.astype(np.float32)))
            demos["matrix_rank"] = {
                "result": rank,
                "formula": r"\text{rank}(\mathbf{A}) = \text{number of linearly independent columns}",
                "explanation": f"Matrix rank measures the amount of unique information in the image. Out of 128 rows, there are {rank} linearly independent rows/columns in this signature matrix.",
                "visualization_data": {"rank": rank, "total_dimensions": img_resized.shape[0]}
            }

            # 12. Transpose
            transpose_mat = sample_8x8.T
            demos["transpose"] = {
                "result": transpose_mat,
                "formula": r"\mathbf{A}^T_{ij} = \mathbf{A}_{ji}",
                "explanation": "Transposing flips a matrix over its diagonal, exchanging rows for columns. In image processing, transposing is used to switch coordinates or flip image orientations.",
                "visualization_data": {"original": sample_8x8.tolist(), "transpose": transpose_mat.tolist()}
            }

            # 13. Eigenvalues & Eigenvectors
            # Create a symmetric 4x4 submatrix to guarantee real eigenvalues
            sym_mat = sample_8x8[0:4, 0:4]
            sym_mat = (sym_mat + sym_mat.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(sym_mat)
            demos["eigenvalues"] = {
                "result": (eigenvalues, eigenvectors),
                "formula": r"\mathbf{M}\mathbf{v} = \lambda \mathbf{v}",
                "explanation": "Eigenvalues (λ) represent scaling factors along principal axes of variance, and eigenvectors (v) represent the directions. These form the foundation of Principal Component Analysis.",
                "visualization_data": {
                    "matrix": sym_mat.tolist(),
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.tolist()
                }
            }

            # 14. Singular Value Decomposition (SVD)
            u, s_vals, vt = np.linalg.svd(sample_8x8)
            # Reconstruct with only top 2 singular values to show approximation
            s_mat = np.zeros((8, 8))
            for i in range(2):
                s_mat[i, i] = s_vals[i]
            reconstructed_8x8 = np.dot(u, np.dot(s_mat, vt))
            demos["svd"] = {
                "result": (u, s_vals, vt),
                "formula": r"\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T",
                "explanation": "SVD factorizes a matrix into rotation (U, V) and scaling (Σ) components. It is used to compress signature images and filter high-frequency noise.",
                "visualization_data": {
                    "singular_values": s_vals.tolist(),
                    "original": sample_8x8.tolist(),
                    "reconstructed_top2": reconstructed_8x8.tolist()
                }
            }

            # 15. Principal Component Analysis (PCA)
            # Center the image rows
            mean_row = np.mean(img_resized, axis=0)
            centered_img = img_resized - mean_row
            # Compute covariance
            cov_mat = np.cov(centered_img.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_mat)
            # Sort descending
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Cumulative variance explained
            total_variance = np.sum(eigenvals)
            variance_explained = eigenvals / total_variance if total_variance > 0 else np.zeros_like(eigenvals)
            cum_variance = np.cumsum(variance_explained)

            demos["pca"] = {
                "result": (eigenvals, eigenvecs, cum_variance),
                "formula": r"\mathbf{W}_k = \text{Top } k \text{ eigenvectors of } \mathbf{\Sigma}_{cov}",
                "explanation": "PCA projects high-dimensional features into a lower-dimensional subspace while maximizing variance. This captures the most discriminative signature characteristics.",
                "visualization_data": {
                    "variance_explained": variance_explained[:20].tolist(),
                    "cumulative_variance": cum_variance[:20].tolist()
                }
            }

            return demos

        except Exception as e:
            logger.error("Failed to run linear algebra demonstrations: %s", e)
            raise
