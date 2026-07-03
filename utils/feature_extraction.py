"""
Feature extraction module for AI Signature Verification & Anti-Forgery System.

Extracts geometrical, structural, and topological features from processed
signature images and compiles them into a unified feature vector of size 256.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Any
from config import FEATURE_VECTOR_SIZE, HISTOGRAM_BINS, logger


class FeatureExtractor:
    """Extracts features from processed signature images to form a descriptor vector.

    Provides methods to compute individual features like pixel density, projection
    histograms, center of mass, aspect ratio, and stroke thickness distribution,
    then combines them into a normalized feature vector.
    """

    def __init__(self) -> None:
        """Initialize the FeatureExtractor."""
        logger.debug("FeatureExtractor initialized.")

    def extract_all_features(self, processed_image: np.ndarray) -> Dict[str, Any]:
        """Extract all signature features from a preprocessed binary image.

        Args:
            processed_image: Preprocessed binary signature image (normalized 0-255).

        Returns:
            Dict containing:
                - pixel_density (float)
                - horizontal_projection (np.ndarray)
                - vertical_projection (np.ndarray)
                - center_of_mass (Tuple[float, float])
                - bounding_box (Tuple[int, int, int, int])
                - aspect_ratio (float)
                - stroke_distribution (np.ndarray)
                - feature_vector (np.ndarray of shape (256,))
        """
        try:
            # Ensure image is binary (0 and 255 or 0 and 1)
            _, binary_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)

            # Get features
            density = self.get_pixel_density(binary_image)
            horiz_proj, vert_proj = self.get_projection_histograms(binary_image)
            com = self.get_center_of_mass(binary_image)
            bbox = self.get_bounding_box(binary_image)
            aspect_ratio = self.get_aspect_ratio(binary_image)
            stroke_dist = self.get_stroke_distribution(binary_image)

            features = {
                "pixel_density": density,
                "horizontal_projection": horiz_proj,
                "vertical_projection": vert_proj,
                "center_of_mass": com,
                "bounding_box": bbox,
                "aspect_ratio": aspect_ratio,
                "stroke_distribution": stroke_dist,
            }

            # Generate the unified feature vector
            features["feature_vector"] = self.generate_feature_vector(features, binary_image.shape)
            return features

        except Exception as e:
            logger.error("Failed to extract features: %s", e)
            raise

    def get_pixel_density(self, image: np.ndarray) -> float:
        """Calculate the ratio of foreground (signature) pixels to total pixels.

        Args:
            image: Binary image.

        Returns:
            Pixel density score (0.0 to 1.0).
        """
        foreground_pixels = np.sum(image > 0)
        total_pixels = image.size
        return float(foreground_pixels / total_pixels) if total_pixels > 0 else 0.0

    def get_projection_histograms(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute horizontal and vertical projection histograms.

        Args:
            image: Binary image.

        Returns:
            Tuple of (horizontal_projection, vertical_projection) as numpy arrays.
        """
        # Sum along columns (vertical projection) and rows (horizontal projection)
        horizontal = np.sum(image > 0, axis=1, dtype=np.float32)
        vertical = np.sum(image > 0, axis=0, dtype=np.float32)

        # Normalize projections to avoid scaling issues
        if np.max(horizontal) > 0:
            horizontal /= np.max(horizontal)
        if np.max(vertical) > 0:
            vertical /= np.max(vertical)

        return horizontal, vertical

    def get_center_of_mass(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate the center of mass (centroid) of the signature.

        Args:
            image: Binary image.

        Returns:
            Tuple of (y, x) coordinates of the center of mass normalized by dimensions.
        """
        # Binarize image to 0 and 1
        binary = (image > 0).astype(np.float32)
        if np.sum(binary) == 0:
            return 0.5, 0.5

        com = ndimage.center_of_mass(binary)
        # Normalize COM coordinates by image dimensions to make it invariant to translation
        normalized_y = float(com[0] / image.shape[0])
        normalized_x = float(com[1] / image.shape[1])
        return normalized_y, normalized_x

    def get_bounding_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Find the bounding box of the active signature region.

        Args:
            image: Binary image.

        Returns:
            Tuple of (x, y, width, height) of the bounding box.
        """
        coords = np.column_stack(np.where(image > 0))
        if coords.size == 0:
            return 0, 0, image.shape[1], image.shape[0]

        # coords are in row, col (y, x) format
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        w = int(max_x - min_x + 1)
        h = int(max_y - min_y + 1)
        return int(min_x), int(min_y), w, h

    def get_aspect_ratio(self, image: np.ndarray) -> float:
        """Calculate the aspect ratio of the signature bounding box.

        Args:
            image: Binary image.

        Returns:
            Aspect ratio (width / height).
        """
        _, _, w, h = self.get_bounding_box(image)
        return float(w / h) if h > 0 else 1.0

    def get_stroke_distribution(self, image: np.ndarray) -> np.ndarray:
        """Compute the stroke width distribution using distance transform.

        Args:
            image: Binary image.

        Returns:
            Normalized histogram of local stroke thickness.
        """
        # Distance transform finds distance from each foreground pixel to nearest background pixel
        # This is a proxy for local stroke width/thickness
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        stroke_values = dist_transform[image > 0]

        if stroke_values.size == 0:
            return np.zeros(HISTOGRAM_BINS, dtype=np.float32)

        # Compute histogram of distances
        hist, _ = np.histogram(stroke_values, bins=HISTOGRAM_BINS, range=(0, 10))
        hist = hist.astype(np.float32)

        # Normalize histogram
        if np.sum(hist) > 0:
            hist /= np.sum(hist)

        return hist

    def generate_feature_vector(self, features: Dict[str, Any], image_shape: Tuple[int, int]) -> np.ndarray:
        """Combine all individual features into a fixed-length normalized vector.

        Args:
            features: Dictionary containing extracted individual features.
            image_shape: Height and width of the input image.

        Returns:
            Numpy array of shape (FEATURE_VECTOR_SIZE,) representing the signature descriptor.
        """
        # 1. Pixel density (1 val)
        density = np.array([features["pixel_density"]], dtype=np.float32)

        # 2. Resized projections to fixed HISTOGRAM_BINS (32 + 32 = 64 vals)
        h_proj = features["horizontal_projection"]
        v_proj = features["vertical_projection"]

        # Downsample or interpolate projection histograms to fit HISTOGRAM_BINS
        h_proj_resized = self._resize_1d_array(h_proj, HISTOGRAM_BINS)
        v_proj_resized = self._resize_1d_array(v_proj, HISTOGRAM_BINS)

        # 3. Center of mass (2 vals)
        com = np.array(features["center_of_mass"], dtype=np.float32)

        # 4. Normalized Bounding Box (4 vals)
        x, y, w, h = features["bounding_box"]
        bbox_norm = np.array([
            x / image_shape[1],
            y / image_shape[0],
            w / image_shape[1],
            h / image_shape[0]
        ], dtype=np.float32)

        # 5. Aspect ratio (1 val)
        # Log scaling to compress large aspect ratios and normalize
        aspect = np.array([np.log1p(features["aspect_ratio"])], dtype=np.float32)

        # 6. Stroke width distribution (32 vals)
        stroke_dist = features["stroke_distribution"]

        # Concatenate all features
        vector = np.concatenate([
            density,
            h_proj_resized,
            v_proj_resized,
            com,
            bbox_norm,
            aspect,
            stroke_dist
        ])

        # Normalization (L2 normalization of the combined feature vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        # Pad or truncate the vector to match the global FEATURE_VECTOR_SIZE (e.g. 256)
        if len(vector) < FEATURE_VECTOR_SIZE:
            padding = np.zeros(FEATURE_VECTOR_SIZE - len(vector), dtype=np.float32)
            vector = np.concatenate([vector, padding])
        elif len(vector) > FEATURE_VECTOR_SIZE:
            vector = vector[:FEATURE_VECTOR_SIZE]

        return vector

    def _resize_1d_array(self, arr: np.ndarray, target_size: int) -> np.ndarray:
        """Resize a 1D array using linear interpolation.

        Args:
            arr: Input 1D array.
            target_size: Target dimension.

        Returns:
            Resized 1D array.
        """
        if len(arr) == 0:
            return np.zeros(target_size, dtype=np.float32)

        # Create interpolation grid
        src_indices = np.linspace(0, len(arr) - 1, len(arr))
        target_indices = np.linspace(0, len(arr) - 1, target_size)

        resized = np.interp(target_indices, src_indices, arr)
        return resized.astype(np.float32)
