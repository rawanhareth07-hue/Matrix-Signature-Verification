"""
Image processing pipeline for signature verification.

Provides a configurable multi-step preprocessing pipeline that converts raw
signature images into clean, normalized representations suitable for feature
extraction and comparison. Each step is individually accessible for
visualization in the Streamlit UI.
"""

import cv2
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple
from skimage.morphology import skeletonize as skimage_skeletonize

from config import (
    DEFAULT_IMAGE_SIZE,
    GRAYSCALE_THRESHOLD,
    NOISE_KERNEL_SIZE,
    MORPH_KERNEL_SIZE,
    logger,
)


class ImageProcessor:
    """Multi-step image processing pipeline for handwritten signatures.

    The pipeline converts raw input images through a series of transformations
    designed to isolate the signature strokes from background noise, normalize
    dimensions, and produce a consistent representation regardless of scanning
    conditions or paper quality.

    Attributes:
        target_size: Desired output dimensions (width, height) for resizing.
    """

    def __init__(self, target_size: Tuple[int, int] = None) -> None:
        """Initialize the image processor.

        Args:
            target_size: Target (width, height) for image resizing.
                         Defaults to config.DEFAULT_IMAGE_SIZE.
        """
        self.target_size = target_size if target_size is not None else DEFAULT_IMAGE_SIZE
        logger.info(
            "ImageProcessor initialized with target_size=%s", self.target_size
        )

    def process_signature(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute the full preprocessing pipeline on a signature image.

        The pipeline proceeds through these stages in order:
        1. Original – keep an unmodified copy
        2. Grayscale – convert colour to single-channel
        3. Resized – standardize dimensions
        4. Denoised – Gaussian blur to remove high-frequency noise
        5. Thresholded – Otsu binarization for clean black/white separation
        6. Morphological – opening then closing to clean stroke edges
        7. Normalized – scale pixel intensities to full 0-255 range
        8. Skeletonized – thin strokes to single-pixel width
        9. Cropped – remove empty borders and re-fit to target size

        Args:
            image: Input image as a numpy array (BGR or grayscale).

        Returns:
            OrderedDict mapping step names to their resulting numpy arrays.
        """
        if image is None or image.size == 0:
            logger.error("Received empty or None image.")
            # Return a dict of blank images so downstream code doesn't crash
            blank = np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )
            return OrderedDict([
                ('original', blank), ('grayscale', blank),
                ('resized', blank), ('denoised', blank),
                ('thresholded', blank), ('morphological', blank),
                ('normalized', blank), ('skeletonized', blank),
                ('cropped', blank),
            ])

        steps = OrderedDict()

        # Stage 1 – preserve the original for display comparison
        steps['original'] = image.copy()

        # Stage 2 – grayscale conversion
        grayscale = self.convert_to_grayscale(image)
        steps['grayscale'] = grayscale

        # Stage 3 – resize to standard dimensions
        resized = self.resize_image(grayscale)
        steps['resized'] = resized

        # Stage 4 – Gaussian noise removal
        denoised = self.remove_noise(resized)
        steps['denoised'] = denoised

        # Stage 5 – binarize using Otsu's method
        thresholded = self.apply_threshold(denoised)
        steps['thresholded'] = thresholded

        # Stage 6 – morphological cleanup
        morphed = self.morphological_operations(thresholded)
        steps['morphological'] = morphed

        # Stage 7 – intensity normalization
        normalized = self.normalize_image(morphed)
        steps['normalized'] = normalized

        # Stage 8 – skeletonization for stroke analysis
        skeleton = self.skeletonize_signature(normalized)
        steps['skeletonized'] = skeleton

        # Stage 9 – crop whitespace borders
        cropped = self.crop_borders(normalized)
        steps['cropped'] = cropped

        logger.info("Signature processing pipeline completed successfully.")
        return steps

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to single-channel grayscale.

        If the image is already grayscale (2D array), it is returned unchanged.
        Multi-channel images are converted using OpenCV's BGR-to-gray formula
        which weights channels according to human luminance perception.

        Args:
            image: Input image (BGR or grayscale numpy array).

        Returns:
            Single-channel grayscale image.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to convert_to_grayscale.")
                return np.zeros(self.target_size[::-1], dtype=np.uint8)

            # Already grayscale – nothing to do
            if len(image.shape) == 2:
                return image.copy()

            # Handle BGRA (4-channel) images by dropping the alpha channel first
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            logger.error("Grayscale conversion failed: %s", e)
            return np.zeros(self.target_size[::-1], dtype=np.uint8)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize an image to the configured target dimensions.

        Uses INTER_AREA interpolation for downscaling (preserves detail)
        and INTER_LINEAR for upscaling.

        Args:
            image: Input image (any size).

        Returns:
            Image resized to (target_size[0], target_size[1]).
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to resize_image.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            h, w = image.shape[:2]
            # Choose interpolation method based on direction of scaling
            if h * w > self.target_size[0] * self.target_size[1]:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR

            resized = cv2.resize(
                image,
                (self.target_size[0], self.target_size[1]),
                interpolation=interpolation,
            )
            return resized
        except cv2.error as e:
            logger.error("Image resizing failed: %s", e)
            return np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce high-frequency noise using Gaussian blur.

        The kernel size is sourced from config.NOISE_KERNEL_SIZE. Gaussian
        blur is preferred over median blur for signatures because it better
        preserves the smooth curves of handwriting strokes.

        Args:
            image: Grayscale input image.

        Returns:
            Denoised grayscale image.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to remove_noise.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            ksize = NOISE_KERNEL_SIZE
            # Kernel size must be odd for Gaussian blur
            if ksize % 2 == 0:
                ksize += 1

            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        except cv2.error as e:
            logger.error("Noise removal failed: %s", e)
            return image.copy() if image is not None else np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Binarize the image using Otsu's automatic thresholding.

        Otsu's method determines an optimal threshold by minimizing
        intra-class variance, making it robust across varying scan
        qualities without manual threshold tuning.

        Args:
            image: Grayscale input image.

        Returns:
            Binary image (pixel values are 0 or 255).
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to apply_threshold.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            # Ensure the image is single-channel
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, thresholded = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresholded
        except cv2.error as e:
            logger.error("Thresholding failed: %s", e)
            return image.copy() if image is not None else np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological opening followed by closing.

        Opening (erosion then dilation) removes small bright spots (noise).
        Closing (dilation then erosion) fills small dark gaps in strokes.
        Together they clean up the binary signature while preserving overall
        stroke geometry.

        Args:
            image: Binary input image.

        Returns:
            Morphologically cleaned binary image.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to morphological_operations.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            ksize = MORPH_KERNEL_SIZE
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (ksize, ksize)
            )
            # Opening removes isolated noise pixels
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            # Closing fills small holes within strokes
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        except cv2.error as e:
            logger.error("Morphological operations failed: %s", e)
            return image.copy() if image is not None else np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel intensities to span the full 0-255 range.

        Uses min-max normalization so that the darkest pixel becomes 0 and
        the brightest becomes 255, maximizing contrast.

        Args:
            image: Input image (any bit depth).

        Returns:
            Normalized uint8 image with values spanning 0-255.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to normalize_image.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            # If the image is uniform (all same value), return as-is
            if image.max() == image.min():
                return image.astype(np.uint8)

            normalized = cv2.normalize(
                image, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            return normalized
        except cv2.error as e:
            logger.error("Image normalization failed: %s", e)
            return image.copy() if image is not None else np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def skeletonize_signature(self, image: np.ndarray) -> np.ndarray:
        """Thin signature strokes to single-pixel width.

        Uses scikit-image's morphological skeletonization which iteratively
        removes boundary pixels while preserving connectivity. This is
        useful for extracting the topological structure of a signature
        independent of pen pressure or stroke width.

        Args:
            image: Binary input image (0/255 values expected).

        Returns:
            Skeletonized binary image (0 or 255 uint8).
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to skeletonize_signature.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            # skimage.skeletonize expects a boolean array where True = foreground
            # Signature strokes are dark (0) on white (255) background in
            # thresholded images, so we invert first.
            binary = (image < GRAYSCALE_THRESHOLD).astype(bool)

            # Guard against fully blank images (no foreground)
            if not binary.any():
                return np.zeros_like(image, dtype=np.uint8)

            skeleton = skimage_skeletonize(binary)
            # Convert boolean skeleton back to uint8 (255 for stroke pixels)
            return (skeleton.astype(np.uint8)) * 255
        except Exception as e:
            logger.error("Skeletonization failed: %s", e)
            return image.copy() if image is not None else np.zeros(
                (self.target_size[1], self.target_size[0]), dtype=np.uint8
            )

    def crop_borders(self, image: np.ndarray) -> np.ndarray:
        """Crop empty borders around the signature content and resize back.

        Finds the bounding box of all non-zero (or non-white) pixels, crops
        to that region, then resizes the result back to target_size so
        output dimensions remain consistent across all signatures.

        Args:
            image: Grayscale or binary input image.

        Returns:
            Cropped and resized image matching target_size.
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image passed to crop_borders.")
                return np.zeros(
                    (self.target_size[1], self.target_size[0]), dtype=np.uint8
                )

            # Invert if signature is dark on light background so that
            # findNonZero locates stroke pixels
            if np.mean(image) > 127:
                work = cv2.bitwise_not(image)
            else:
                work = image.copy()

            # Locate non-zero (foreground) pixels
            coords = cv2.findNonZero(work)
            if coords is None:
                # No content found – return blank image at target size
                logger.warning("No content found during border cropping.")
                return self.resize_image(image)

            x, y, w, h = cv2.boundingRect(coords)

            # Add a small padding (5% of dimension) to avoid clipping strokes
            pad_x = max(int(w * 0.05), 2)
            pad_y = max(int(h * 0.05), 2)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)

            cropped = image[y:y + h, x:x + w]

            # Resize back to target dimensions for consistency
            return self.resize_image(cropped)
        except cv2.error as e:
            logger.error("Border cropping failed: %s", e)
            return self.resize_image(image)
