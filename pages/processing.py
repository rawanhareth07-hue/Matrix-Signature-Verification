"""
Image processing pipeline visualization page for AI Signature Verification.

Exposes every mathematical step of the OpenCV image preprocessing pipeline
(Grayscale conversion, Gaussian blurring, Otsu thresholding, Morphological close/open,
normalization, skeletonization, cropping).
"""

import time
import numpy as np
import streamlit as st
from config import SUPPORTED_FORMATS, logger
from utils.image_processing import ImageProcessor
from styles.theme import get_section_header
from components.cards import render_metric_card


def render_processing() -> None:
    """Render the Image Processing Pipeline visualization page."""
    st.markdown(
        get_section_header(
            "Image Processing Pipeline",
            "Deconstruct and visualize each mathematical step of the signature normalization flow"
        ),
        unsafe_allow_html=True
    )

    processor = ImageProcessor()
    img_rgb = None

    # Retrieve image from session state if uploaded
    if "raw_image" in st.session_state and st.session_state["raw_image"] is not None:
        img_rgb = st.session_state["raw_image"]
        st.info("Using current image uploaded from the 'Upload Signature' page.")
    else:
        # Fallback file uploader directly on the processing page
        uploaded_file = st.file_uploader(
            "Upload a signature to view the step-by-step pipeline",
            type=SUPPORTED_FORMATS,
            key="processing_uploader"
        )
        if uploaded_file is not None:
            try:
                from PIL import Image
                pil_image = Image.open(uploaded_file)
                img_rgb = np.array(pil_image.convert("RGB"))
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if img_rgb is not None:
        try:
            # Measure processing latency
            t_start = time.time()
            steps = processor.process_signature(img_rgb)
            latency = time.time() - t_start

            # Pipeline step descriptions
            descriptions = {
                "original": "The raw input signature image in full color (RGB). Represents the initial matrices with dimensions height x width x channels.",
                "grayscale": "Reduces dimensionality by mapping RGB channels into a single intensity channel (luminosity matrix) using standard weights: Y = 0.299R + 0.587G + 0.114B.",
                "resized": "Standardizes the resolution of the signature matrix to 128x128. This step ensures that all signature matrices have uniform size and shape.",
                "denoised": "Applies a Gaussian kernel filter to suppress high-frequency noise and scanning artifacts while retaining edge boundaries.",
                "thresholded": "Converts intensity values to a binary representation (0 or 255) using Otsu's bimodal threshold optimization. Automatically finds the optimal separation boundary.",
                "morphological": "Performs mathematical morphology (dilation and erosion sequences) to fill interior stroke gaps and eliminate disconnected speckles.",
                "normalized": "Rescales pixel intensities to map standard bounds [0, 255] and improves local contrast alignment.",
                "skeletonized": "Thinning algorithm that reduces signature strokes to a single-pixel skeleton width. Retains structural topology, stroke crossings, and geometry.",
                "cropped": "Crops the active bounding box containing foreground pixels and resizes back to 128x128. Removes arbitrary borders, aligning the signature center."
            }

            # Steps order
            steps_order = [
                ("original", "1. Original Image"),
                ("grayscale", "2. Grayscale Intensity Matrix"),
                ("resized", "3. Standardized Resolution Matrix"),
                ("denoised", "4. Gaussian Denoising Filter"),
                ("thresholded", "5. Otsu Binarization (Thresholding)"),
                ("morphological", "6. Morphological Operations"),
                ("normalized", "7. Intensity Normalization"),
                ("skeletonized", "8. Topological Skeletonization"),
                ("cropped", "9. Bounding-Box Centered & Cropped")
            ]

            # Display in a 2-column grid
            for idx in range(0, len(steps_order), 2):
                col1, col2 = st.columns(2)

                # Column 1
                with col1:
                    key1, title1 = steps_order[idx]
                    img1 = steps[key1]
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.subheader(title1)
                    # Convert single-channel to RGB for Streamlit rendering if needed, or use grayscale
                    st.image(img1, use_container_width=True, channels="GRAY" if len(img1.shape) == 2 and key1 != "original" else "RGB")
                    st.markdown(f"<p style='color:#A0AEC0; font-size:0.85rem;'>{descriptions[key1]}</p>", unsafe_allow_html=True)
                    # Print stats
                    h, w = img1.shape[:2]
                    density = np.sum(img1 > 0) / img1.size if key1 != "original" else 1.0
                    st.caption(f"Dimensions: {w}x{h} px | Foreground Density: {density*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Column 2
                with col2:
                    if idx + 1 < len(steps_order):
                        key2, title2 = steps_order[idx + 1]
                        img2 = steps[key2]
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        st.subheader(title2)
                        st.image(img2, use_container_width=True, channels="GRAY" if len(img2.shape) == 2 else "RGB")
                        st.markdown(f"<p style='color:#A0AEC0; font-size:0.85rem;'>{descriptions[key2]}</p>", unsafe_allow_html=True)
                        # Print stats
                        h, w = img2.shape[:2]
                        density = np.sum(img2 > 0) / img2.size
                        st.caption(f"Dimensions: {w}x{h} px | Foreground Density: {density*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

            # Footer Metrics
            st.markdown("<br><hr style='border-top: 1px solid #2D3748;'>", unsafe_allow_html=True)
            st.subheader("Pipeline Execution Summary")
            
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                render_metric_card("Total Pipeline Steps", "9", "🏁", "#1B6EF3")
            with f_col2:
                render_metric_card("Execution Latency", f"{latency:.3f} seconds", "⚡", "#00BCD4")
            with f_col3:
                h_final, w_final = steps["cropped"].shape[:2]
                render_metric_card("Final Matrix Dimension", f"{w_final}x{h_final}", "📐", "#4CAF50")

        except Exception as e:
            st.error(f"Error compiling image processing steps: {e}")
            logger.error("Error in processing page: %s", e, exc_info=True)
    else:
        st.warning("Please upload a signature image first (either here or in the 'Upload Signature' page).")
