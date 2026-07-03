"""
Signature upload and registration page for AI Signature Verification System.

Handles drag-and-drop file upload, displays image previews and metadata,
triggers image preprocessing and feature extraction, and registers signatures
into the database.
"""

import os
import time
from PIL import Image
import numpy as np
import streamlit as st
from config import SUPPORTED_FORMATS, UPLOAD_DIR, logger
from utils.image_processing import ImageProcessor
from utils.feature_extraction import FeatureExtractor
from models.database import DatabaseManager
from styles.theme import get_section_header
from components.cards import render_metric_card, render_info_card
from components.charts import render_feature_radar, render_projection_chart


def render_upload() -> None:
    """Render the Upload Signature and registration page."""
    # Section Title
    st.markdown(
        get_section_header(
            "Upload Signature",
            "Register signature specimens or analyze features in real-time"
        ),
        unsafe_allow_html=True
    )

    # Database instance
    db = DatabaseManager()

    # Drag and Drop File Uploader
    uploaded_file = st.file_uploader(
        "Drag and drop your signature image here",
        type=SUPPORTED_FORMATS,
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )

    if uploaded_file is not None:
        try:
            # Load image using PIL
            pil_image = Image.open(uploaded_file)
            img_format = pil_image.format
            img_size_kb = len(uploaded_file.getvalue()) / 1024.0

            # Convert PIL image to numpy array (RGB) for processing
            img_rgb = np.array(pil_image.convert("RGB"))
            img_width, img_height = pil_image.size

            # Create columns for image preview and metadata
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.subheader("Signature Preview")
                st.image(pil_image, use_container_width=True, caption=uploaded_file.name)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.subheader("Image Information")
                
                # Info items
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Format:** {img_format}")
                st.write(f"**Dimensions:** {img_width} x {img_height} pixels")
                st.write(f"**File Size:** {img_size_kb:.2f} KB")

                # Registration form
                st.markdown("<hr style='margin: 1rem 0; border-top: 1px solid #2D3748;'>", unsafe_allow_html=True)
                st.subheader("Signature Specimen Registry")
                
                user_name = st.text_input(
                    "Assign User Name (Required for database registration)",
                    placeholder="e.g., John Doe"
                )
                notes = st.text_area(
                    "Optional Specimen Notes",
                    placeholder="e.g., Reference specimen collected on 2026-07-03",
                    height=80
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Control buttons
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns([1, 1])

            # Instantiate processing objects
            processor = ImageProcessor()
            extractor = FeatureExtractor()

            # Session state initialization for results
            if "processed_steps" not in st.session_state:
                st.session_state["processed_steps"] = None
            if "extracted_features" not in st.session_state:
                st.session_state["extracted_features"] = None
            if "feature_vector" not in st.session_state:
                st.session_state["feature_vector"] = None
            if "upload_success" not in st.session_state:
                st.session_state["upload_success"] = False

            with btn_col1:
                # Trigger analysis
                if st.button("🔬 Analyze Signature features", use_container_width=True):
                    with st.spinner("Executing image processing pipeline..."):
                        t_start = time.time()
                        
                        # Process image
                        processed_steps = processor.process_signature(img_rgb)
                        # Extract features
                        features = extractor.extract_all_features(processed_steps["cropped"])
                        
                        t_elapsed = time.time() - t_start
                        
                        # Save results to session state
                        st.session_state["processed_steps"] = processed_steps
                        st.session_state["extracted_features"] = features
                        st.session_state["feature_vector"] = features["feature_vector"]
                        st.session_state["analysis_time"] = t_elapsed
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.session_state["raw_image"] = img_rgb
                        
                        st.success(f"Analysis completed in {t_elapsed:.3f} seconds!")

            with btn_col2:
                # Save to database
                if st.button("💾 Register in Signature Database", use_container_width=True):
                    if not user_name.strip():
                        st.error("Please enter a user name before saving to the database.")
                    else:
                        with st.spinner("Processing and registering signature specimen..."):
                            try:
                                # Ensure we have processed data
                                if st.session_state["feature_vector"] is None or st.session_state["processed_steps"] is None:
                                    # Perform on-the-fly analysis
                                    processed_steps = processor.process_signature(img_rgb)
                                    features = extractor.extract_all_features(processed_steps["cropped"])
                                    st.session_state["processed_steps"] = processed_steps
                                    st.session_state["extracted_features"] = features
                                    st.session_state["feature_vector"] = features["feature_vector"]

                                # Save the uploaded file to UPLOAD_DIR
                                timestamp = int(time.time())
                                safe_username = "".join([c if c.isalnum() else "_" for c in user_name])
                                filename = f"{safe_username}_{timestamp}_{uploaded_file.name}"
                                save_path = os.path.join(UPLOAD_DIR, filename)
                                
                                # Save the image
                                pil_image.save(save_path)

                                # Register in database
                                record_id = db.add_signature(
                                    user_name=user_name.strip(),
                                    image_path=save_path,
                                    feature_vector=st.session_state["feature_vector"],
                                    notes=notes.strip()
                                )

                                st.success(f"Signature registered successfully! Specimen ID: #{record_id}")
                                logger.info("Registered signature for %s. ID: %d", user_name, record_id)
                                st.session_state["upload_success"] = True
                            except Exception as db_err:
                                st.error(f"Failed to register signature in database: {db_err}")
                                logger.error("Specimen registration error: %s", db_err, exc_info=True)

            # Display Analysis Results if available
            if st.session_state["extracted_features"] is not None:
                st.markdown("<br><hr style='border-top: 2px solid #1B6EF3;'>", unsafe_allow_html=True)
                st.subheader("Feature Analysis Results")

                features = st.session_state["extracted_features"]

                # Display features in Metric Cards
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    render_metric_card("Pixel Density", f"{features['pixel_density']*100:.2f}%", "📝", "#1B6EF3")
                with m_col2:
                    render_metric_card("Aspect Ratio", f"{features['aspect_ratio']:.3f}", "📐", "#00BCD4")
                with m_col3:
                    com_y, com_x = features["center_of_mass"]
                    render_metric_card("Center of Mass (X, Y)", f"({com_x:.2f}, {com_y:.2f})", "🎯", "#4CAF50")
                with m_col4:
                    _, _, w, h = features["bounding_box"]
                    render_metric_card("Bounding Box", f"{w}x{h} px", "🔲", "#9C27B0")

                # Visual charts
                c_col1, c_col2 = st.columns([1, 1])

                with c_col1:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.subheader("Normalized Geometric Profile")
                    
                    # Package metrics for radar display
                    radar_data = {
                        "Pixel Density": features["pixel_density"],
                        "Aspect Ratio": min(1.0, features["aspect_ratio"] / 5.0),  # scaled
                        "Center of Mass X": com_x,
                        "Center of Mass Y": com_y,
                        "BBox Width Ratio": features["bounding_box"][2] / DEFAULT_IMAGE_SIZE_W_NORM if 'DEFAULT_IMAGE_SIZE_W_NORM' in locals() else features["bounding_box"][2] / 128.0,
                        "BBox Height Ratio": features["bounding_box"][3] / DEFAULT_IMAGE_SIZE_H_NORM if 'DEFAULT_IMAGE_SIZE_H_NORM' in locals() else features["bounding_box"][3] / 128.0,
                    }
                    render_feature_radar(radar_data)
                    st.markdown('</div>', unsafe_allow_html=True)

                with c_col2:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.subheader("Projection Profiles")
                    render_projection_chart(
                        features["horizontal_projection"],
                        features["vertical_projection"]
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Info about how this feature vector relates to linear algebra
                st.markdown("<br>", unsafe_allow_html=True)
                render_info_card(
                    "Linear Algebra Connection",
                    "The feature vector displayed above represents a point in a 256-dimensional Hilbert space. "
                    "In this system, verifying a signature is equivalent to measuring the similarity of vectors in this high-dimensional space. "
                    "Visit the 'Linear Algebra' and 'Verification' tabs to see how the dot product, vector norms, and cosine similarity "
                    "are used to perform biometric forgery detection.",
                    "📐"
                )

        except Exception as e:
            st.error(f"Error reading or analyzing the uploaded file: {e}")
            logger.error("Error in upload file execution: %s", e, exc_info=True)
    else:
        # Prompt when empty
        st.info("Please upload a signature image to begin feature extraction and registration.")
