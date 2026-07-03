"""
Verification page for AI Signature Verification & Anti-Forgery System.

Provides the signature comparison and authentication workflow. Users can upload
a query signature, compare it against all stored reference signatures using
multiple distance metrics, and receive a detailed verification verdict.
"""

import time
import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
from PIL import Image

from config import (
    SIMILARITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    SUPPORTED_FORMATS,
    DEFAULT_IMAGE_SIZE,
    logger,
)
from models.database import DatabaseManager
from utils.image_processing import ImageProcessor
from utils.feature_extraction import FeatureExtractor
from utils.similarity import SimilarityEngine
from styles.theme import get_custom_css, get_section_header, get_status_badge
from components.cards import render_kpi_cards, render_metric_card, render_info_card
from components.charts import render_comparison_bar


# ──────────────────────────────────────────────────────────────────────────────
# Helper: styled result cards
# ──────────────────────────────────────────────────────────────────────────────

def _verdict_html(decision: str, similarity: float, confidence: float) -> str:
    """Return HTML block for the large verification verdict banner.

    Args:
        decision: 'Genuine' or 'Forged'.
        similarity: Overall similarity score 0-1.
        confidence: Confidence score 0-1.

    Returns:
        HTML string ready for st.markdown(unsafe_allow_html=True).
    """
    if decision.lower() == "genuine":
        color = "#00C853"
        icon = "✅"
        glow = "rgba(0, 200, 83, 0.25)"
    else:
        color = "#FF1744"
        icon = "🚨"
        glow = "rgba(255, 23, 68, 0.25)"

    return f"""
    <div style="
        background: linear-gradient(135deg, {glow}, transparent);
        border: 2px solid {color};
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 0 30px {glow};
    ">
        <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
        <div style="
            font-size: 36px;
            font-weight: 800;
            color: {color};
            letter-spacing: 2px;
            text-transform: uppercase;
        ">{decision}</div>
        <div style="
            margin-top: 16px;
            display: flex;
            justify-content: center;
            gap: 48px;
        ">
            <div>
                <div style="color: #8899AA; font-size: 13px; text-transform: uppercase;">Similarity</div>
                <div style="font-size: 28px; font-weight: 700; color: #E0E6ED;">{similarity:.1%}</div>
            </div>
            <div>
                <div style="color: #8899AA; font-size: 13px; text-transform: uppercase;">Confidence</div>
                <div style="font-size: 28px; font-weight: 700; color: #E0E6ED;">{confidence:.1%}</div>
            </div>
        </div>
    </div>
    """


def _metric_grid_html(metrics: Dict[str, float]) -> str:
    """Return an HTML grid showing individual distance metrics.

    Args:
        metrics: Dictionary with metric names as keys and float values.

    Returns:
        HTML string for the metrics grid.
    """
    cards = ""
    icons = {
        "cosine_similarity": "📐",
        "euclidean_distance": "📏",
        "manhattan_distance": "🗺️",
        "correlation": "🔗",
    }
    labels = {
        "cosine_similarity": "Cosine Similarity",
        "euclidean_distance": "Euclidean Distance",
        "manhattan_distance": "Manhattan Distance",
        "correlation": "Correlation",
    }
    for key in ["cosine_similarity", "euclidean_distance", "manhattan_distance", "correlation"]:
        value = metrics.get(key, 0.0)
        icon = icons.get(key, "📊")
        label = labels.get(key, key)
        # For distances, lower is better; for similarities, higher is better
        if "distance" in key:
            display = f"{value:.4f}"
        else:
            display = f"{value:.4f}"
        cards += f"""
        <div style="
            background: #1E2530;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(27, 110, 243, 0.15);
            transition: transform 0.2s, box-shadow 0.2s;
        ">
            <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
            <div style="color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: #E0E6ED; margin-top: 6px;">{display}</div>
        </div>
        """
    return f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 16px 0;
    ">
        {cards}
    </div>
    """


def _history_row_html(record: Dict) -> str:
    """Return a styled HTML row for a single verification history entry.

    Args:
        record: Verification record dictionary from the database.

    Returns:
        HTML table-row string.
    """
    decision = record.get("decision", "Unknown")
    if decision.lower() == "genuine":
        badge_color = "#00C853"
    else:
        badge_color = "#FF1744"

    similarity = record.get("similarity_score", 0.0)
    proc_time = record.get("processing_time", 0.0)
    created = record.get("created_at", "N/A")
    method = record.get("method", "multi-metric")

    return f"""
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
        <td style="padding: 12px 16px; color: #8899AA; font-size: 13px;">{created}</td>
        <td style="padding: 12px 16px; color: #E0E6ED; font-weight: 600;">{similarity:.1%}</td>
        <td style="padding: 12px 16px;">
            <span style="
                background: {badge_color}22;
                color: {badge_color};
                padding: 4px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            ">{decision}</span>
        </td>
        <td style="padding: 12px 16px; color: #8899AA; font-size: 13px;">{method}</td>
        <td style="padding: 12px 16px; color: #8899AA; font-size: 13px;">{proc_time:.3f}s</td>
    </tr>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Main render function
# ──────────────────────────────────────────────────────────────────────────────

def render_verification() -> None:
    """Render the Signature Verification page.

    Workflow
    --------
    1. Accept a query signature (from session state or file upload).
    2. Compare against all stored reference signatures.
    3. Display detailed verification results with metrics and charts.
    4. Persist the result and show verification history.
    """
    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        get_section_header(
            "Signature Verification",
            subtitle="Compare & Authenticate Signatures",
        ),
        unsafe_allow_html=True,
    )

    # ── Initialize services ──────────────────────────────────────────────────
    db = DatabaseManager()
    processor = ImageProcessor()
    extractor = FeatureExtractor()
    engine = SimilarityEngine()

    # ── Obtain the query signature ───────────────────────────────────────────
    feature_vector: Optional[np.ndarray] = None
    preview_image: Optional[np.ndarray] = None

    # Check if a processed signature already lives in session state
    if (
        "processed_steps" in st.session_state
        and "extracted_features" in st.session_state
        and "feature_vector" in st.session_state
    ):
        feature_vector = st.session_state["feature_vector"]
        # Use the first available processed image for preview
        steps: Dict = st.session_state["processed_steps"]
        preview_image = steps.get("normalized", steps.get("grayscale", steps.get("original")))

        st.markdown(
            """<div style="
                background: #1E2530;
                border-radius: 12px;
                padding: 16px;
                border-left: 4px solid #1B6EF3;
                margin-bottom: 16px;
            ">
                <span style="color:#1B6EF3; font-weight:600;">ℹ️ Using signature from current processing session.</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Show the preview
        if preview_image is not None:
            col_preview, col_info = st.columns([1, 2])
            with col_preview:
                st.image(preview_image, caption="Query Signature", use_container_width=True)
            with col_info:
                st.markdown(
                    f"""<div style="background:#1E2530; border-radius:12px; padding:20px;">
                        <div style="color:#8899AA; font-size:13px;">FEATURE VECTOR</div>
                        <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Dimension: <b>{len(feature_vector)}</b></div>
                        <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Mean: <b>{np.mean(feature_vector):.4f}</b></div>
                        <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Std: <b>{np.std(feature_vector):.4f}</b></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        # File uploader fallback
        st.markdown(
            """<div style="
                background: #1E2530;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                border: 1px dashed rgba(27,110,243,0.3);
            ">
                <p style="color:#8899AA; margin:0;">Upload a signature image to verify against the stored database.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload Signature Image",
            type=SUPPORTED_FORMATS,
            key="verification_uploader",
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        )

        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_array = np.array(pil_image)

                # Process the image
                processed = processor.process_signature(image_array)
                features = extractor.extract_all_features(processed)
                feature_vector = extractor.generate_feature_vector(features)
                preview_image = image_array

                # Persist in session state for reuse
                st.session_state["processed_steps"] = processed
                st.session_state["extracted_features"] = features
                st.session_state["feature_vector"] = feature_vector

                # Show preview
                col_preview, col_info = st.columns([1, 2])
                with col_preview:
                    st.image(image_array, caption="Uploaded Signature", use_container_width=True)
                with col_info:
                    st.markdown(
                        f"""<div style="background:#1E2530; border-radius:12px; padding:20px;">
                            <div style="color:#8899AA; font-size:13px;">IMAGE INFO</div>
                            <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Size: <b>{pil_image.size[0]}×{pil_image.size[1]}</b></div>
                            <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Format: <b>{uploaded_file.type}</b></div>
                            <div style="color:#8899AA; font-size:13px; margin-top:12px;">FEATURE VECTOR</div>
                            <div style="color:#E0E6ED; font-size:15px; margin-top:4px;">Dimension: <b>{len(feature_vector)}</b></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                logger.error("Failed to process uploaded signature: %s", exc)
                st.error(f"Failed to process the uploaded image: {exc}")
                return
        else:
            st.info("👆 Please upload a signature image to begin verification.")
            return

    # ── Load stored signatures ───────────────────────────────────────────────
    try:
        stored_signatures: List[Dict] = db.get_all_signatures()
    except Exception as exc:
        logger.error("Failed to load stored signatures: %s", exc)
        st.error(f"Database error while loading signatures: {exc}")
        return

    if not stored_signatures:
        st.warning(
            "⚠️ No reference signatures in the database. "
            "Please add signatures via the **Upload & Process** page before verifying."
        )
        return

    st.markdown(
        f"""<div style="
            background: #1E2530;
            border-radius: 12px;
            padding: 14px 20px;
            margin: 12px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <span style="font-size:20px;">📂</span>
            <span style="color:#E0E6ED;">Database contains <b>{len(stored_signatures)}</b> reference signature(s)</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Verify button ────────────────────────────────────────────────────────
    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    verify_clicked = st.button(
        "🔍  Verify Signature",
        use_container_width=True,
        type="primary",
        key="btn_verify",
    )

    if verify_clicked and feature_vector is not None:
        with st.spinner("Analyzing signature…"):
            start_time = time.time()

            try:
                # Build list of (id, vector) tuples for comparison
                stored_vectors: List[Tuple[int, np.ndarray]] = []
                for sig in stored_signatures:
                    vec = sig.get("feature_vector")
                    if vec is not None:
                        if isinstance(vec, (bytes, str)):
                            vec = np.frombuffer(
                                vec if isinstance(vec, bytes) else vec.encode("latin-1"),
                                dtype=np.float64,
                            )
                        stored_vectors.append((sig["id"], np.asarray(vec, dtype=np.float64)))

                if not stored_vectors:
                    st.error("No valid feature vectors found in the database.")
                    return

                # Find the most similar signature
                best_id, comparison = engine.find_most_similar(
                    np.asarray(feature_vector, dtype=np.float64),
                    stored_vectors,
                )

                processing_time = time.time() - start_time

                decision: str = comparison.get("decision", "Forged")
                overall_sim: float = comparison.get("overall_similarity", 0.0)
                confidence: float = comparison.get("confidence", 0.0)

            except Exception as exc:
                logger.error("Verification failed: %s", exc)
                st.error(f"Verification error: {exc}")
                return

        # ── Display results ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            """<div style="color:#E0E6ED; font-size:22px; font-weight:700; margin-bottom:8px;">
                📋 Verification Results
            </div>""",
            unsafe_allow_html=True,
        )

        # Verdict banner
        st.markdown(_verdict_html(decision, overall_sim, confidence), unsafe_allow_html=True)

        # Side-by-side comparison with best match
        best_signature = db.get_signature_by_id(best_id)
        if best_signature:
            st.markdown(
                """<div style="color:#E0E6ED; font-size:16px; font-weight:600; margin:20px 0 8px;">
                    🔎 Most Similar Reference Signature
                </div>""",
                unsafe_allow_html=True,
            )
            col_query, col_match = st.columns(2)
            with col_query:
                st.markdown(
                    """<div style="background:#1E2530; border-radius:12px; padding:12px; text-align:center;">
                        <div style="color:#8899AA; font-size:12px; text-transform:uppercase; margin-bottom:8px;">Query Signature</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if preview_image is not None:
                    st.image(preview_image, use_container_width=True)
            with col_match:
                st.markdown(
                    f"""<div style="background:#1E2530; border-radius:12px; padding:12px; text-align:center;">
                        <div style="color:#8899AA; font-size:12px; text-transform:uppercase; margin-bottom:8px;">Best Match — {best_signature.get('user_name', 'Unknown')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                match_path = best_signature.get("image_path", "")
                if match_path and os.path.exists(match_path):
                    st.image(match_path, use_container_width=True)
                else:
                    st.info("Reference image file not found.")

        # Metrics grid
        st.markdown(_metric_grid_html(comparison), unsafe_allow_html=True)

        # Comparison bar chart
        chart_metrics = {
            "Cosine Similarity": comparison.get("cosine_similarity", 0.0),
            "Correlation": comparison.get("correlation", 0.0),
            "Overall Similarity": overall_sim,
            "Confidence": confidence,
        }
        render_comparison_bar(chart_metrics)

        # Processing time footer
        st.markdown(
            f"""<div style="
                background: #1E2530;
                border-radius: 12px;
                padding: 14px 20px;
                margin-top: 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color:#8899AA;">⏱️ Processing Time</span>
                <span style="color:#E0E6ED; font-weight:700;">{processing_time:.3f} seconds</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Persist result ───────────────────────────────────────────────────
        try:
            uploaded_path = st.session_state.get("last_uploaded_path", "session_upload")
            db.add_verification(
                signature_id=best_id,
                uploaded_image_path=str(uploaded_path),
                similarity_score=float(overall_sim),
                decision=decision,
                method="multi-metric",
                processing_time=float(processing_time),
            )
            logger.info(
                "Verification saved — decision=%s, similarity=%.4f, time=%.3fs",
                decision,
                overall_sim,
                processing_time,
            )
        except Exception as exc:
            logger.warning("Could not save verification result: %s", exc)

    # ── Verification History ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """<div style="color:#E0E6ED; font-size:20px; font-weight:700; margin-bottom:12px;">
            📜 Verification History
        </div>""",
        unsafe_allow_html=True,
    )

    try:
        history: List[Dict] = db.get_verification_history()
    except Exception as exc:
        logger.error("Failed to load verification history: %s", exc)
        st.error(f"Could not load history: {exc}")
        history = []

    if not history:
        st.markdown(
            """<div style="
                background: #1E2530;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                color: #8899AA;
            ">
                No verification records yet. Run your first verification above!
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Show latest 20
        recent = history[:20]
        rows_html = "".join(_history_row_html(r) for r in recent)
        table_html = f"""
        <div style="overflow-x: auto; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
            <table style="width: 100%; border-collapse: collapse; background: #1E2530;">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(27,110,243,0.3);">
                        <th style="padding: 14px 16px; text-align: left; color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Date</th>
                        <th style="padding: 14px 16px; text-align: left; color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Similarity</th>
                        <th style="padding: 14px 16px; text-align: left; color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Decision</th>
                        <th style="padding: 14px 16px; text-align: left; color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Method</th>
                        <th style="padding: 14px 16px; text-align: left; color: #8899AA; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Time</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        if len(history) > 20:
            st.caption(f"Showing 20 of {len(history)} total records.")
