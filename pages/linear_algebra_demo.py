"""
Linear Algebra educational demonstration page for AI Signature Verification.

Renders LaTeX mathematical formulas and interactive Plotly visualizations for
matrix representation, vector norms, linear transformations, singular value decomposition (SVD),
and principal component analysis (PCA).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import SUPPORTED_FORMATS, logger
from utils.linear_algebra import LinearAlgebraDemo
from utils.image_processing import ImageProcessor
from utils.feature_extraction import FeatureExtractor
from styles.theme import get_section_header
from components.cards import render_metric_card


def render_linear_algebra() -> None:
    """Render the Linear Algebra educational demonstration page."""
    st.markdown(
        get_section_header(
            "Linear Algebra & Feature Space",
            "Explore the mathematical foundations of signature verification and image decomposition"
        ),
        unsafe_allow_html=True
    )

    demo_engine = LinearAlgebraDemo()
    img_gray = None
    feature_vector = None

    # Try to load existing data from session state
    if "raw_image" in st.session_state and st.session_state["raw_image"] is not None:
        raw_img = st.session_state["raw_image"]
        # Standard processing to get grayscale and feature vector
        processor = ImageProcessor()
        extractor = FeatureExtractor()
        processed = processor.process_signature(raw_img)
        img_gray = processed["cropped"]
        feature_vector = st.session_state.get("feature_vector")
        if feature_vector is None:
            features = extractor.extract_all_features(img_gray)
            feature_vector = features["feature_vector"]
        st.info("Using current signature specimen from session state.")
    else:
        # Fallback file uploader
        uploaded_file = st.file_uploader(
            "Upload a signature to view linear algebra operations",
            type=SUPPORTED_FORMATS,
            key="la_uploader"
        )
        if uploaded_file is not None:
            try:
                from PIL import Image
                pil_image = Image.open(uploaded_file)
                raw_img = np.array(pil_image.convert("RGB"))
                processor = ImageProcessor()
                extractor = FeatureExtractor()
                processed = processor.process_signature(raw_img)
                img_gray = processed["cropped"]
                features = extractor.extract_all_features(img_gray)
                feature_vector = features["feature_vector"]
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if img_gray is not None and feature_vector is not None:
        try:
            # Generate demonstrations dict
            demos = demo_engine.get_all_demonstrations(img_gray, feature_vector)

            # Let the user choose categories of concepts
            tabs = st.tabs([
                "📊 Matrix Representations",
                "📐 Vectors & Norms",
                "🔄 Linear Transformations",
                "🔬 Singular Value Decomposition (SVD)",
                "📈 Dimensionality Reduction (PCA & Rank)"
            ])

            # ──────────────────────────────────────────────────────────────
            # Tab 1: Matrix Representations
            # ──────────────────────────────────────────────────────────────
            with tabs[0]:
                st.subheader("Image Matrix Decomposition")

                # Concept 1: Original Matrix
                d1 = demos["original_matrix"]
                with st.expander("1. Grayscale Image Matrix Representation", expanded=True):
                    st.latex(d1["formula"])
                    st.write(d1["explanation"])
                    
                    # Plot heatmap of 8x8 block
                    z_data = np.array(d1["visualization_data"]["matrix"])
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        colorscale="Blues",
                        text=np.round(z_data, 1),
                        texttemplate="%{text}",
                        showscale=False
                    ))
                    fig.update_layout(
                        title="8x8 Center Pixel Intensity Values (A)",
                        width=450, height=450,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Concept 2: Binary Matrix
                d2 = demos["binary_matrix"]
                with st.expander("2. Binarized Selection Matrix", expanded=False):
                    st.latex(d2["formula"])
                    st.write(d2["explanation"])
                    
                    z_bin = np.array(d2["visualization_data"]["matrix"])
                    fig_bin = go.Figure(data=go.Heatmap(
                        z=z_bin,
                        colorscale="Greys",
                        text=z_bin.astype(int),
                        texttemplate="%{text}",
                        showscale=False
                    ))
                    fig_bin.update_layout(
                        title="8x8 Center Binary Values (B)",
                        width=450, height=450,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig_bin, use_container_width=True)

                # Concept 3: Matrix Dimensions
                d3 = demos["matrix_dimensions"]
                with st.expander("3. Dimensions and Grid Span", expanded=False):
                    st.latex(d3["formula"])
                    st.write(d3["explanation"])
                    r, c = d3["result"]
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        render_metric_card("Matrix Rows (m)", f"{r}", "↕️", "#1B6EF3")
                    with col_m2:
                        render_metric_card("Matrix Columns (n)", f"{c}", "↔️", "#00BCD4")

            # ──────────────────────────────────────────────────────────────
            # Tab 2: Vectors & Norms
            # ──────────────────────────────────────────────────────────────
            with tabs[1]:
                st.subheader("Feature Spaces & Vector Norms")

                # Concept 4: Feature Vector
                d4 = demos["feature_vector"]
                with st.expander("1. High-Dimensional Feature Vector", expanded=True):
                    st.latex(d4["formula"])
                    st.write(d4["explanation"])
                    
                    # Display 1D Feature Vector as a bar chart
                    vec_y = np.array(d4["visualization_data"]["vector"])
                    fig_vec = go.Figure(data=[go.Bar(
                        x=list(range(len(vec_y))),
                        y=vec_y,
                        marker_color="#1B6EF3"
                    )])
                    fig_vec.update_layout(
                        title="256-Dimensional Feature Vector Values",
                        xaxis_title="Feature Dimension Index",
                        yaxis_title="Normalized Value",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig_vec, use_container_width=True)

                # Concept 5 & 9: Norms & Vector Length
                d5 = demos["vector_length"]
                d9 = demos["vector_norm"]
                with st.expander("2. Vector Length & Norm Space", expanded=False):
                    st.latex(d9["formula"])
                    st.write(d9["explanation"])
                    
                    l1, l2, linf = d9["result"]
                    col_n1, col_n2, col_n3 = st.columns(3)
                    with col_n1:
                        render_metric_card("Manhattan Norm (L1)", f"{l1:.3f}", "📏", "#E91E63")
                    with col_n2:
                        render_metric_card("Euclidean Norm (L2)", f"{l2:.3f}", "📏", "#1B6EF3")
                    with col_n3:
                        render_metric_card("Chebyshev Norm (L-inf)", f"{linf:.3f}", "📏", "#9C27B0")

                # Concept 8 & 10: Dot Product & Cosine Similarity
                d8 = demos["dot_product"]
                d10 = demos["cosine_similarity"]
                with st.expander("3. Vector Dot Product & Cosine Similarity", expanded=False):
                    st.latex(d10["formula"])
                    st.write(d10["explanation"])
                    
                    dot_val = d8["result"]
                    cos_sim = d10["result"]
                    
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        render_metric_card("Vector Dot Product", f"{dot_val:.2f}", "🤝", "#4CAF50")
                    with col_p2:
                        render_metric_card("Cosine Similarity (cos θ)", f"{cos_sim * 100:.1f}%", "📐", "#FF9800")

            # ──────────────────────────────────────────────────────────────
            # Tab 3: Linear Transformations
            # ──────────────────────────────────────────────────────────────
            with tabs[2]:
                st.subheader("Coordinate Transformations & Projections")

                d6 = demos["transformation_matrix"]
                d7 = demos["linear_transformation"]

                with st.expander("1. Transformation Operator (Matrix)", expanded=True):
                    st.latex(d6["formula"])
                    st.write(d6["explanation"])
                    
                    # Show transformation matrix
                    trans_mat = np.array(d6["visualization_data"]["matrix"])
                    st.markdown(
                        f"""
                        <div style="background-color:#1E2530; padding:1rem; border-radius:8px; border:1px solid #2D3748; text-align:center;">
                            <span style="font-size:1.15rem; font-family:monospace;">
                                T = [[{trans_mat[0, 0]:.4f}, {trans_mat[0, 1]:.4f}], 
                                     [{trans_mat[1, 0]:.4f}, {trans_mat[1, 1]:.4f}]]
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.expander("2. Visualizing Linear Coordinate Mapping", expanded=True):
                    st.latex(d7["formula"])
                    st.write(d7["explanation"])
                    
                    # Transform points plot
                    t_data = d7["visualization_data"]
                    orig_x, orig_y = t_data["original_x"], t_data["original_y"]
                    tran_x, tran_y = t_data["transformed_x"], t_data["transformed_y"]

                    fig_trans = go.Figure()
                    # Add original points
                    fig_trans.add_trace(go.Scatter(
                        x=orig_x, y=orig_y,
                        mode="lines+markers",
                        name="Original Matrix Coordinates",
                        line=dict(color="#A0AEC0", dash="dash"),
                        marker=dict(size=8)
                    ))
                    # Add transformed points
                    fig_trans.add_trace(go.Scatter(
                        x=tran_x, y=tran_y,
                        mode="lines+markers",
                        name="Transformed Coordinates (Rotated 30°)",
                        line=dict(color="#1B6EF3", width=3),
                        marker=dict(size=10, symbol="square")
                    ))
                    
                    fig_trans.update_layout(
                        title="Linear Transformation Mapping in 2D Space",
                        xaxis=dict(range=[-2, 2], scaleanchor="y", scaleratio=1),
                        yaxis=dict(range=[-2, 2]),
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig_trans, use_container_width=True)

                # Concept 12: Transpose
                d12 = demos["transpose"]
                with st.expander("3. Matrix Transposition", expanded=False):
                    st.latex(d12["formula"])
                    st.write(d12["explanation"])
                    
                    t_orig = np.array(d12["visualization_data"]["original"])
                    t_trans = np.array(d12["visualization_data"]["transpose"])

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        fig_o = go.Figure(data=go.Heatmap(z=t_orig, colorscale="Blues", showscale=False))
                        fig_o.update_layout(title="Original Specimen Block", height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
                        st.plotly_chart(fig_o, use_container_width=True)
                    with col_t2:
                        fig_t = go.Figure(data=go.Heatmap(z=t_trans, colorscale="Blues", showscale=False))
                        fig_t.update_layout(title="Transposed Matrix Block (A^T)", height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
                        st.plotly_chart(fig_t, use_container_width=True)

            # ──────────────────────────────────────────────────────────────
            # Tab 4: Singular Value Decomposition (SVD)
            # ──────────────────────────────────────────────────────────────
            with tabs[3]:
                st.subheader("Singular Value Decomposition (SVD) Matrix Compression")
                
                d14 = demos["svd"]
                st.latex(d14["formula"])
                st.write(d14["explanation"])

                svd_data = d14["visualization_data"]
                s_vals = svd_data["singular_values"]
                orig_block = np.array(svd_data["original"])
                recon_block = np.array(svd_data["reconstructed_top2"])

                # Singular Values Bar Chart
                fig_sv = go.Figure(data=[go.Bar(
                    x=[f"σ{i+1}" for i in range(len(s_vals))],
                    y=s_vals,
                    marker_color="#00BCD4"
                )])
                fig_sv.update_layout(
                    title="Singular Values Spectrum (Σ Diagonal Elements)",
                    xaxis_title="Singular Value Component",
                    yaxis_title="Magnitude",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FFFFFF")
                )
                st.plotly_chart(fig_sv, use_container_width=True)

                # Reconstruction side-by-side comparison
                col_sv1, col_sv2 = st.columns(2)
                with col_sv1:
                    fig_orig = go.Figure(data=go.Heatmap(z=orig_block, colorscale="Blues", showscale=False))
                    fig_orig.update_layout(title="Original Specimen Matrix", height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
                    st.plotly_chart(fig_orig, use_container_width=True)
                with col_sv2:
                    fig_recon = go.Figure(data=go.Heatmap(z=recon_block, colorscale="Blues", showscale=False))
                    fig_recon.update_layout(title="Approximated Matrix (Rank 2 SVD)", height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
                    st.plotly_chart(fig_recon, use_container_width=True)

            # ──────────────────────────────────────────────────────────────
            # Tab 5: Dimensionality Reduction (PCA & Rank)
            # ──────────────────────────────────────────────────────────────
            with tabs[4]:
                st.subheader("Dimensionality Reduction & Matrix Rank")

                # Concept 11: Rank
                d11 = demos["matrix_rank"]
                with st.expander("1. Matrix Rank & Linear Independence", expanded=True):
                    st.latex(d11["formula"])
                    st.write(d11["explanation"])
                    
                    rank_val = d11["result"]
                    total_dim = d11["visualization_data"]["total_dimensions"]
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        render_metric_card("Matrix Rank (k)", f"{rank_val}", "⛓️", "#FF1744")
                    with col_r2:
                        render_metric_card("Total Dimensions", f"{total_dim}", "📦", "#4CAF50")

                # Concept 13: Eigenvalues (Symmetric matrix eigenvalues)
                d13 = demos["eigenvalues"]
                with st.expander("2. Eigenvalues and Eigenvectors", expanded=False):
                    st.latex(d13["formula"])
                    st.write(d13["explanation"])
                    
                    e_vals = d13["visualization_data"]["eigenvalues"]
                    # Draw values
                    fig_ev = go.Figure(data=[go.Bar(
                        x=[f"λ{i+1}" for i in range(len(e_vals))],
                        y=e_vals,
                        marker_color="#9C27B0"
                    )])
                    fig_ev.update_layout(
                        title="Eigenvalues of Symmetric Block Covariance Matrix",
                        xaxis_title="Eigenvalue Component",
                        yaxis_title="Magnitude",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig_ev, use_container_width=True)

                # Concept 15: PCA Variance
                d15 = demos["pca"]
                with st.expander("3. Principal Component Analysis (PCA) Projection", expanded=True):
                    st.latex(d15["formula"])
                    st.write(d15["explanation"])
                    
                    pca_data = d15["visualization_data"]
                    var_exp = pca_data["variance_explained"]
                    cum_var = pca_data["cumulative_variance"]

                    # Draw variance graph
                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(var_exp))],
                        y=var_exp,
                        name="Individual Variance Explained",
                        marker_color="#1B6EF3"
                    ))
                    fig_pca.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(len(cum_var))],
                        y=cum_var,
                        mode="lines+markers",
                        name="Cumulative Variance Explained",
                        line=dict(color="#FF9800", width=2)
                    ))
                    
                    fig_pca.update_layout(
                        title="PCA Scree Plot (Eigenvalue Spectrum Projection)",
                        xaxis_title="Principal Component",
                        yaxis_title="Variance Ratio",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FFFFFF"),
                        legend=dict(x=0.01, y=0.99)
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)

        except Exception as e:
            st.error(f"Error compiling linear algebra visuals: {e}")
            logger.error("Linear algebra demo page crash: %s", e, exc_info=True)
    else:
        st.warning("Please upload a signature image first (either in the 'Upload Signature' page or here) to view the matrix computations.")
