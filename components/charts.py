"""
Plotly chart components for the AI Signature Verification System.

All charts follow a consistent dark theme with the application's
blue-accent colour scheme and transparent backgrounds.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List


# ──────────────────────────────────────────────────────────────
# Shared Layout Helpers
# ──────────────────────────────────────────────────────────────
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FFFFFF", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0"),
    ),
)

_AXIS_STYLE = dict(
    gridcolor="rgba(45,55,72,0.5)",
    zerolinecolor="rgba(45,55,72,0.5)",
    tickfont=dict(color="#A0AEC0"),
    title=dict(font=dict(color="#FFFFFF")),
)


def _apply_dark(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply the standard dark layout to a Plotly figure."""
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=16, color="#FFFFFF")),
        height=height,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ──────────────────────────────────────────────────────────────
# Chart Functions
# ──────────────────────────────────────────────────────────────

def render_verification_pie_chart(genuine: int, forged: int) -> None:
    """Render a donut chart showing genuine vs forged distribution.

    Args:
        genuine: Count of genuine verifications.
        forged:  Count of forged verifications.
    """
    if genuine == 0 and forged == 0:
        st.info("No verification data available yet.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Genuine", "Forged"],
                values=[genuine, forged],
                hole=0.55,
                marker=dict(
                    colors=["#1B6EF3", "#FF1744"],
                    line=dict(color="#0E1117", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
                hoverinfo="label+value+percent",
            )
        ]
    )
    _apply_dark(fig, "Verification Results", height=380)
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def render_similarity_distribution(scores: List[float]) -> None:
    """Render a histogram of similarity scores.

    Args:
        scores: List of similarity score values between 0 and 1.
    """
    if not scores:
        st.info("No similarity data available yet.")
        return

    fig = go.Figure(
        data=[
            go.Histogram(
                x=scores,
                nbinsx=25,
                marker=dict(
                    color="rgba(27, 110, 243, 0.7)",
                    line=dict(color="#1B6EF3", width=1),
                ),
                hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    _apply_dark(fig, "Similarity Score Distribution", height=380)
    fig.update_xaxes(title_text="Similarity Score", range=[0, 1])
    fig.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def render_activity_timeline(verifications: List[Dict]) -> None:
    """Render a line chart of verification activity over time.

    Args:
        verifications: List of dicts with at least 'date' and 'count' keys.
    """
    if not verifications:
        st.info("No activity data available yet.")
        return

    dates = [v.get("date", "") for v in verifications]
    counts = [v.get("count", 0) for v in verifications]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=dates,
                y=counts,
                mode="lines+markers",
                line=dict(color="#1B6EF3", width=3, shape="spline"),
                marker=dict(size=8, color="#1B6EF3", line=dict(color="white", width=1)),
                fill="tozeroy",
                fillcolor="rgba(27, 110, 243, 0.1)",
                hovertemplate="Date: %{x}<br>Verifications: %{y}<extra></extra>",
            )
        ]
    )
    _apply_dark(fig, "Verification Activity", height=380)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Verifications")
    st.plotly_chart(fig, use_container_width=True)


def render_feature_radar(features: Dict[str, float]) -> None:
    """Render a radar / spider chart of feature values.

    Args:
        features: Mapping of feature name to normalised value (0-1).
    """
    if not features:
        st.info("No feature data available.")
        return

    categories = list(features.keys())
    values = list(features.values())
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(27, 110, 243, 0.2)",
                line=dict(color="#1B6EF3", width=2),
                marker=dict(size=6, color="#1B6EF3"),
            )
        ]
    )
    _apply_dark(fig, "Feature Profile", height=420)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.15 if values else 1],
                gridcolor="rgba(45,55,72,0.5)",
                tickfont=dict(color="#A0AEC0", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(45,55,72,0.5)",
                tickfont=dict(color="#A0AEC0", size=11),
            ),
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def render_projection_chart(
    horizontal: np.ndarray, vertical: np.ndarray
) -> None:
    """Render bar charts of horizontal and vertical projection histograms.

    Args:
        horizontal: 1-D array – horizontal projection profile.
        vertical:   1-D array – vertical projection profile.
    """
    col1, col2 = st.columns(2)

    with col1:
        fig_h = go.Figure(
            data=[
                go.Bar(
                    y=list(range(len(horizontal))),
                    x=horizontal,
                    orientation="h",
                    marker=dict(color="rgba(27, 110, 243, 0.7)"),
                )
            ]
        )
        _apply_dark(fig_h, "Horizontal Projection", height=350)
        fig_h.update_xaxes(title_text="Pixel Sum")
        fig_h.update_yaxes(title_text="Row", autorange="reversed")
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        fig_v = go.Figure(
            data=[
                go.Bar(
                    x=list(range(len(vertical))),
                    y=vertical,
                    marker=dict(color="rgba(0, 188, 212, 0.7)"),
                )
            ]
        )
        _apply_dark(fig_v, "Vertical Projection", height=350)
        fig_v.update_xaxes(title_text="Column")
        fig_v.update_yaxes(title_text="Pixel Sum")
        st.plotly_chart(fig_v, use_container_width=True)


def render_comparison_bar(metrics: Dict[str, float]) -> None:
    """Render a grouped bar chart of similarity metrics.

    Args:
        metrics: Mapping of metric name to value (0-1).
    """
    if not metrics:
        st.info("No comparison data available.")
        return

    names = list(metrics.keys())
    values = list(metrics.values())

    colors = [
        "#1B6EF3" if v >= 0.75 else "#FFD600" if v >= 0.50 else "#FF1744"
        for v in values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker=dict(color=colors, line=dict(color="white", width=0.5)),
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                textfont=dict(color="white", size=12),
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        ]
    )
    _apply_dark(fig, "Similarity Metrics Comparison", height=380)
    fig.update_yaxes(title_text="Score", range=[0, 1])
    fig.update_xaxes(title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)


def render_processing_steps_chart(
    processing_times: Dict[str, float],
) -> None:
    """Render a horizontal bar chart of processing step durations.

    Args:
        processing_times: Mapping of step name to duration in seconds.
    """
    if not processing_times:
        st.info("No processing time data available.")
        return

    steps = list(processing_times.keys())
    times = list(processing_times.values())

    fig = go.Figure(
        data=[
            go.Bar(
                y=steps,
                x=times,
                orientation="h",
                marker=dict(
                    color=times,
                    colorscale=[[0, "#1B6EF3"], [1, "#00BCD4"]],
                    line=dict(color="white", width=0.5),
                ),
                text=[f"{t:.3f}s" for t in times],
                textposition="auto",
                textfont=dict(color="white", size=11),
                hovertemplate="%{y}: %{x:.4f}s<extra></extra>",
            )
        ]
    )
    _apply_dark(fig, "Processing Time by Step", height=max(300, len(steps) * 45))
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True)
