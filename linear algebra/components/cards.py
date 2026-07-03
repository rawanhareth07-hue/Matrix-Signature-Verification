"""
Reusable Streamlit card components for the AI Signature Verification System.

Provides KPI card rows, metric cards, info cards, and comparison cards
built on the design system defined in styles.theme.
"""

import streamlit as st
from typing import Any, Dict, Optional

from styles.theme import get_card_html, get_section_header, COLORS


def render_kpi_cards(stats: Dict) -> None:
    """Render a responsive row of KPI cards from a statistics dictionary.

    Args:
        stats: Dictionary with keys – total_signatures, total_verifications,
               genuine_count, forged_count, avg_similarity,
               avg_processing_time, accuracy.
    """
    total_sigs = stats.get("total_signatures", 0)
    total_veri = stats.get("total_verifications", 0)
    genuine = stats.get("genuine_count", 0)
    forged = stats.get("forged_count", 0)
    accuracy = stats.get("accuracy", 0.0)
    avg_sim = stats.get("avg_similarity", 0.0)
    avg_time = stats.get("avg_processing_time", 0.0)

    cols = st.columns(4)

    with cols[0]:
        st.markdown(
            get_card_html(
                title="Total Signatures",
                value=str(total_sigs),
                icon="📝",
                color=COLORS["primary"],
            ),
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            get_card_html(
                title="Verifications",
                value=str(total_veri),
                icon="✅",
                color=COLORS["success"],
            ),
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            get_card_html(
                title="Genuine",
                value=str(genuine),
                icon="🛡️",
                color=COLORS["success"],
            ),
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            get_card_html(
                title="Forged",
                value=str(forged),
                icon="⚠️",
                color=COLORS["danger"],
            ),
            unsafe_allow_html=True,
        )

    # Second row
    cols2 = st.columns(3)

    with cols2[0]:
        st.markdown(
            get_card_html(
                title="Accuracy",
                value=f"{accuracy:.1f}%",
                icon="📊",
                color=COLORS["primary"],
            ),
            unsafe_allow_html=True,
        )

    with cols2[1]:
        st.markdown(
            get_card_html(
                title="Avg Similarity",
                value=f"{avg_sim:.2f}",
                icon="🎯",
                color="#00BCD4",
            ),
            unsafe_allow_html=True,
        )

    with cols2[2]:
        st.markdown(
            get_card_html(
                title="Avg Processing",
                value=f"{avg_time:.2f}s",
                icon="⚡",
                color=COLORS["warning"],
            ),
            unsafe_allow_html=True,
        )


def render_metric_card(
    title: str,
    value: Any,
    icon: str,
    color: str,
    delta: Optional[str] = None,
) -> None:
    """Render a single metric card.

    Args:
        title: Metric label.
        value: Display value.
        icon:  Emoji or icon.
        color: Accent colour (hex).
        delta: Optional change indicator.
    """
    st.markdown(
        get_card_html(
            title=title,
            value=str(value),
            icon=icon,
            color=color,
            delta=delta,
        ),
        unsafe_allow_html=True,
    )


def render_info_card(
    title: str, content: str, icon: str = "ℹ️"
) -> None:
    """Render an informational card with an accent left border.

    Args:
        title:   Card heading.
        content: Body text (may contain HTML).
        icon:    Leading emoji.
    """
    html = f"""
    <div class="info-card">
        <div class="info-title">{icon} {title}</div>
        <div class="info-content">{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_comparison_card(
    label: str,
    value1: Any,
    value2: Any,
    name1: str,
    name2: str,
) -> None:
    """Render a side-by-side comparison card.

    Args:
        label:  Comparison category name.
        value1: Left-side value.
        value2: Right-side value.
        name1:  Left-side label.
        name2:  Right-side label.
    """
    html = f"""
    <div class="comparison-card">
        <div class="comparison-label">{label}</div>
        <div class="comparison-values">
            <div class="comparison-item">
                <div class="value">{value1}</div>
                <div class="name">{name1}</div>
            </div>
            <div class="comparison-vs">VS</div>
            <div class="comparison-item">
                <div class="value">{value2}</div>
                <div class="name">{name2}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
