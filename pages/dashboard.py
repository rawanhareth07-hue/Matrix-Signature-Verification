"""
Dashboard page for the AI Signature Verification & Anti-Forgery System.

Displays system-wide KPIs, verification statistics, activity timelines,
similarity distributions, and a recent-activity table.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List

from styles.theme import get_section_header, get_status_badge, COLORS
from components.cards import render_kpi_cards
from components.charts import (
    render_verification_pie_chart,
    render_activity_timeline,
    render_similarity_distribution,
)


def _load_stats() -> Dict:
    """Load statistics from the database, returning safe defaults on failure."""
    try:
        from models.database import DatabaseManager

        db = DatabaseManager()
        stats = db.get_statistics()
        if stats is None:
            stats = {}
        return stats
    except Exception:
        return {
            "total_signatures": 0,
            "total_verifications": 0,
            "genuine_count": 0,
            "forged_count": 0,
            "accuracy": 0.0,
            "avg_similarity": 0.0,
            "avg_processing_time": 0.0,
        }


def _load_recent_verifications(limit: int = 10) -> List[Dict]:
    """Fetch the most recent verifications from the database."""
    try:
        from models.database import DatabaseManager

        db = DatabaseManager()
        records = db.get_recent_verifications(limit=limit)
        return records if records else []
    except Exception:
        return []


def _load_activity_timeline() -> List[Dict]:
    """Build a daily activity timeline from the database."""
    try:
        from models.database import DatabaseManager

        db = DatabaseManager()
        timeline = db.get_activity_timeline()
        return timeline if timeline else []
    except Exception:
        return []


def _load_similarity_scores() -> List[float]:
    """Retrieve all similarity scores for the histogram."""
    try:
        from models.database import DatabaseManager

        db = DatabaseManager()
        scores = db.get_similarity_scores()
        return scores if scores else []
    except Exception:
        return []


def render_dashboard() -> None:
    """Render the main dashboard page with KPIs, charts, and recent activity."""
    # ── Section Header ──────────────────────────────────────
    st.markdown(
        get_section_header("Dashboard", "System Overview & Analytics"),
        unsafe_allow_html=True,
    )

    # ── Load Data ───────────────────────────────────────────
    stats = _load_stats()
    timeline = _load_activity_timeline()
    scores = _load_similarity_scores()
    recent = _load_recent_verifications()

    # ── KPI Cards ───────────────────────────────────────────
    render_kpi_cards(stats)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row: Pie + Timeline ──────────────────────────
    col_pie, col_timeline = st.columns(2)

    with col_pie:
        render_verification_pie_chart(
            genuine=stats.get("genuine_count", 0),
            forged=stats.get("forged_count", 0),
        )

    with col_timeline:
        render_activity_timeline(timeline)

    # ── Similarity Distribution ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    render_similarity_distribution(scores)

    # ── Recent Activity Table ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        get_section_header("Recent Activity", "Last 10 verifications"),
        unsafe_allow_html=True,
    )

    if recent:
        rows = []
        for rec in recent:
            rows.append(
                {
                    "Date": rec.get("date", "N/A"),
                    "User": rec.get("user", "Unknown"),
                    "Similarity": f"{rec.get('similarity', 0):.3f}",
                    "Decision": rec.get("decision", "N/A"),
                    "Processing Time": f"{rec.get('processing_time', 0):.3f}s",
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "📭 No verification records yet. "
            "Upload and verify a signature to see activity here."
        )
