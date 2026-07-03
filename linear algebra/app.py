"""
AI Signature Verification & Anti-Forgery System
================================================

Main application entry point.

This module bootstraps the Streamlit application, configures the page,
applies the custom theme, renders the sidebar navigation, and routes
to the selected page.

Usage:
    streamlit run app.py
"""

import streamlit as st
from config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    PAGE_ICON,
    SIDEBAR_STATE,
    logger,
)
from styles.theme import get_custom_css, get_section_header
from models.database import DatabaseManager

# ──────────────────────────────────────────────────────────────
# Page Configuration (MUST be the first Streamlit command)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state=SIDEBAR_STATE,
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"### {APP_NAME}\n{APP_DESCRIPTION}\n\nVersion: {APP_VERSION}",
    },
)

# ──────────────────────────────────────────────────────────────
# Apply Custom Theme CSS
# ──────────────────────────────────────────────────────────────
st.markdown(f"<style>{get_custom_css()}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Initialize Database on Startup
# ──────────────────────────────────────────────────────────────
if "db_initialized" not in st.session_state:
    try:
        db = DatabaseManager()
        db.initialize_database()
        st.session_state["db_initialized"] = True
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)
        st.error(f"Database initialization error: {exc}")

# ──────────────────────────────────────────────────────────────
# Page Registry
# ──────────────────────────────────────────────────────────────
PAGE_MAP = {
    "🏠 Dashboard": "dashboard",
    "📤 Upload Signature": "upload",
    "🔬 Image Processing": "processing",
    "📐 Linear Algebra": "linear_algebra",
    "✅ Verification": "verification",
    "🗄️ Database": "database",
    "📊 Reports": "reports",
    "⚙️ Settings": "settings",
}

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding header
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.25rem;">🛡️</div>
            <h2 style="
                margin: 0;
                background: linear-gradient(135deg, #1B6EF3, #00BCD4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.15rem;
                font-weight: 700;
                line-height: 1.3;
            ">AI Signature<br>Verification System</h2>
            <p style="color: #A0AEC0; font-size: 0.75rem; margin-top: 0.35rem;">
                v{APP_VERSION}
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #2D3748; margin: 0 0 0.75rem 0;">
        """,
        unsafe_allow_html=True,
    )

    # Navigation radio
    selected_label = st.radio(
        "Navigation",
        list(PAGE_MAP.keys()),
        label_visibility="collapsed",
    )
    selected_page = PAGE_MAP[selected_label]

    # Sidebar footer
    st.markdown(
        """
        <hr style="border: none; border-top: 1px solid #2D3748; margin: 1.5rem 0 0.75rem 0;">
        <div style="text-align: center; padding-bottom: 1rem;">
            <p style="color: #A0AEC0; font-size: 0.7rem; margin: 0;">
                Powered by Linear Algebra & AI
            </p>
            <p style="color: #4A5568; font-size: 0.65rem; margin: 0.25rem 0 0 0;">
                © 2026 AI Security Lab
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Page Router
# ──────────────────────────────────────────────────────────────
try:
    if selected_page == "dashboard":
        from pages.dashboard import render_dashboard
        render_dashboard()

    elif selected_page == "upload":
        from pages.upload import render_upload
        render_upload()

    elif selected_page == "processing":
        from pages.processing import render_processing
        render_processing()

    elif selected_page == "linear_algebra":
        from pages.linear_algebra_demo import render_linear_algebra
        render_linear_algebra()

    elif selected_page == "verification":
        from pages.verification import render_verification
        render_verification()

    elif selected_page == "database":
        from pages.database_page import render_database
        render_database()

    elif selected_page == "reports":
        from pages.reports import render_reports
        render_reports()

    elif selected_page == "settings":
        from pages.settings import render_settings
        render_settings()

    else:
        st.error("Unknown page selected.")

except Exception as exc:
    logger.error("Error rendering page '%s': %s", selected_page, exc, exc_info=True)
    st.error(f"An error occurred while loading the page: {exc}")
    st.info("Please try refreshing the page or selecting a different section.")
