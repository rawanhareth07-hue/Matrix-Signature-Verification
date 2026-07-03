"""
Theme and styling module for the AI Signature Verification & Anti-Forgery System.

Provides enterprise-grade CSS styling, KPI card templates, status badges,
and section headers for a premium dark-mode Streamlit application.
"""

import streamlit as st
from typing import Optional

from config import APP_NAME, APP_VERSION, PAGE_ICON, SIDEBAR_STATE


# ──────────────────────────────────────────────────────────────
# Design Tokens
# ──────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B6EF3",
    "secondary": "#0D47A1",
    "background": "#0E1117",
    "card_bg": "#1E2530",
    "card_border": "#2D3748",
    "success": "#00C853",
    "danger": "#FF1744",
    "warning": "#FFD600",
    "text_primary": "#FFFFFF",
    "text_secondary": "#A0AEC0",
    "accent_start": "#1B6EF3",
    "accent_end": "#00BCD4",
}


def get_custom_css() -> str:
    """Return enterprise-grade CSS for the Streamlit application.

    Returns:
        str: A large CSS block covering KPI cards, sidebar, buttons,
             tabs, expanders, file uploaders, tables, animations,
             responsive breakpoints, scrollbar, status badges,
             section headers, and progress bars.
    """
    return f"""
    <style>
    /* ── Global Reset & Base ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ── Animations ──────────────────────────────────────────── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50%      {{ transform: scale(1.05); }}
    }}

    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    @keyframes gradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* ── KPI / Glassmorphism Cards ───────────────────────────── */
    .kpi-card {{
        background: linear-gradient(135deg, rgba(30, 37, 48, 0.95), rgba(30, 37, 48, 0.80));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(45, 55, 72, 0.6);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['accent_start']}, {COLORS['accent_end']});
        border-radius: 16px 16px 0 0;
    }}

    .kpi-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(27, 110, 243, 0.20);
        border-color: {COLORS['primary']};
    }}

    .kpi-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }}

    .kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
        margin: 0.25rem 0;
        letter-spacing: -0.5px;
    }}

    .kpi-title {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .kpi-delta {{
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.4rem;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }}

    .kpi-delta.positive {{
        color: {COLORS['success']};
        background: rgba(0, 200, 83, 0.12);
    }}

    .kpi-delta.negative {{
        color: {COLORS['danger']};
        background: rgba(255, 23, 68, 0.12);
    }}

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0E1117 0%, #151B26 100%);
        border-right: 1px solid {COLORS['card_border']};
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 {{
        background: linear-gradient(135deg, {COLORS['accent_start']}, {COLORS['accent_end']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    section[data-testid="stSidebar"] .stRadio > label {{
        color: {COLORS['text_secondary']};
        font-weight: 500;
    }}

    /* ── Metric Cards ──────────────────────────────────────── */
    .metric-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 12px;
        padding: 1.25rem;
        animation: slideUp 0.5s ease-out;
    }}

    .metric-card .icon-container {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
    }}

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(27, 110, 243, 0.35);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ── Tabs ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 10px;
        padding: 10px 20px;
        color: {COLORS['text_secondary']};
        font-weight: 500;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']}) !important;
        color: white !important;
        border-color: {COLORS['primary']} !important;
    }}

    /* ── Expanders ─────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 12px;
        font-weight: 600;
        color: {COLORS['text_primary']};
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS['primary']};
        background: rgba(27, 110, 243, 0.08);
    }}

    .streamlit-expanderContent {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-top: none;
        border-radius: 0 0 12px 12px;
    }}

    /* ── File Uploader ─────────────────────────────────────── */
    .stFileUploader {{
        background: {COLORS['card_bg']};
        border: 2px dashed {COLORS['card_border']};
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }}

    .stFileUploader:hover {{
        border-color: {COLORS['primary']};
        background: rgba(27, 110, 243, 0.04);
    }}

    /* ── Tables / Dataframes ──────────────────────────────── */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    .stDataFrame table {{
        border-collapse: separate;
        border-spacing: 0;
    }}

    .stDataFrame thead th {{
        background: {COLORS['card_bg']} !important;
        color: {COLORS['text_secondary']} !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        padding: 12px 16px !important;
        border-bottom: 2px solid {COLORS['primary']} !important;
    }}

    .stDataFrame tbody td {{
        background: rgba(30, 37, 48, 0.5) !important;
        color: {COLORS['text_primary']} !important;
        padding: 10px 16px !important;
        border-bottom: 1px solid {COLORS['card_border']} !important;
    }}

    .stDataFrame tbody tr:hover td {{
        background: rgba(27, 110, 243, 0.08) !important;
    }}

    /* ── Status Badges ─────────────────────────────────────── */
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .status-badge.genuine {{
        background: rgba(0, 200, 83, 0.15);
        color: {COLORS['success']};
        border: 1px solid rgba(0, 200, 83, 0.3);
    }}

    .status-badge.forged {{
        background: rgba(255, 23, 68, 0.15);
        color: {COLORS['danger']};
        border: 1px solid rgba(255, 23, 68, 0.3);
    }}

    .status-badge.pending {{
        background: rgba(255, 214, 0, 0.15);
        color: {COLORS['warning']};
        border: 1px solid rgba(255, 214, 0, 0.3);
    }}

    /* ── Section Headers ──────────────────────────────────── */
    .section-header {{
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        position: relative;
        animation: fadeIn 0.5s ease-out;
    }}

    .section-header::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['accent_start']}, {COLORS['accent_end']});
        border-radius: 3px;
    }}

    .section-header h2 {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .section-header p {{
        font-size: 0.9rem;
        color: {COLORS['text_secondary']};
        margin: 0.25rem 0 0 0;
    }}

    /* ── Progress Bars ─────────────────────────────────────── */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {COLORS['accent_start']}, {COLORS['accent_end']});
        border-radius: 10px;
    }}

    .custom-progress {{
        background: {COLORS['card_border']};
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}

    .custom-progress-bar {{
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, {COLORS['accent_start']}, {COLORS['accent_end']});
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* ── Custom Scrollbar ──────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['card_border']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary']};
    }}

    /* ── Info Card ─────────────────────────────────────────── */
    .info-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        animation: fadeIn 0.4s ease-out;
        border-left: 4px solid {COLORS['primary']};
    }}

    .info-card .info-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 0.5rem;
    }}

    .info-card .info-content {{
        font-size: 0.85rem;
        color: {COLORS['text_secondary']};
        line-height: 1.6;
    }}

    /* ── Comparison Card ──────────────────────────────────── */
    .comparison-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        animation: fadeIn 0.5s ease-out;
    }}

    .comparison-label {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }}

    .comparison-values {{
        display: flex;
        justify-content: space-around;
        align-items: center;
    }}

    .comparison-item {{
        text-align: center;
    }}

    .comparison-item .value {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
    }}

    .comparison-item .name {{
        font-size: 0.75rem;
        color: {COLORS['text_secondary']};
        margin-top: 0.25rem;
    }}

    .comparison-vs {{
        font-size: 0.75rem;
        font-weight: 700;
        color: {COLORS['primary']};
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(27, 110, 243, 0.12);
    }}

    /* ── Processing Step Card ──────────────────────────────── */
    .step-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.4s ease-out;
    }}

    .step-card .step-name {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 0.5rem;
    }}

    .step-card .step-desc {{
        font-size: 0.8rem;
        color: {COLORS['text_secondary']};
        line-height: 1.5;
    }}

    .step-card .step-stats {{
        font-size: 0.75rem;
        color: {COLORS['primary']};
        margin-top: 0.5rem;
        font-weight: 500;
    }}

    /* ── LaTeX / Math Styling ──────────────────────────────── */
    .math-card {{
        background: linear-gradient(135deg, rgba(30, 37, 48, 0.95), rgba(14, 17, 23, 0.95));
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {COLORS['primary']};
    }}

    .math-card h3 {{
        color: {COLORS['text_primary']};
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    .math-card .explanation {{
        color: {COLORS['text_secondary']};
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    /* ── Responsive Design ─────────────────────────────────── */
    @media (max-width: 768px) {{
        .kpi-card {{
            padding: 1rem;
        }}
        .kpi-value {{
            font-size: 1.3rem;
        }}
        .kpi-title {{
            font-size: 0.7rem;
        }}
        .section-header h2 {{
            font-size: 1.35rem;
        }}
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
    }}

    @media (max-width: 480px) {{
        .kpi-card {{
            padding: 0.75rem;
        }}
        .kpi-value {{
            font-size: 1.1rem;
        }}
    }}
    </style>
    """


def apply_theme() -> None:
    """Apply the custom CSS theme and set Streamlit page configuration.

    Configures the page with the app title, icon, layout, and sidebar
    state, then injects the enterprise-grade CSS into the page.
    """
    try:
        st.set_page_config(
            page_title=APP_NAME,
            page_icon=PAGE_ICON,
            layout="wide",
            initial_sidebar_state=SIDEBAR_STATE,
        )
    except st.errors.StreamlitAPIException:
        # Page config can only be set once; ignore if already set.
        pass

    st.markdown(get_custom_css(), unsafe_allow_html=True)


def get_card_html(
    title: str,
    value: str,
    icon: str,
    color: str = "#1B6EF3",
    delta: Optional[str] = None,
) -> str:
    """Generate HTML for a KPI card with icon, value, and optional delta.

    Args:
        title: Card heading label.
        value: Primary metric value to display.
        icon:  Emoji or icon character.
        color: Accent colour for the card (CSS hex).
        delta: Optional change indicator (e.g. '+5%').

    Returns:
        str: Complete HTML string for the KPI card.
    """
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta.startswith("+") else "negative"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    return f"""
    <div class="kpi-card" style="border-top: 3px solid {color};">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-title">{title}</div>
        {delta_html}
    </div>
    """


def get_status_badge(status: str) -> str:
    """Generate HTML for a coloured status badge.

    Args:
        status: One of 'genuine', 'forged', or 'pending'.

    Returns:
        str: HTML span element styled as a status badge.
    """
    css_class = status.lower()
    if css_class not in ("genuine", "forged", "pending"):
        css_class = "pending"
    label = status.capitalize()
    return f'<span class="status-badge {css_class}">{label}</span>'


def get_section_header(title: str, subtitle: str = "") -> str:
    """Generate HTML for a section header with gradient underline.

    Args:
        title:    Main heading text.
        subtitle: Optional description shown below the heading.

    Returns:
        str: HTML block for the section header.
    """
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    return f"""
    <div class="section-header">
        <h2>{title}</h2>
        {subtitle_html}
    </div>
    """
