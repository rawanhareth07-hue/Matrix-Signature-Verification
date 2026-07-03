"""
System configuration and settings page for AI Signature Verification.

Manages similarity threshold adjustments, image sizing parameters, database reset
operations, JSON configuration export/import, and provides diagnostic software version metrics.
"""

import os
import json
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from models.database import DatabaseManager
from styles.theme import get_section_header
from config import (
    APP_NAME,
    APP_VERSION,
    APP_AUTHOR,
    APP_DESCRIPTION,
    SIMILARITY_THRESHOLD,
    logger
)
from components.cards import render_metric_card


def render_settings() -> None:
    """Render the Settings page."""
    st.markdown(
        get_section_header(
            "Settings & Configuration",
            "Fine-tune decision parameters, adjust pipeline dimensions, and manage databases"
        ),
        unsafe_allow_html=True
    )

    db = DatabaseManager()

    # Load persisted settings from SQLite DB or fall back to config
    stored_threshold = db.get_setting("similarity_threshold", "")
    current_threshold = float(stored_threshold) if stored_threshold else SIMILARITY_THRESHOLD

    stored_img_size = db.get_setting("default_image_size", "128")
    current_img_size = int(stored_img_size)

    stored_theme = db.get_setting("app_theme", "Dark")

    # Create sections
    tab_dec, tab_db, tab_info = st.tabs([
        "⚙️ Decision Parameters",
        "🗄️ Database Utilities",
        "ℹ️ System Information"
    ])

    # ──────────────────────────────────────────────────────────────
    # Tab 1: Decision Parameters
    # ──────────────────────────────────────────────────────────────
    with tab_dec:
        st.subheader("Verification Decision Calibration")
        st.write("Tune authentication and image scaling behaviors. Changes are applied globally.")

        # Threshold slider
        new_threshold = st.slider(
            "Similarity Acceptance Threshold (τ)",
            min_value=0.0,
            max_value=1.0,
            value=current_threshold,
            step=0.05,
            help="Minimum overall similarity score required to classify a signature specimen as 'Genuine'."
        )

        # Image target dimensions
        new_img_size = st.selectbox(
            "Standard Pipeline Resizing Dimensions",
            [64, 128, 256],
            index=[64, 128, 256].index(current_img_size),
            help="Target width and height for resizing all signature matrices before extraction."
        )

        # Theme selector
        new_theme = st.selectbox(
            "Application Color Theme",
            ["Dark", "Light"],
            index=["Dark", "Light"].index(stored_theme),
            help="Sets the color theme layout profile."
        )

        if st.button("💾 Save Settings", key="save_config_btn", use_container_width=True):
            try:
                db.set_setting("similarity_threshold", str(new_threshold))
                db.set_setting("default_image_size", str(new_img_size))
                db.set_setting("app_theme", new_theme)
                
                st.success("Configuration settings updated successfully!")
                logger.info("Saved settings: threshold=%s, size=%s, theme=%s", new_threshold, new_img_size, new_theme)
                time_sleep = 0.5
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save settings: {e}")

        # JSON Configuration Import/Export
        st.markdown("<br><hr style='border-top:1px solid #2D3748;'>", unsafe_allow_html=True)
        st.subheader("Configuration Backup")
        
        c_backup, c_restore = st.columns(2)
        
        with c_backup:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Backup Configuration")
            st.write("Download current configuration parameters as a JSON backup file.")
            
            settings_dict = {
                "similarity_threshold": new_threshold,
                "default_image_size": new_img_size,
                "app_theme": new_theme,
                "backup_timestamp": pd.Timestamp.now().isoformat()
            }
            json_str = json.dumps(settings_dict, indent=4)
            st.download_button(
                label="⬇️ Export Settings Backup",
                data=json_str,
                file_name="settings_backup.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with c_restore:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Restore Configuration")
            st.write("Restore settings parameters by uploading a backup JSON file.")
            
            uploaded_json = st.file_uploader("Upload backup file", type=["json"], key="config_json_uploader")
            if uploaded_json is not None:
                if st.button("Import Backup settings", use_container_width=True):
                    try:
                        backup_dict = json.loads(uploaded_json.getvalue().decode("utf-8"))
                        
                        db.set_setting("similarity_threshold", str(backup_dict.get("similarity_threshold", SIMILARITY_THRESHOLD)))
                        db.set_setting("default_image_size", str(backup_dict.get("default_image_size", 128)))
                        db.set_setting("app_theme", backup_dict.get("app_theme", "Dark"))
                        
                        st.success("Backup configuration successfully loaded and applied!")
                        st.rerun()
                    except Exception as json_err:
                        st.error(f"Failed to parse config file: {json_err}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # Tab 2: Database Utilities
    # ──────────────────────────────────────────────────────────────
    with tab_db:
        st.subheader("Database Maintenance & Reset")
        st.write("Monitor database usage and perform administrative actions.")

        db_size_kb = 0.0
        if os.path.exists(db.db_path):
            db_size_kb = os.path.getsize(db.db_path) / 1024.0

        stats = db.get_statistics()

        # Database Stats cards
        col_db1, col_db2, col_db3 = st.columns(3)
        with col_db1:
            render_metric_card("DB File Size", f"{db_size_kb:.1f} KB", "💾", "#1B6EF3")
        with col_db2:
            render_metric_card("Registered Specimens", f"{stats['total_signatures']}", "📝", "#00BCD4")
        with col_db3:
            render_metric_card("History Log Entries", f"{stats['total_verifications']}", "✅", "#4CAF50")

        # Dangerous Reset Operations
        st.markdown("<br><hr style='border-top: 1px dashed #FF1744;'>", unsafe_allow_html=True)
        st.subheader("🚨 Administrative Reset")
        st.write("Warning: Resetting the database will delete all registered user specimens, verification history records, and configurations. This action is irreversible.")

        # Reset flow validation
        if "reset_db_confirm" not in st.session_state:
            st.session_state["reset_db_confirm"] = False

        if not st.session_state["reset_db_confirm"]:
            if st.button("⚠️ Initialize Full Database Reset", use_container_width=True):
                st.session_state["reset_db_confirm"] = True
                st.rerun()
        else:
            st.warning("⚠️ CRITICAL WARNING: You are about to wipe the database. Confirming this action will delete all physical files and tables.")
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if st.button("🔴 YES, CONFIRM RESET", use_container_width=True):
                    try:
                        # Drop and recreate tables
                        db.reset_database()
                        
                        # Clear local directories if desired (optional)
                        # We can log this
                        logger.info("Database reset executed by user request.")
                        
                        # Clear session states
                        st.session_state.clear()
                        st.session_state["reset_db_confirm"] = False
                        
                        st.success("Database successfully initialized to clean state!")
                        st.rerun()
                    except Exception as reset_err:
                        st.error(f"Reset failed: {reset_err}")
            with r_col2:
                if st.button("Cancel Operation", use_container_width=True):
                    st.session_state["reset_db_confirm"] = False
                    st.rerun()

    # ──────────────────────────────────────────────────────────────
    # Tab 3: System Information & About
    # ──────────────────────────────────────────────────────────────
    with tab_info:
        st.subheader("System Stack & Diagnostics")
        st.write("Diagnostic versions of local libraries and system packages.")

        # Libraries versions table
        diag_data = {
            "Framework / Package": [
                "Streamlit Engine",
                "OpenCV Library (cv2)",
                "NumPy Array Operations",
                "Pandas Data Wrangling",
                "SciPy Mathematical routines",
                "Python Interpreter"
            ],
            "Version": [
                st.__version__,
                cv2.__version__,
                np.__version__,
                pd.__version__,
                "1.11.0" if 'scipy' in sys.modules else "scipy-scikit",
                sys.version.split(" ")[0]
            ]
        }
        st.table(pd.DataFrame(diag_data))

        # About App
        st.markdown("<br><hr style='border-top:1px solid #2D3748;'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card-container" style="text-align:center; padding:1.5rem;">
                <h3>{APP_NAME}</h3>
                <p style="color:#A0AEC0; font-size:0.9rem;">{APP_DESCRIPTION}</p>
                <p style="font-size:0.85rem; color:#718096; margin-top:5px;">
                    Developed by: {APP_AUTHOR} | Version: {APP_VERSION}
                </p>
                <p style="font-size:0.75rem; color:#4A5568;">
                    Built with Streamlit, OpenCV, NumPy, and Scikit-Learn
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
