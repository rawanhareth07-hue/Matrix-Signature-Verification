"""
Reports and analytics generation page for AI Signature Verification.

Allows exporting system verification history logs and KPI matrices to PDF, Excel,
and CSV formats. Displays interactive Plotly charts tracking system statistics
and verification attempts.
"""

import os
import streamlit as st
from models.database import DatabaseManager
from utils.report_generator import ReportGenerator
from styles.theme import get_section_header
from components.cards import render_kpi_cards, render_info_card
from components.charts import (
    render_verification_pie_chart,
    render_similarity_distribution,
    render_activity_timeline
)


def render_reports() -> None:
    """Render the Reports & Analytics page."""
    st.markdown(
        get_section_header(
            "Reports & Analytics",
            "Generate formal compliance reports and explore deep verification analytics"
        ),
        unsafe_allow_html=True
    )

    db = DatabaseManager()
    reporter = ReportGenerator(db)

    # Load stats and history for charts
    stats = db.get_statistics()
    history = db.get_verification_history()

    # Layout: Generation controls on the left, system summary info on the right
    col_gen, col_sys = st.columns([2, 1])

    with col_gen:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📋 Document Generation Engine")
        
        report_format = st.selectbox(
            "Select Output Format",
            ["PDF Document", "Excel Workbook", "CSV Data Log"]
        )
        
        st.write("The generated report captures all KPIs, verification history lists, system metadata, and stats.")
        
        if st.button("🚀 Generate Report", use_container_width=True):
            with st.spinner("Generating document..."):
                try:
                    filepath = ""
                    mime_type = "application/octet-stream"
                    file_label = "Download Report File"
                    
                    if report_format == "PDF Document":
                        filepath = reporter.generate_pdf_report()
                        mime_type = "application/pdf"
                        file_label = "⬇️ Download PDF Report"
                        
                    elif report_format == "Excel Workbook":
                        filepath = reporter.generate_excel_report()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_label = "⬇️ Download Excel Report"
                        
                    else:  # CSV
                        filepath = reporter.generate_csv_report()
                        mime_type = "text/csv"
                        file_label = "⬇️ Download CSV Log"

                    # Ready for download
                    if os.path.exists(filepath):
                        st.success(f"Report generated successfully! File: {os.path.basename(filepath)}")
                        
                        with open(filepath, "rb") as f:
                            bytes_data = f.read()
                            
                        st.download_button(
                            label=file_label,
                            data=bytes_data,
                            file_name=os.path.basename(filepath),
                            mime=mime_type,
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
                    st.exception(e)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sys:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🖥️ System Diagnostics")
        
        # Get DB file size
        db_size_mb = 0.0
        if os.path.exists(db.db_path):
            db_size_mb = os.path.getsize(db.db_path) / (1024.0 * 1024.0)

        st.write(f"**Database Size:** {db_size_mb:.2f} MB")
        st.write(f"**Database Path:** `{db.db_path}`")
        st.write(f"**Total Records:** {stats['total_signatures'] + stats['total_verifications']} entries")
        st.write(f"**Verification Rate:** {stats['total_verifications']} attempts")
        st.markdown('</div>', unsafe_allow_html=True)

    # Analytics Section (Show charts only if verification data exists)
    st.markdown("<br><hr style='border-top: 2px solid #1B6EF3;'>", unsafe_allow_html=True)
    st.subheader("📈 Interactive Verification Analytics")

    if stats["total_verifications"] > 0:
        # Display KPI Cards Row
        render_kpi_cards(stats)
        st.markdown("<br>", unsafe_allow_html=True)

        # Charts grid
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Verification Decisions Ratio")
            render_verification_pie_chart(stats["genuine_count"], stats["forged_count"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_col2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Verification Activity Timeline")
            render_activity_timeline(history)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Bottom Chart - Similarity Distribution
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Verification Similarity Score Distribution")
        scores = [h["similarity_score"] for h in history if "similarity_score" in h]
        render_similarity_distribution(scores)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        render_info_card(
            "Analytics Not Available",
            "Verification charts will be displayed here once signatures are verified. "
            "Please upload signature specimens, verify some attempts, and return to see visual statistics.",
            "📊"
        )
