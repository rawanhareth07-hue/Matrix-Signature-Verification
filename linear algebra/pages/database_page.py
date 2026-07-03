"""
Database management page for AI Signature Verification System.

Enables searching, viewing, updating, and deleting signature specimens from the SQLite database.
Supports bulk export of histories/specimens and importing entries via CSV.
"""

import os
import streamlit as st
import pandas as pd
from models.database import DatabaseManager
from styles.theme import get_section_header
from components.cards import render_metric_card


def render_database() -> None:
    """Render the Database Management page."""
    st.markdown(
        get_section_header(
            "Database Management",
            "Browse, edit, and manage registered signature specimens and verification logs"
        ),
        unsafe_allow_html=True
    )

    db = DatabaseManager()

    # Get stats
    stats = db.get_statistics()
    
    # Render mini stats row
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        render_metric_card("Registered Specimens", f"{stats['total_signatures']}", "📝", "#1B6EF3")
    with col_s2:
        render_metric_card("Total Verifications", f"{stats['total_verifications']}", "✅", "#00BCD4")

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize edit/delete action states in session state
    if "edit_id" not in st.session_state:
        st.session_state["edit_id"] = None
    if "delete_id" not in st.session_state:
        st.session_state["delete_id"] = None

    # Search Bar
    search_query = st.text_input("🔍 Search signatures by username", placeholder="Enter username here...")

    # Load Signatures
    if search_query:
        signatures = db.search_signatures(search_query)
    else:
        signatures = db.get_all_signatures()

    # Handle Edit Forms first if active
    if st.session_state["edit_id"] is not None:
        edit_id = st.session_state["edit_id"]
        sig_data = db.get_signature_by_id(edit_id)
        
        if sig_data:
            st.markdown(f'<div class="card-container" style="border:1px solid #1B6EF3;">', unsafe_allow_html=True)
            st.subheader(f"✏️ Edit Specimen Details (ID #{edit_id})")
            
            with st.form("edit_specimen_form"):
                new_username = st.text_input("Username", value=sig_data["user_name"])
                new_notes = st.text_area("Notes", value=sig_data["notes"])
                
                col_ef1, col_ef2 = st.columns([1, 1])
                with col_ef1:
                    submit_edit = st.form_submit_button("Save changes", use_container_width=True)
                with col_ef2:
                    cancel_edit = st.form_submit_button("Cancel", use_container_width=True)
                
                if submit_edit:
                    if not new_username.strip():
                        st.error("Username cannot be empty.")
                    else:
                        db.update_signature(edit_id, user_name=new_username.strip(), notes=new_notes.strip())
                        st.success("Specimen details updated successfully!")
                        st.session_state["edit_id"] = None
                        st.rerun()
                
                if cancel_edit:
                    st.session_state["edit_id"] = None
                    st.rerun()
            st.markdown('</div><br>', unsafe_allow_html=True)

    # Handle Delete Confirmation first if active
    if st.session_state["delete_id"] is not None:
        del_id = st.session_state["delete_id"]
        sig_data = db.get_signature_by_id(del_id)
        
        if sig_data:
            st.markdown('<div class="card-container" style="border:1px solid #FF1744;">', unsafe_allow_html=True)
            st.warning(f"⚠️ Are you sure you want to delete signature specimen for '{sig_data['user_name']}' (ID #{del_id})?")
            st.write("This will permanently remove the record from database and delete its image file from uploads folder.")
            
            col_df1, col_df2 = st.columns([1, 1])
            with col_df1:
                if st.button("🔴 Confirm Delete", use_container_width=True):
                    # Delete file if exists
                    if os.path.exists(sig_data["image_path"]):
                        try:
                            os.remove(sig_data["image_path"])
                        except Exception as file_err:
                            logger.error("Failed to delete file: %s", file_err)
                    
                    db.delete_signature(del_id)
                    st.success("Specimen deleted successfully!")
                    st.session_state["delete_id"] = None
                    st.rerun()
            with col_df2:
                if st.button("Cancel Delete", use_container_width=True):
                    st.session_state["delete_id"] = None
                    st.rerun()
            st.markdown('</div><br>', unsafe_allow_html=True)

    # Display signatures table
    if signatures:
        st.subheader("Registered Signature Specimens")
        
        # Build Table rows manually to display edit/delete buttons
        for sig in signatures:
            with st.container():
                st.markdown(
                    f"""
                    <div class="card-container" style="padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span style="font-weight: bold; font-size: 1.1rem; color: #1B6EF3;">{sig['user_name']}</span>
                                <span style="color: #A0AEC0; font-size: 0.8rem; margin-left: 10px;">ID: #{sig['id']}</span>
                                <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 5px;">Notes: {sig['notes'] if sig['notes'] else 'No notes added'}</div>
                                <div style="font-size: 0.75rem; color: #718096; margin-top: 5px;">Registered: {sig['created_at']}</div>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.8rem; font-family: monospace; color: #A0AEC0;">{os.path.basename(sig['image_path'])}</span>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Buttons row using streamlit columns
                act_col1, act_col2, act_col3 = st.columns([1, 1, 4])
                with act_col1:
                    if st.button("✏️ Edit", key=f"edit_btn_{sig['id']}", use_container_width=True):
                        st.session_state["edit_id"] = sig["id"]
                        st.rerun()
                with act_col2:
                    if st.button("🗑️ Delete", key=f"del_btn_{sig['id']}", use_container_width=True):
                        st.session_state["delete_id"] = sig["id"]
                        st.rerun()
                with act_col3:
                    # Provide image preview expander
                    with st.expander("🖼️ View Thumbnail"):
                        if os.path.exists(sig["image_path"]):
                            st.image(sig["image_path"], width=150)
                        else:
                            st.error("Image file missing on disk.")
            
            st.markdown("<hr style='border-top:1px solid #2D3748; margin: 0.5rem 0;'>", unsafe_allow_html=True)
    else:
        st.info("No signature specimens found in database matching search criteria.")

    # Bulk Operations Panel
    st.markdown("<br><hr style='border-top: 2px solid #1B6EF3;'>", unsafe_allow_html=True)
    st.subheader("Bulk Operations")

    col_op1, col_op2 = st.columns(2)

    with col_op1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📤 Export Data")
        st.write("Export signature registry database and verification history log files.")
        
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("Export to CSV", use_container_width=True):
                try:
                    path = db.export_data("csv")
                    with open(path, "r", encoding="utf-8") as f:
                        csv_data = f.read()
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=os.path.basename(path),
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success("CSV file ready for download!")
                except Exception as ex_err:
                    st.error(f"CSV Export failed: {ex_err}")

        with ex_col2:
            if st.button("Export to Excel", use_container_width=True):
                try:
                    path = db.export_data("excel")
                    with open(path, "rb") as f:
                        xlsx_data = f.read()
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=xlsx_data,
                        file_name=os.path.basename(path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.success("Excel workbook ready for download!")
                except Exception as ex_err:
                    st.error(f"Excel Export failed: {ex_err}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_op2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📥 Import Specimens")
        st.write("Import signature entries from a CSV. The CSV should contain columns: `user_name` and `image_path` (absolute path).")
        
        uploaded_csv = st.file_uploader("Choose CSV file", type=["csv"], key="bulk_csv_importer")
        if uploaded_csv is not None:
            if st.button("Start Bulk Import", use_container_width=True):
                try:
                    # Save temporary CSV to read
                    temp_path = os.path.join(os.path.dirname(db.db_path), "temp_import.csv")
                    with open(temp_path, "wb") as temp_f:
                        temp_f.write(uploaded_csv.getvalue())
                    
                    count = db.import_data(temp_path)
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                    st.success(f"Successfully imported {count} signatures into database!")
                    st.rerun()
                except Exception as imp_err:
                    st.error(f"Import failed: {imp_err}")
                    logger.error("Import failed: %s", imp_err, exc_info=True)
        st.markdown('</div>', unsafe_allow_html=True)
