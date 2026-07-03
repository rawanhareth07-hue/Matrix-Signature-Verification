"""
Database management module for AI Signature Verification & Anti-Forgery System.

Provides thread-safe SQLite database operations for storing signatures,
verification results, and application settings. Uses pickle serialization
for numpy feature vectors and supports CSV/Excel import/export.
"""

import os
import sqlite3
import pickle
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from config import DB_PATH, RESULTS_DIR, logger


class DatabaseManager:
    """Manages all database operations for signature storage and verification tracking.

    Provides a thread-safe SQLite interface with context manager support for
    automatic connection cleanup. Handles serialization of numpy arrays for
    feature vector storage and supports bulk import/export operations.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
        conn: Active SQLite connection with WAL journal mode.
        cursor: Database cursor for executing queries.
    """

    def __init__(self, db_path: str = None) -> None:
        """Initialize the database manager and establish connection.

        Args:
            db_path: Path to the SQLite database file. Falls back to
                     config.DB_PATH when not provided.
        """
        self.db_path = db_path if db_path is not None else DB_PATH
        try:
            # check_same_thread=False allows multi-threaded access for Streamlit
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # WAL mode improves concurrent read performance
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.cursor = self.conn.cursor()
            self.initialize_database()
            logger.info("Database connection established at %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("Failed to connect to database at %s: %s", self.db_path, e)
            raise

    def __enter__(self) -> "DatabaseManager":
        """Support usage as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the database connection is closed on exit."""
        self.close()

    def initialize_database(self) -> None:
        """Create the required database tables if they do not already exist.

        Tables:
            signatures: Stores registered signature images and their feature vectors.
            verifications: Records each verification attempt with scores and decisions.
            settings: Key-value store for application configuration persistence.
        """
        try:
            self.cursor.executescript("""
                CREATE TABLE IF NOT EXISTS signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature_id INTEGER,
                    uploaded_image_path TEXT,
                    similarity_score REAL,
                    decision TEXT,
                    method TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signature_id) REFERENCES signatures(id)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            self.conn.commit()
            logger.info("Database tables initialized successfully.")
        except sqlite3.Error as e:
            logger.error("Failed to initialize database tables: %s", e)
            raise

    def add_signature(self, user_name: str, image_path: str,
                      feature_vector: np.ndarray, notes: str = '') -> int:
        """Register a new signature in the database.

        The feature vector is serialized using pickle before storage so that
        arbitrary numpy arrays can be round-tripped without loss.

        Args:
            user_name: Name of the person whose signature is being stored.
            image_path: Filesystem path to the saved signature image.
            feature_vector: Extracted feature vector as a numpy array.
            notes: Optional free-text notes about this signature.

        Returns:
            The auto-generated row ID of the newly inserted signature.
        """
        try:
            # Serialize the numpy array to bytes for BLOB storage
            vector_blob = pickle.dumps(feature_vector)
            self.cursor.execute(
                "INSERT INTO signatures (user_name, image_path, feature_vector, notes) "
                "VALUES (?, ?, ?, ?)",
                (user_name, image_path, vector_blob, notes)
            )
            self.conn.commit()
            inserted_id = self.cursor.lastrowid
            logger.info("Added signature id=%d for user '%s'.", inserted_id, user_name)
            return inserted_id
        except (sqlite3.Error, pickle.PicklingError) as e:
            logger.error("Failed to add signature for '%s': %s", user_name, e)
            self.conn.rollback()
            raise

    def get_all_signatures(self) -> List[Dict]:
        """Retrieve every registered signature with deserialized feature vectors.

        Returns:
            A list of dictionaries, each containing id, user_name, image_path,
            feature_vector (as np.ndarray), created_at, and notes.
        """
        try:
            self.cursor.execute("SELECT * FROM signatures ORDER BY created_at DESC")
            rows = self.cursor.fetchall()
            signatures = []
            for row in rows:
                signatures.append({
                    'id': row['id'],
                    'user_name': row['user_name'],
                    'image_path': row['image_path'],
                    'feature_vector': pickle.loads(row['feature_vector']),
                    'created_at': row['created_at'],
                    'notes': row['notes'],
                })
            logger.info("Retrieved %d signatures.", len(signatures))
            return signatures
        except (sqlite3.Error, pickle.UnpicklingError) as e:
            logger.error("Failed to retrieve signatures: %s", e)
            return []

    def get_signature_by_id(self, sig_id: int) -> Optional[Dict]:
        """Retrieve a single signature record by its primary key.

        Args:
            sig_id: The integer ID of the signature to look up.

        Returns:
            A dictionary with the signature data, or None if not found.
        """
        try:
            self.cursor.execute("SELECT * FROM signatures WHERE id = ?", (sig_id,))
            row = self.cursor.fetchone()
            if row is None:
                logger.warning("Signature id=%d not found.", sig_id)
                return None
            return {
                'id': row['id'],
                'user_name': row['user_name'],
                'image_path': row['image_path'],
                'feature_vector': pickle.loads(row['feature_vector']),
                'created_at': row['created_at'],
                'notes': row['notes'],
            }
        except (sqlite3.Error, pickle.UnpicklingError) as e:
            logger.error("Failed to retrieve signature id=%d: %s", sig_id, e)
            return None

    def delete_signature(self, sig_id: int) -> bool:
        """Delete a signature record by its primary key.

        Args:
            sig_id: The integer ID of the signature to delete.

        Returns:
            True if a row was deleted, False otherwise.
        """
        try:
            self.cursor.execute("DELETE FROM signatures WHERE id = ?", (sig_id,))
            self.conn.commit()
            deleted = self.cursor.rowcount > 0
            if deleted:
                logger.info("Deleted signature id=%d.", sig_id)
            else:
                logger.warning("No signature found with id=%d to delete.", sig_id)
            return deleted
        except sqlite3.Error as e:
            logger.error("Failed to delete signature id=%d: %s", sig_id, e)
            self.conn.rollback()
            return False

    def update_signature(self, sig_id: int, **kwargs) -> bool:
        """Update mutable fields of an existing signature.

        Only whitelisted columns (user_name, notes) are accepted to prevent
        SQL injection through arbitrary column names.

        Args:
            sig_id: The integer ID of the signature to update.
            **kwargs: Column-value pairs to update (user_name, notes).

        Returns:
            True if the update affected a row, False otherwise.
        """
        allowed_columns = {'user_name', 'notes'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_columns}
        if not updates:
            logger.warning("No valid columns provided for update on id=%d.", sig_id)
            return False
        try:
            set_clause = ", ".join(f"{col} = ?" for col in updates)
            values = list(updates.values()) + [sig_id]
            self.cursor.execute(
                f"UPDATE signatures SET {set_clause} WHERE id = ?", values
            )
            self.conn.commit()
            updated = self.cursor.rowcount > 0
            if updated:
                logger.info("Updated signature id=%d fields: %s", sig_id, list(updates.keys()))
            return updated
        except sqlite3.Error as e:
            logger.error("Failed to update signature id=%d: %s", sig_id, e)
            self.conn.rollback()
            return False

    def add_verification(self, signature_id: int, uploaded_image_path: str,
                         similarity_score: float, decision: str,
                         method: str, processing_time: float) -> int:
        """Record a verification attempt.

        Args:
            signature_id: ID of the reference signature compared against.
            uploaded_image_path: Path to the uploaded test image.
            similarity_score: Computed overall similarity score (0-1).
            decision: Verdict string, typically 'Genuine' or 'Forged'.
            method: Description of the comparison method used.
            processing_time: Wall-clock seconds taken for the verification.

        Returns:
            The auto-generated row ID of the verification record.
        """
        try:
            self.cursor.execute(
                "INSERT INTO verifications "
                "(signature_id, uploaded_image_path, similarity_score, decision, method, processing_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (signature_id, uploaded_image_path, similarity_score,
                 decision, method, processing_time)
            )
            self.conn.commit()
            inserted_id = self.cursor.lastrowid
            logger.info(
                "Recorded verification id=%d: sig=%d score=%.4f decision=%s",
                inserted_id, signature_id, similarity_score, decision
            )
            return inserted_id
        except sqlite3.Error as e:
            logger.error("Failed to add verification for sig=%d: %s", signature_id, e)
            self.conn.rollback()
            raise

    def get_verification_history(self) -> List[Dict]:
        """Retrieve the full verification history joined with signature user names.

        Returns:
            List of dicts with verification details and the associated user name.
        """
        try:
            self.cursor.execute("""
                SELECT v.*, s.user_name
                FROM verifications v
                LEFT JOIN signatures s ON v.signature_id = s.id
                ORDER BY v.created_at DESC
            """)
            rows = self.cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    'id': row['id'],
                    'signature_id': row['signature_id'],
                    'user_name': row['user_name'] if row['user_name'] else 'Unknown',
                    'uploaded_image_path': row['uploaded_image_path'],
                    'similarity_score': row['similarity_score'],
                    'decision': row['decision'],
                    'method': row['method'],
                    'processing_time': row['processing_time'],
                    'created_at': row['created_at'],
                })
            logger.info("Retrieved %d verification records.", len(history))
            return history
        except sqlite3.Error as e:
            logger.error("Failed to retrieve verification history: %s", e)
            return []

    def get_statistics(self) -> Dict:
        """Compute aggregate statistics across all signatures and verifications.

        Returns:
            Dictionary containing total_signatures, total_verifications,
            genuine_count, forged_count, avg_similarity, avg_processing_time,
            and accuracy (genuine_count / total_verifications).
        """
        try:
            stats: Dict[str, Any] = {}

            self.cursor.execute("SELECT COUNT(*) FROM signatures")
            stats['total_signatures'] = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT COUNT(*) FROM verifications")
            stats['total_verifications'] = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT COUNT(*) FROM verifications WHERE decision = 'Genuine'"
            )
            stats['genuine_count'] = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT COUNT(*) FROM verifications WHERE decision = 'Forged'"
            )
            stats['forged_count'] = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT AVG(similarity_score) FROM verifications"
            )
            avg_sim = self.cursor.fetchone()[0]
            stats['avg_similarity'] = round(avg_sim, 4) if avg_sim is not None else 0.0

            self.cursor.execute(
                "SELECT AVG(processing_time) FROM verifications"
            )
            avg_time = self.cursor.fetchone()[0]
            stats['avg_processing_time'] = round(avg_time, 4) if avg_time is not None else 0.0

            # Accuracy is defined as the proportion of genuine verdicts
            if stats['total_verifications'] > 0:
                stats['accuracy'] = round(
                    stats['genuine_count'] / stats['total_verifications'], 4
                )
            else:
                stats['accuracy'] = 0.0

            logger.info("Computed statistics: %s", stats)
            return stats
        except sqlite3.Error as e:
            logger.error("Failed to compute statistics: %s", e)
            return {
                'total_signatures': 0, 'total_verifications': 0,
                'genuine_count': 0, 'forged_count': 0,
                'avg_similarity': 0.0, 'avg_processing_time': 0.0,
                'accuracy': 0.0,
            }

    def search_signatures(self, query: str) -> List[Dict]:
        """Search for signatures whose user_name matches a pattern.

        Uses SQL LIKE with wildcards for flexible substring matching.

        Args:
            query: The search string (case-insensitive substring match).

        Returns:
            List of matching signature dictionaries.
        """
        try:
            self.cursor.execute(
                "SELECT * FROM signatures WHERE user_name LIKE ? ORDER BY created_at DESC",
                (f"%{query}%",)
            )
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'user_name': row['user_name'],
                    'image_path': row['image_path'],
                    'feature_vector': pickle.loads(row['feature_vector']),
                    'created_at': row['created_at'],
                    'notes': row['notes'],
                })
            logger.info("Search for '%s' returned %d results.", query, len(results))
            return results
        except (sqlite3.Error, pickle.UnpicklingError) as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []

    def export_data(self, format: str, output_dir: str = None) -> str:
        """Export verification and signature data to a file.

        Args:
            format: Output format, either 'csv' or 'excel' (case-insensitive).
            output_dir: Directory to write the export file. Defaults to
                        config.RESULTS_DIR.

        Returns:
            The absolute path of the generated export file.

        Raises:
            ValueError: If an unsupported format is requested.
        """
        output_dir = output_dir if output_dir is not None else RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if format.lower() == 'csv':
                filepath = os.path.join(output_dir, f"export_{timestamp}.csv")
                history = self.get_verification_history()
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if history:
                        writer = csv.DictWriter(f, fieldnames=history[0].keys())
                        writer.writeheader()
                        writer.writerows(history)
                    else:
                        # Write header-only file when no data exists
                        writer = csv.writer(f)
                        writer.writerow([
                            'id', 'signature_id', 'user_name',
                            'uploaded_image_path', 'similarity_score',
                            'decision', 'method', 'processing_time', 'created_at'
                        ])
                logger.info("Exported CSV to %s", filepath)
                return filepath

            elif format.lower() == 'excel':
                filepath = os.path.join(output_dir, f"export_{timestamp}.xlsx")
                # Build DataFrames from query results
                history = self.get_verification_history()
                sigs = self.get_all_signatures()
                # Strip non-serializable feature vectors for Excel
                sig_records = [
                    {k: v for k, v in s.items() if k != 'feature_vector'}
                    for s in sigs
                ]
                df_history = pd.DataFrame(history) if history else pd.DataFrame()
                df_sigs = pd.DataFrame(sig_records) if sig_records else pd.DataFrame()

                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df_history.to_excel(writer, sheet_name='Verifications', index=False)
                    df_sigs.to_excel(writer, sheet_name='Signatures', index=False)
                logger.info("Exported Excel to %s", filepath)
                return filepath

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error("Export failed (format=%s): %s", format, e)
            raise

    def import_data(self, file_path: str) -> int:
        """Import signatures from a CSV file.

        Expected CSV columns: user_name, image_path, notes (optional).
        A zero-filled feature vector is created as a placeholder for each
        imported record; callers should recompute features after import.

        Args:
            file_path: Absolute path to the CSV file to import.

        Returns:
            The number of signature records successfully imported.
        """
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_name = row.get('user_name', '').strip()
                    image_path = row.get('image_path', '').strip()
                    notes = row.get('notes', '').strip()
                    if not user_name or not image_path:
                        logger.warning("Skipping incomplete row: %s", row)
                        continue
                    # Placeholder feature vector – should be recomputed
                    placeholder_vector = np.zeros(256, dtype=np.float32)
                    self.add_signature(user_name, image_path, placeholder_vector, notes)
                    count += 1
            logger.info("Imported %d signatures from %s.", count, file_path)
            return count
        except Exception as e:
            logger.error("Import failed from %s: %s", file_path, e)
            raise

    def reset_database(self) -> bool:
        """Drop all tables and recreate the schema from scratch.

        Returns:
            True if the reset succeeded, False on error.
        """
        try:
            self.cursor.executescript("""
                DROP TABLE IF EXISTS verifications;
                DROP TABLE IF EXISTS signatures;
                DROP TABLE IF EXISTS settings;
            """)
            self.conn.commit()
            self.initialize_database()
            logger.info("Database reset successfully.")
            return True
        except sqlite3.Error as e:
            logger.error("Database reset failed: %s", e)
            return False

    def get_setting(self, key: str, default: str = '') -> str:
        """Read a value from the settings key-value store.

        Args:
            key: The setting key to look up.
            default: Value to return if the key does not exist.

        Returns:
            The stored value string, or *default* if absent.
        """
        try:
            self.cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = self.cursor.fetchone()
            return row['value'] if row else default
        except sqlite3.Error as e:
            logger.error("Failed to get setting '%s': %s", key, e)
            return default

    def set_setting(self, key: str, value: str) -> None:
        """Write a value into the settings key-value store.

        Uses INSERT OR REPLACE to handle both new keys and updates.

        Args:
            key: The setting key.
            value: The value to store.
        """
        try:
            self.cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
            self.conn.commit()
            logger.info("Setting '%s' = '%s' saved.", key, value)
        except sqlite3.Error as e:
            logger.error("Failed to set setting '%s': %s", key, e)
            self.conn.rollback()

    def close(self) -> None:
        """Close the database connection and release resources."""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed.")
        except sqlite3.Error as e:
            logger.error("Error closing database connection: %s", e)
