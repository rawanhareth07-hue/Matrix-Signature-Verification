"""
Configuration module for AI Signature Verification & Anti-Forgery System.

Contains all application-wide constants, paths, thresholds, and settings.
Centralizes configuration to enable easy tuning without touching business logic.
"""

import os
import logging
from typing import Dict, Tuple

# ──────────────────────────────────────────────────────────────
# Application Metadata
# ──────────────────────────────────────────────────────────────
APP_NAME: str = "AI Signature Verification & Anti-Forgery System"
APP_VERSION: str = "1.0.0"
APP_AUTHOR: str = "AI Security Lab"
APP_DESCRIPTION: str = (
    "Intelligent system for handwritten signature verification, "
    "forgery detection, and linear algebra education."
)

# ──────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(BASE_DIR, "database.db")
UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")
IMAGES_DIR: str = os.path.join(BASE_DIR, "images")
STYLES_DIR: str = os.path.join(BASE_DIR, "styles")

# ──────────────────────────────────────────────────────────────
# Image Processing Settings
# ──────────────────────────────────────────────────────────────
SUPPORTED_FORMATS: list = ["png", "jpg", "jpeg", "bmp", "tiff"]
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 128)
GRAYSCALE_THRESHOLD: int = 127
NOISE_KERNEL_SIZE: int = 3
MORPH_KERNEL_SIZE: int = 3

# ──────────────────────────────────────────────────────────────
# Verification Thresholds
# ──────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.75
CONFIDENCE_THRESHOLD: float = 0.80
DISTANCE_WEIGHTS: Dict[str, float] = {
    "cosine": 0.40,
    "euclidean": 0.25,
    "manhattan": 0.15,
    "correlation": 0.20,
}

# ──────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────
FEATURE_VECTOR_SIZE: int = 256
HISTOGRAM_BINS: int = 32
PCA_COMPONENTS: int = 50

# ──────────────────────────────────────────────────────────────
# UI Settings
# ──────────────────────────────────────────────────────────────
DEFAULT_THEME: str = "dark"
PAGE_ICON: str = "🛡️"
SIDEBAR_STATE: str = "expanded"

# ──────────────────────────────────────────────────────────────
# Report Settings
# ──────────────────────────────────────────────────────────────
REPORT_FORMATS: list = ["PDF", "Excel", "CSV"]

# ──────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: str = os.path.join(BASE_DIR, "app.log")

# ──────────────────────────────────────────────────────────────
# Initialize Required Directories
# ──────────────────────────────────────────────────────────────
for _dir in [UPLOAD_DIR, RESULTS_DIR, DATA_DIR, ASSETS_DIR, IMAGES_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Configure Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)

logger: logging.Logger = logging.getLogger(APP_NAME)
