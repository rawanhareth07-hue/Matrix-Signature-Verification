# AI Signature Verification & Anti-Forgery System 🛡️

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-ready enterprise application for handwritten signature verification, biometric anti-forgery detection, and interactive Linear Algebra education. 

This application uses a modular clean architecture separating business logic from UI, enabling future deep-learning model swaps without changing frontend layouts.

---

## 🌟 Key Features

1. **Enterprise Dashboard**: Provides overview KPIs (Total Signatures, Verifications, Genuine/Forged counts, Latency, Accuracy) with interactive Plotly visualizations.
2. **Dynamic Processing Pipeline**: Visualizes each of the 9 image processing steps in real-time (Grayscale, Gaussian Blur, Otsu threshold, Morphological opening/closing, Skeletonization, Centering/Cropping).
3. **Feature Space Extraction**: Extracts geometric profile maps, projection histograms (horizontal & vertical), aspect ratios, center of mass, and stroke distribution histograms using distance transforms.
4. **Vector Similarity Matching**: Performs biometric matching using Cosine Similarity, Euclidean L2 distance, Manhattan L1 distance, and Pearson Correlation.
5. **Linear Algebra Demonstration**: Visualizes core math operations side-by-side with LaTeX equations and explanations (Eigenvalues/Eigenvectors, SVD reconstructions, PCA variance scree plots, matrix transposes, and coordinate rotations).
6. **Robust Specimen database**: Full SQLite specimen registry with CRUD operations, search filters, CSV bulk imports, and data exports.
7. **Compliance Reporting**: Generates formatted executive PDF reports (using `fpdf2`), multi-sheet Excel files (using `openpyxl`), and raw CSV log outputs.

---

## 📐 Linear Algebra Concepts Covered

- **Matrix Representations**: Mapping image intensities to $\mathbf{A} \in \mathbb{R}^{m \times n}$ matrices and binarizing via threshold functions.
- **Vector Spaces & Norms**: Converting matrices to high-dimensional feature vectors $\mathbf{x} \in \mathbb{R}^{256}$ and measuring magnitudes using L1, L2, and L-infinity norms.
- **Linear Transformations**: Coordinate mappings representing translation, scaling, and rotations ($\mathbf{v'} = \mathbf{T}\mathbf{v}$).
- **Cosine Similarity**: Computing angular similarity between projection profiles: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$.
- **Matrix Rank**: Determining independent details using column and row rank: $\text{rank}(\mathbf{A})$.
- **Singular Value Decomposition (SVD)**: Decomposing blocks to rotations and scaling eigenvalues: $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ to compress and filter noise.
- **Principal Component Analysis (PCA)**: Projecting feature dimensions to directions of maximum variance using symmetric covariance matrices.

---

## 📂 Project Structure

```text
├── app.py                      # Application bootstrap & sidebar router
├── config.py                   # App metadata, default thresholds, directory settings
├── database.db                 # SQLite database (auto-initialized on startup)
├── requirements.txt            # System dependencies with version pins
├── README.md                   # System documentation
├── models/
│   ├── __init__.py
│   └── database.py             # SQLite CRUD, stats, CSV import/export
├── components/
│   ├── __init__.py
│   ├── cards.py                # Reusable HTML metric & KPI cards
│   └── charts.py               # Plotly chart templates (radar, histograms, timeline)
├── utils/
│   ├── __init__.py
│   ├── image_processing.py     # 9-step OpenCV normalization pipeline
│   ├── feature_extraction.py    # Bounding boxes, projections, L2 vector normalizer
│   ├── linear_algebra.py       # SVD, PCA, eigenvalues, dot product computations
│   ├── similarity.py           # Cosine/Euclidean/Manhattan weighted matching
│   └── report_generator.py     # PDF, Excel, and CSV exporter
├── styles/
│   ├── __init__.py
│   └── theme.py                # Custom Dark-mode CSS styles & HTML layout templates
├── pages/
│   ├── __init__.py
│   ├── dashboard.py            # Overview charts & analytics
│   ├── upload.py               # Specimen file ingestion
│   ├── processing.py           # Preprocessing step visualizer
│   ├── linear_algebra_demo.py  # Educational math screens
│   ├── verification.py         # Signature match verification UI
│   ├── database_page.py        # Specimen browser & CSV bulk import
│   ├── reports.py              # Report generation layout
│   └── settings.py             # Threshold slider settings & DB reset tools
├── uploads/                    # Stores physically registered signature images
└── results/                    # Stores exported PDF/Excel report output files
```

---

## 🚀 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd linear-algebra
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

The application automatically creates the required directories (`uploads/`, `results/`) and initializes the database (`database.db`) on startup.

---

## 🛠️ Technology Stack

- **UI Framework**: Streamlit
- **Computer Vision**: OpenCV, Pillow (PIL)
- **Numerical Operations**: NumPy, SciPy (ndimage)
- **Machine Learning**: Scikit-Learn (PCA, Standard Scaler)
- **Visualization**: Plotly Graph Objects, Express, Matplotlib
- **Database**: SQLite3
- **Reports**: fpdf2, openpyxl, pandas
- **Development**: Python 3.12+

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
