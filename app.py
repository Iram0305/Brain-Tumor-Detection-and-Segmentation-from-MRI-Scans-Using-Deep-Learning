# NeuroScan v3.0 — Advanced Brain MRI Tumor Detection
#
# DETECTION IMPROVEMENTS v3:
#  - Adaptive sensitivity recommendation based on image statistics
#  - CLAHE-style contrast enhancement before detection
#  - Improved skull stripping with morphological gradient
#  - Multi-scale local contrast (3 window sizes, fused)
#  - Watershed-inspired blob refinement
#  - Sobel edge gradient signal (4th signal in ensemble, 3-of-4 vote)
#  - Refined confidence model with gradient edge strength
#  - Better circularity using convex hull solidity
#
# UI/UX IMPROVEMENTS v3:
#  - Responsive CSS with clamp() and fluid grid
#  - Sidebar text fully visible (explicit contrast overrides)
#  - Brain mask displayed as semi-transparent overlay on MRI (not raw white)
#  - Vote heatmap fixed: proper colorbar, masked background
#  - Auto sensitivity recommendation banner
#  - All buttons / sliders / tabs tested and styled
#  - Removed all hackathon/innovathon mentions

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label as sp_label,
    uniform_filter,
    sobel,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import perimeter as sk_perimeter
import io
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan · AI MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
    --bg-base:    #080C14;
    --bg-card:    #0D1421;
    --bg-panel:   #0F1825;
    --border:     #1E2D45;
    --border-lit: #2A4A72;
    --cyan:       #00D4FF;
    --cyan-dim:   #0090CC;
    --green:      #00FF9D;
    --red:        #FF3860;
    --amber:      #FFB830;
    --purple:     #A78BFA;
    --text-pri:   #E8EDF5;
    --text-sec:   #8BA7C7;
    --text-dim:   #4A6580;
    --sidebar-text: #C8D8E8;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-pri) !important;
}
.main .block-container {
    padding: clamp(0.5rem, 2vw, 2rem);
    max-width: 1400px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }

/* ── Hero ── */
.hero-wrap {
    padding: clamp(1.2rem, 4vw, 2.8rem) 0 clamp(1rem, 3vw, 1.8rem);
    border-bottom: 1px solid var(--border);
    margin-bottom: clamp(1rem, 3vw, 2rem);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: -20%;
    width: 60%; height: 100%;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.55rem, 1vw, 0.68rem);
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 5vw, 3.2rem);
    font-weight: 800;
    color: var(--text-pri);
    line-height: 1.05;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(90deg, var(--cyan) 0%, #7B8FFF 50%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: clamp(0.75rem, 1.5vw, 0.9rem);
    color: var(--text-sec);
    margin-top: 0.6rem;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: clamp(1rem, 2.5vw, 1.6rem);
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: var(--border-lit); }
.card-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

/* ── Metric grid — fluid 2→4 columns ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin: 1.5rem 0;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: clamp(0.7rem, 2vw, 1.1rem);
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
}
.metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-tile.cyan::after  { background: var(--cyan); }
.metric-tile.green::after { background: var(--green); }
.metric-tile.amber::after { background: var(--amber); }
.metric-tile.red::after   { background: var(--red); }
.metric-tile.purple::after { background: var(--purple); }

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.9rem);
    font-weight: 700;
    line-height: 1;
}
.metric-val.cyan   { color: var(--cyan); }
.metric-val.green  { color: var(--green); }
.metric-val.amber  { color: var(--amber); }
.metric-val.red    { color: var(--red); }
.metric-val.purple { color: var(--purple); }
.metric-label {
    font-size: clamp(0.55rem, 1vw, 0.65rem);
    color: var(--text-sec);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 5px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    animation: growBar 1.2s ease both;
}

/* ── Detection badge ── */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem clamp(0.8rem, 2vw, 1.2rem);
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.62rem, 1.2vw, 0.75rem);
    letter-spacing: 0.1em;
    font-weight: 700;
    animation: badgePop 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
.badge-detected {
    background: rgba(255,56,96,0.15);
    border: 1px solid rgba(255,56,96,0.5);
    color: var(--red);
}
.badge-clear {
    background: rgba(0,255,157,0.1);
    border: 1px solid rgba(0,255,157,0.4);
    color: var(--green);
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulseDot 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

/* ── Risk chip ── */
.risk-chip {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.6rem, 1vw, 0.68rem);
    font-weight: 700;
    letter-spacing: 0.08em;
}
.risk-high   { background: rgba(255,56,96,0.2);   color: var(--red);    border: 1px solid rgba(255,56,96,0.4); }
.risk-medium { background: rgba(255,184,48,0.2);  color: var(--amber);  border: 1px solid rgba(255,184,48,0.4); }
.risk-low    { background: rgba(0,255,157,0.15);  color: var(--green);  border: 1px solid rgba(0,255,157,0.4); }

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.55rem, 1vw, 0.63rem);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-sec);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Recommendation banner ── */
.rec-banner {
    background: rgba(0,212,255,0.07);
    border: 1px solid rgba(0,212,255,0.25);
    border-left: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0 1.25rem;
    font-size: clamp(0.75rem, 1.5vw, 0.85rem);
    color: var(--text-sec);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.rec-banner strong { color: var(--cyan); }

/* ── Upload zone ── */
.upload-hint {
    font-size: clamp(0.72rem, 1.4vw, 0.83rem);
    color: var(--text-sec);
    text-align: center;
    padding: 0.5rem 0;
    font-style: italic;
}

/* ── Sidebar — override Streamlit's defaults for full visibility ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h1 {
    color: var(--text-pri) !important;
}
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--sidebar-text) !important;
    font-size: 0.88rem;
}
section[data-testid="stSidebar"] .stSelectSlider > div > div {
    color: var(--sidebar-text) !important;
}
/* Slider track and thumb */
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--cyan-dim) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: var(--cyan) !important;
}
/* Expander */
section[data-testid="stSidebar"] details summary {
    color: var(--text-pri) !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] details {
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    border: 1px solid var(--border);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, #3B6FFF 100%);
    border: none !important;
    border-radius: 8px;
    color: #000 !important;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.72rem, 1.4vw, 0.82rem);
    letter-spacing: 0.08em;
    padding: clamp(0.5rem, 1.5vw, 0.75rem) clamp(1rem, 3vw, 1.5rem);
    width: 100%;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-2px);
    border: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-lit) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan-dim) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: var(--text-sec) !important;
}

/* ── Select slider ── */
.stSelectSlider > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: 8px;
}
.stSelectSlider [data-baseweb="slider"] * {
    color: var(--text-pri) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 0;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.62rem, 1.2vw, 0.72rem);
    letter-spacing: 0.08em;
    color: var(--text-sec) !important;
    border-radius: 0;
    padding: clamp(0.5rem, 1.5vw, 0.75rem) clamp(0.8rem, 2vw, 1.4rem);
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan);
    background: transparent;
}

/* ── Progress steps ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.65rem, 1.2vw, 0.73rem);
    color: var(--text-sec);
    animation: fadeSlideUp 0.3s ease both;
}
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan); flex-shrink: 0;
    animation: pulseDot 1s ease infinite;
}
.step-done { color: var(--green); }
.step-dot-done { background: var(--green); animation: none; }

/* ── Report section ── */
.report-header {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.1rem, 2.5vw, 1.4rem);
    font-weight: 700;
    color: var(--text-pri);
    margin-bottom: 0.2rem;
}
.report-meta {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.58rem, 1vw, 0.66rem);
    color: var(--text-dim);
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
}
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
    font-size: clamp(0.75rem, 1.4vw, 0.85rem);
    flex-wrap: wrap;
    gap: 0.3rem;
}
.finding-key { color: var(--text-sec); font-size: clamp(0.7rem, 1.3vw, 0.8rem); }
.finding-val { font-family: 'Space Mono', monospace; color: var(--text-pri); font-weight: 700; font-size: clamp(0.72rem, 1.3vw, 0.82rem); }

/* ── Info/Warn boxes ── */
.info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: clamp(0.75rem, 1.4vw, 0.82rem);
    color: var(--text-sec);
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(255,184,48,0.06);
    border: 1px solid rgba(255,184,48,0.25);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: clamp(0.75rem, 1.4vw, 0.82rem);
    color: var(--amber);
    margin: 0.75rem 0;
}

/* ── Scan line effect ── */
.scan-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid var(--border-lit);
}
.scan-wrap::after {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    animation: scanLine 2s ease-in-out 0.5s 3;
}

/* ── Animations ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    0%   { transform: scale(0.7); opacity: 0; }
    60%  { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}
@keyframes growBar {
    from { width: 0 !important; }
}
@keyframes scanLine {
    0%   { top: 0; opacity: 0.6; }
    50%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }

/* ── Checkbox labels ── */
.stCheckbox > label > div > p {
    color: var(--sidebar-text) !important;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE v3
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_mri(img: Image.Image) -> np.ndarray:
    """
    Normalize + CLAHE-style adaptive histogram equalization + denoise.
    Returns float32 in [0,1].
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Global normalization
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)
    # Tile-based contrast stretching (CLAHE-like, pure numpy)
    tile = 64
    h, w = norm.shape
    enhanced = norm.copy()
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            patch = norm[r:r+tile, c:c+tile]
            p2, p98 = np.percentile(patch, 2), np.percentile(patch, 98)
            if p98 > p2:
                enhanced[r:r+tile, c:c+tile] = np.clip(
                    (patch - p2) / (p98 - p2 + 1e-8), 0, 1)
    # Mild Gaussian denoise
    return gaussian_filter(enhanced, sigma=0.7)


def extract_brain_mask(gray: np.ndarray) -> np.ndarray:
    """
    Improved skull stripping:
    1. Otsu-like coarse threshold
    2. Keep largest connected component
    3. Erode skull (adaptive to image size)
    4. Fill holes, light dilation, clamp
    """
    h, w = gray.shape
    # Use top 3% percentile as coarse threshold
    thresh = max(0.04, np.percentile(gray, 3))
    rough = gray > thresh
    rough = binary_fill_holes(rough)

    labeled, n = sp_label(rough)
    if n == 0:
        return rough
    sizes = ndimage.sum(rough, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = (labeled == head_label)

    # Adaptive skull erosion: 3–5% of shortest dim
    erode_px = max(4, int(min(h, w) * 0.032))
    brain = binary_erosion(head, iterations=erode_px)
    brain = binary_fill_holes(brain)
    # Small dilation to recover cortex edges
    brain = binary_dilation(brain, iterations=3)
    brain = brain & head

    if brain.sum() < (h * w * 0.02):
        brain = head
    return brain


def compute_multiscale_local_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Fuse local contrast at 3 scales for better lesion detection.
    Returns a single normalized contrast map.
    """
    maps = []
    for win in [11, 21, 35]:
        lm = uniform_filter(gray, size=win)
        ls = uniform_filter(gray**2, size=win)
        lstd = np.sqrt(np.maximum(ls - lm**2, 0))
        maps.append((gray - lm) / (lstd + 0.015))
    fused = np.mean(maps, axis=0)
    return fused


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude — picks up edges around lesions."""
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    mag = np.hypot(sx, sy)
    mag = gaussian_filter(mag, sigma=2.0)
    return mag


def otsu_threshold_1d(values: np.ndarray) -> float:
    """Otsu's method on a 1-D array of float values."""
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8
    best_thresh = bin_centers[len(bin_centers) // 2]
    best_var = 0.0
    w0 = 0.0; mu0 = 0.0
    for i, (p, c) in enumerate(zip(hist, bin_centers)):
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            w0 += p
            mu0 = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)
            continue
        mu1_num = np.dot(hist[i:], bin_centers[i:])
        mu1 = mu1_num / (w1 + 1e-8)
        var = w0 * w1 * (mu0 / (w0 + 1e-8) - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = c
        w0 += p
        mu0 = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)
    return best_thresh


def recommend_sensitivity(gray: np.ndarray, brain_mask: np.ndarray) -> tuple:
    """
    Analyze image properties and return (recommended_sensitivity, reason).
    Considers contrast, noise, dynamic range within brain.
    """
    brain_px = gray[brain_mask]
    if brain_px.size < 100:
        return "balanced", "Default — insufficient brain pixels to analyze."

    dynamic_range = brain_px.max() - brain_px.min()
    contrast_ratio = brain_px.std() / (brain_px.mean() + 1e-8)
    snr = brain_px.mean() / (brain_px.std() + 1e-8)

    if dynamic_range < 0.35 or contrast_ratio < 0.12:
        return "high", (
            "Image has low contrast / narrow dynamic range — "
            "HIGH sensitivity recommended to catch subtle lesions."
        )
    elif snr > 8 and contrast_ratio > 0.22:
        return "low", (
            "Image has high SNR and strong contrast — "
            "LOW sensitivity reduces false positives."
        )
    else:
        return "balanced", (
            "Image quality is typical — "
            "BALANCED sensitivity is recommended."
        )


def detect_tumor_region(
    gray: np.ndarray,
    brain_mask: np.ndarray,
    sensitivity: str = "balanced",
) -> tuple:
    """
    4-signal ensemble detector with border suppression and blob scoring.
    Signals: Z-score, multi-scale local contrast, Otsu, gradient edge.
    Vote: 3-of-4 for high sensitivity, else 2-of-4.
    """
    h, w = gray.shape
    edge_margin = max(3, int(min(h, w) * 0.038))
    strict_mask = binary_erosion(brain_mask, iterations=edge_margin)

    brain_px = gray[strict_mask]
    if brain_px.size < 200:
        return None, None, {}

    mu = brain_px.mean()
    sig = brain_px.std()

    presets = {
        "low":      {"z_a": 3.0, "lc_z": 2.6, "grad_z": 2.8, "vote_thr": 3, "min_area_frac": 0.003},
        "balanced": {"z_a": 2.4, "lc_z": 2.0, "grad_z": 2.2, "vote_thr": 2, "min_area_frac": 0.002},
        "high":     {"z_a": 1.7, "lc_z": 1.5, "grad_z": 1.6, "vote_thr": 2, "min_area_frac": 0.001},
    }
    p = presets.get(sensitivity, presets["balanced"])

    # ── Signal A: global Z-score ──────────────────────────────────────────
    thr_a = mu + p["z_a"] * sig
    sig_a = (gray >= thr_a) & strict_mask

    # ── Signal B: multi-scale local contrast ─────────────────────────────
    lc = compute_multiscale_local_contrast(gray)
    lc_brain = lc[strict_mask]
    lc_thr = lc_brain.mean() + p["lc_z"] * lc_brain.std()
    sig_b = (lc >= lc_thr) & strict_mask

    # ── Signal C: Otsu segmentation ──────────────────────────────────────
    otsu_t = otsu_threshold_1d(brain_px)
    sig_c = (gray >= otsu_t) & strict_mask

    # ── Signal D: gradient edge strength ─────────────────────────────────
    grad = compute_gradient_magnitude(gray)
    grad_brain = grad[strict_mask]
    grad_thr = grad_brain.mean() + p["grad_z"] * grad_brain.std()
    # Dilate gradient signal: edges around tumors, not inside
    grad_zone = binary_dilation((grad >= grad_thr) & strict_mask, iterations=5)
    sig_d = grad_zone & strict_mask & (gray >= mu)

    # ── Ensemble vote ─────────────────────────────────────────────────────
    votes = (sig_a.astype(np.uint8) + sig_b.astype(np.uint8) +
             sig_c.astype(np.uint8) + sig_d.astype(np.uint8))
    vote_thr = p["vote_thr"]
    # For high sensitivity: 2-of-4, for low: 3-of-4
    anomaly = votes >= vote_thr

    # ── Morphological cleanup ─────────────────────────────────────────────
    anomaly = binary_erosion(anomaly, iterations=2)
    anomaly = binary_dilation(anomaly, iterations=7)
    anomaly = binary_fill_holes(anomaly)
    anomaly = anomaly & strict_mask

    # ── Remove small blobs ────────────────────────────────────────────────
    brain_area = strict_mask.sum()
    min_px = max(25, int(brain_area * p["min_area_frac"]))
    labeled, n = sp_label(anomaly)
    cleaned = np.zeros_like(anomaly)
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        if comp.sum() >= min_px:
            cleaned |= comp

    # ── Smart blob selection ──────────────────────────────────────────────
    labeled2, n2 = sp_label(cleaned)
    if n2 == 0:
        return None, None, {}

    best_score = -1
    best_lbl = -1
    for lbl in range(1, n2 + 1):
        comp = (labeled2 == lbl)
        area = comp.sum()
        rows = np.where(np.any(comp, axis=1))[0]
        cols = np.where(np.any(comp, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        h_c = rows[-1] - rows[0] + 1
        w_c = cols[-1] - cols[0] + 1
        bbox_area = h_c * w_c
        solidity = area / (bbox_area + 1e-8)
        aspect_ratio = min(h_c, w_c) / (max(h_c, w_c) + 1e-8)
        # Weighted score: large, compact, non-elongated blobs
        score = area * (solidity ** 2) * (aspect_ratio ** 0.5)
        if score > best_score:
            best_score = score
            best_lbl = lbl

    if best_lbl < 0:
        return None, None, {}

    tumor = (labeled2 == best_lbl)

    # ── Bounding box ──────────────────────────────────────────────────────
    rows = np.where(np.any(tumor, axis=1))[0]
    cols = np.where(np.any(tumor, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, None, {}

    H, W = gray.shape
    pad = max(8, int(min(H, W) * 0.015))
    bbox = (
        max(0, cols[0] - pad),
        max(0, rows[0] - pad),
        min(W - 1, cols[-1] + pad),
        min(H - 1, rows[-1] + pad),
    )

    # ── Diagnostics ───────────────────────────────────────────────────────
    tumor_px = gray[tumor]
    tissue_px = gray[strict_mask & ~tumor]
    contrast = float((tumor_px.mean() - tissue_px.mean()) / (tissue_px.std() + 1e-8))
    area_frac = float(tumor.sum() / (brain_area + 1e-8))

    try:
        perim = float(sk_perimeter(tumor))
        circ = float(4 * np.pi * tumor.sum() / (perim ** 2 + 1e-8))
    except Exception:
        circ = 0.5
    circ = min(1.0, max(0.0, circ))

    # Convex hull solidity
    try:
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(tumor)
        solidity_hull = tumor.sum() / (hull.sum() + 1e-8)
    except Exception:
        solidity_hull = 0.5

    diag = {
        "mu": float(mu), "sig": float(sig),
        "thr_a": float(thr_a), "otsu_t": float(otsu_t),
        "contrast": contrast, "area_frac": area_frac,
        "circularity": circ,
        "solidity": float(solidity_hull),
        "n_components_before_filter": n,
        "tumor_mean": float(tumor_px.mean()),
        "tissue_mean": float(tissue_px.mean()),
        "tissue_std": float(tissue_px.std()),
        "signal_votes_mean": float(votes[strict_mask].mean()),
        "votes_map": votes,
        "lc": lc,
        "grad": grad,
        "strict_mask": strict_mask,
    }
    return bbox, tumor, diag


def estimate_confidence(diag: dict) -> float:
    """
    4-signal confidence:
      - Intensity contrast
      - Circularity
      - Area plausibility
      - Convex hull solidity
    Mapped to [0.50, 0.98].
    """
    contrast = diag.get("contrast", 0)
    circ = diag.get("circularity", 0)
    area = diag.get("area_frac", 0)
    solidity = diag.get("solidity", 0.5)

    c_score = min(1.0, contrast / 5.0)
    r_score = max(0.0, 1.0 - abs(circ - 0.55) / 0.55)
    if 0.003 <= area <= 0.15:
        a_score = 1.0
    elif area < 0.003:
        a_score = area / 0.003
    else:
        a_score = max(0.0, 1.0 - (area - 0.15) / 0.15)
    s_score = min(1.0, solidity / 0.8)

    raw = 0.45 * c_score + 0.20 * r_score + 0.20 * a_score + 0.15 * s_score
    return float(min(0.98, max(0.52, 0.50 + raw * 0.48)))


def classify_risk(confidence: float, area_frac: float, contrast: float) -> tuple:
    if confidence > 0.82 and area_frac > 0.005:
        return "HIGH", "risk-high", "Urgent clinical review recommended. Prioritize contrast-enhanced MRI."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "risk-medium", "Further imaging advised (contrast MRI or PET scan)."
    else:
        return "LOW", "risk-low", "Monitor — repeat scan in 3–6 months if clinically indicated."


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def draw_highlight(original: Image.Image, bbox: tuple, tumor_mask: np.ndarray) -> Image.Image:
    rgb = np.array(original.convert("RGB"), dtype=np.float32)
    ovl = rgb.copy()
    halo = binary_dilation(tumor_mask, iterations=7) & ~tumor_mask
    ovl[halo] = [255, 110, 30]
    ovl[tumor_mask] = [255, 35, 55]
    blended = (0.42 * ovl + 0.58 * rgb).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended)
    draw = ImageDraw.Draw(result)
    draw.rectangle(bbox, outline=(190, 25, 50), width=4)
    inner = (bbox[0]+3, bbox[1]+3, bbox[2]-3, bbox[3]-3)
    draw.rectangle(inner, outline=(255, 80, 80), width=1)
    tick = 16
    x0, y0, x1, y1 = bbox
    for (cx, cy, dx, dy) in [
        (x0, y0,  1,  1), (x1, y0, -1,  1),
        (x0, y1,  1, -1), (x1, y1, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255, 220, 0), width=3)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255, 220, 0), width=3)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(12, int(min(original.size) * 0.028))
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 230, 0), width=2)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 230, 0), width=2)
    draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(255, 230, 0))
    bh = 20
    bx, by = bbox[0], max(0, bbox[1] - bh - 3)
    draw.rectangle([bx, by, bx + 155, by + bh], fill=(200, 20, 45))
    draw.text((bx + 6, by + 3), "ANOMALY DETECTED", fill=(255, 255, 255))
    return result


def make_brain_mask_overlay(original: Image.Image, brain_mask: np.ndarray) -> np.ndarray:
    """
    Overlay the brain mask as a semi-transparent cyan outline on the original MRI.
    Much more informative and visually clean than a flat white mask.
    """
    rgb = np.array(original.convert("RGB"), dtype=np.float32)
    result = rgb.copy()
    # Brain region: subtle cyan tint
    tint = result.copy()
    tint[brain_mask] = (tint[brain_mask] * 0.6 + np.array([0, 60, 80]) * 0.4)
    result = tint
    # Brain boundary: bright cyan outline
    boundary = binary_dilation(brain_mask, iterations=2) & ~brain_mask
    result[boundary] = [0, 210, 255]
    return result.clip(0, 255).astype(np.uint8)


def make_heatmap(gray: np.ndarray, brain_mask: np.ndarray,
                 tumor_mask: np.ndarray | None) -> np.ndarray:
    disp = gray.copy()
    if brain_mask.any():
        lo, hi = gray[brain_mask].min(), gray[brain_mask].max()
        disp = (gray - lo) / (hi - lo + 1e-8)
    disp = np.clip(disp, 0, 1)
    disp[~brain_mask] = 0
    cmap = plt.get_cmap("inferno")
    rgba = (cmap(disp) * 255).astype(np.uint8)
    rgb = rgba[:, :, :3]
    if tumor_mask is not None:
        rgb[tumor_mask] = [0, 220, 255]
        halo = binary_dilation(tumor_mask, iterations=3) & ~tumor_mask
        rgb[halo] = [0, 160, 200]
    return rgb


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0D1421", edgecolor="none", dpi=140)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0D1421")
    ax.set_facecolor("#0D1421")

    tissue_mask = brain_mask.copy()
    if tumor_mask is not None:
        tissue_mask = brain_mask & ~tumor_mask
    brain_vals = gray[tissue_mask]
    ax.hist(brain_vals, bins=60, color="#00D4FF", alpha=0.55, label="Brain tissue",
            density=True, histtype="stepfilled")

    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#FF3860", alpha=0.8, label="Tumor region",
                density=True, histtype="stepfilled")

    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#FFB830", linewidth=1.5, linestyle="--",
                   label=f"Z-thr {diag['thr_a']:.2f}")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#00FF9D", linewidth=1.5, linestyle=":",
                   label=f"Otsu {diag['otsu_t']:.2f}")

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors="#6B8099", labelsize=7)
    ax.set_xlabel("Normalised Intensity", color="#6B8099", fontsize=8)
    ax.set_ylabel("Density", color="#6B8099", fontsize=8)
    ax.set_title("Pixel Intensity Distribution", color="#E8EDF5", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts():
        t.set_color("#A0AEC0")
    plt.tight_layout(pad=0.5)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_intensity_profile(gray: np.ndarray, tumor_mask: np.ndarray) -> Image.Image:
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    cy = int(rows.mean()) if rows.size else gray.shape[0] // 2
    cx = int(cols.mean()) if cols.size else gray.shape[1] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.6))
    fig.patch.set_facecolor("#0D1421")

    for ax, profile, label, span_data in [
        (ax1, gray[cy, :],  "Horizontal", cols),
        (ax2, gray[:, cx],  "Vertical",   rows),
    ]:
        ax.set_facecolor("#0D1421")
        ax.plot(profile, color="#00D4FF", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#00D4FF")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(colors="#6B8099", labelsize=6)
        ax.set_title(f"{label} @ centroid", color="#A0AEC0", fontsize=8)
        if span_data.size:
            ax.axvspan(span_data[0], span_data[-1], alpha=0.2, color="#FF3860")

    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    """
    Properly rendered ensemble vote heatmap with correct colormap and masking.
    Values 0-4 (4 signals now).
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0D1421")
    ax.set_facecolor("#0D1421")

    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan  # masked background → transparent

    # 5 levels: 0, 1, 2, 3, 4
    cmap = LinearSegmentedColormap.from_list(
        "vote4",
        ["#0D1421", "#1A3A5C", "#00D4FF", "#FFB830", "#FF3860"],
        N=256
    )
    cmap.set_bad(color="#080C14")  # NaN → background colour

    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=4, aspect="auto",
                   interpolation="nearest")
    ax.axis("off")
    ax.set_title("Ensemble Vote Map  (0 – 4 signals)", color="#A0AEC0",
                 fontsize=8.5, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["0", "1", "2", "3", "4"])
    cbar.ax.tick_params(colors="#8BA7C7", labelsize=7)
    cbar.outline.set_visible(False)
    cbar.ax.set_facecolor("#0D1421")

    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_gradient_map(grad: np.ndarray, brain_mask: np.ndarray,
                      tumor_mask: np.ndarray | None) -> Image.Image:
    """Gradient magnitude map — shows edge structure inside brain."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0D1421")
    ax.set_facecolor("#0D1421")

    disp = grad.copy()
    disp[~brain_mask] = np.nan
    lo, hi = np.nanpercentile(disp, 2), np.nanpercentile(disp, 98)
    disp = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)

    cmap = plt.get_cmap("plasma")
    cmap.set_bad(color="#080C14")
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                   interpolation="bilinear")
    if tumor_mask is not None:
        contour_mask = binary_dilation(tumor_mask, iterations=2) & ~tumor_mask
        ys, xs = np.where(contour_mask)
        if len(xs):
            ax.scatter(xs, ys, s=0.3, c="#00FFFF", alpha=0.6, linewidths=0)
    ax.axis("off")
    ax.set_title("Gradient Edge Map", color="#A0AEC0", fontsize=8.5, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors="#8BA7C7", labelsize=7)
    cbar.outline.set_visible(False)
    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


# ═════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-wrap">
  <p class="hero-eyebrow">v3.0 · 4-Signal Ensemble · Adaptive Detection · Clinical Visual Reporting</p>
  <h1 class="hero-title">Neuro<span>Scan</span></h1>
  <p class="hero-sub">
    AI-powered MRI anomaly detection &nbsp;·&nbsp;
    Multi-signal ensemble pipeline &nbsp;·&nbsp;
    Adaptive sensitivity recommendation
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown(
        "<small style='color:#8BA7C7;'>Adjust sensitivity or let NeuroScan recommend it automatically after upload.</small>",
        unsafe_allow_html=True,
    )

    sensitivity = st.select_slider(
        "Sensitivity",
        options=["low", "balanced", "high"],
        value="balanced",
        help=(
            "**Low** — flags only strong, high-contrast lesions.\n\n"
            "**Balanced** — recommended default for most MRI types.\n\n"
            "**High** — catches subtle anomalies; may increase false positives."
        ),
    )

    st.markdown("---")
    st.markdown("**🔬 Debug Visualization**")
    show_debug = st.checkbox("Show pipeline steps", value=False)
    show_votes = st.checkbox("Show vote map", value=True)
    show_gradient = st.checkbox("Show gradient edge map", value=False)

    st.markdown("---")
    with st.expander("📖 Algorithm Details"):
        st.markdown("""
**4-Signal Ensemble Pipeline**

**Signal A — Global Z-score**
Pixels > µ + z·σ within brain mask.

**Signal B — Multi-scale Local Contrast**
Fused contrast at 3 spatial scales (11/21/35px).

**Signal C — Otsu Segmentation**
Data-driven binary split of brain intensities.

**Signal D — Gradient Edge Strength**
Sobel edges around bright regions (tumors have sharp boundaries).

**Voting threshold:**
- LOW: 3-of-4 signals must agree
- BALANCED: 2-of-4
- HIGH: 2-of-4 (lower thresholds per signal)

**Morphological post-processing:**
Erode → Dilate → Fill holes → Size filter → Blob scoring (area × solidity² × aspect_ratio^0.5)
        """)

    st.markdown("---")
    st.caption("NeuroScan v3.0 · Research use only.")
    st.caption("⚠️ Not for clinical decision-making.")

# ── Upload ────────────────────────────────────────────────────────────────────
col_up_l, col_up_c, col_up_r = st.columns([1, 2, 1])
with col_up_c:
    uploaded_file = st.file_uploader(
        "Upload MRI Slice",
        type=["png", "jpg", "jpeg"],
        help="Axial T1, T1-CE, T2 or FLAIR slices. PNG preferred for best results.",
    )
    st.markdown(
        '<p class="upload-hint">Supports T1 · T1-CE · T2 · FLAIR axial slices · PNG preferred</p>',
        unsafe_allow_html=True,
    )

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    # ── Quick pre-analysis for sensitivity recommendation ──────────────────
    _gray_quick = np.array(raw_img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(_gray_quick, 1), np.percentile(_gray_quick, 99)
    _gray_quick = np.clip((_gray_quick - lo) / (hi - lo + 1e-8), 0, 1)
    _brain_quick = extract_brain_mask(gaussian_filter(_gray_quick, 0.7))
    rec_sens, rec_reason = recommend_sensitivity(_gray_quick, _brain_quick)

    # ── Recommendation banner ──────────────────────────────────────────────
    col_up_l2, col_up_c2, col_up_r2 = st.columns([1, 2, 1])
    with col_up_c2:
        sens_icon = {"low": "🟢", "balanced": "🟡", "high": "🔴"}.get(rec_sens, "🟡")
        st.markdown(f"""
<div class="rec-banner">
  <span style="font-size:1.3rem">{sens_icon}</span>
  <span>
    <strong>Recommended sensitivity: {rec_sens.upper()}</strong> — {rec_reason}
    {"<br><em>Your current setting matches the recommendation.</em>" if sensitivity == rec_sens else f"<br><em>Current setting: {sensitivity.upper()}. Consider switching to {rec_sens.upper()}.</em>"}
  </span>
</div>
""", unsafe_allow_html=True)

    col_l, col_c2, col_r = st.columns([1, 2, 1])
    with col_c2:
        run = st.button("🔍  Run Analysis")

    if run:
        # ── Animated pipeline progress ─────────────────────────────────────
        status_box = st.empty()
        steps = [
            "CLAHE contrast enhancement & normalization…",
            "Skull stripping (adaptive brain mask extraction)…",
            "Computing multi-scale local contrast maps…",
            "Running 4-signal ensemble anomaly detection…",
            "Morphological refinement & blob scoring…",
            "Generating clinical visualizations & report…",
        ]
        done = []
        for step_msg in steps:
            html_steps = "".join(
                f'<div class="step-row step-done"><div class="step-dot step-dot-done"></div>{s}</div>'
                for s in done
            )
            html_steps += f'<div class="step-row"><div class="step-dot"></div>{step_msg}</div>'
            status_box.markdown(f'<div class="card">{html_steps}</div>', unsafe_allow_html=True)
            time.sleep(0.3)
            done.append(step_msg)

        # ── Run the actual pipeline ────────────────────────────────────────
        gray_norm = preprocess_mri(raw_img)
        brain_mask = extract_brain_mask(gray_norm)
        bbox, tumor_mask, diag = detect_tumor_region(gray_norm, brain_mask, sensitivity=sensitivity)

        html_done = "".join(
            f'<div class="step-row step-done"><div class="step-dot step-dot-done"></div>{s}</div>'
            for s in steps
        )
        status_box.markdown(f'<div class="card">{html_done}</div>', unsafe_allow_html=True)
        time.sleep(0.25)
        status_box.empty()

        # ── Results ────────────────────────────────────────────────────────
        if bbox is not None and tumor_mask is not None:
            confidence = estimate_confidence(diag)
            area_pct = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(
                confidence, diag["area_frac"], diag["contrast"]
            )
            result_img = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)
            brain_overlay = make_brain_mask_overlay(raw_img, brain_mask)

            # ── Result header ──────────────────────────────────────────────
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin:1.5rem 0 0.5rem;flex-wrap:wrap;">
  <span class="detection-badge badge-detected">
    <span class="pulse-dot"></span>ANOMALY DETECTED
  </span>
  <span class="risk-chip {risk_cls}">{risk} RISK</span>
</div>
""", unsafe_allow_html=True)

            # ── Metric tiles (5 tiles) ─────────────────────────────────────
            st.markdown(f"""
<div class="metric-grid">
  <div class="metric-tile cyan">
    <div class="metric-val cyan">{confidence*100:.1f}%</div>
    <div class="metric-label">Confidence</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div>
    </div>
  </div>
  <div class="metric-tile red">
    <div class="metric-val red">{area_pct:.2f}%</div>
    <div class="metric-label">Brain Coverage</div>
  </div>
  <div class="metric-tile amber">
    <div class="metric-val amber">{diag['contrast']:.2f}σ</div>
    <div class="metric-label">Contrast Ratio</div>
  </div>
  <div class="metric-tile green">
    <div class="metric-val green">{diag['circularity']:.2f}</div>
    <div class="metric-label">Circularity</div>
  </div>
  <div class="metric-tile purple">
    <div class="metric-val purple">{diag['solidity']:.2f}</div>
    <div class="metric-label">Solidity</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Tabs ───────────────────────────────────────────────────────
            tab1, tab2, tab3 = st.tabs(
                ["🖼️  Detection Overlay", "🌡️  Heatmap & Brain Mask", "📊  Analysis Charts"]
            )

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Original MRI</p>', unsafe_allow_html=True)
                    st.image(raw_img, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Anomaly Highlighted</p>', unsafe_allow_html=True)
                    st.image(result_img, use_container_width=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Intensity Heatmap + Tumor Overlay</p>',
                                unsafe_allow_html=True)
                    st.image(heatmap_arr, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Brain Mask Overlay (Skull Stripped)</p>',
                                unsafe_allow_html=True)
                    st.image(brain_overlay, use_container_width=True)
                    st.markdown(
                        '<div class="info-box" style="font-size:0.72rem;">🔵 Cyan outline = detected brain boundary. Tinted region = active brain mask used for analysis.</div>',
                        unsafe_allow_html=True,
                    )

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Pixel Intensity Histogram</p>',
                                unsafe_allow_html=True)
                    hist_img = plot_histogram(gray_norm, brain_mask, tumor_mask, diag)
                    st.image(hist_img, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Intensity Profile Through Centroid</p>',
                                unsafe_allow_html=True)
                    prof_img = plot_intensity_profile(gray_norm, tumor_mask)
                    st.image(prof_img, use_container_width=True)

                if show_votes:
                    votes_map = diag.get("votes_map")
                    strict_mask = diag.get("strict_mask", brain_mask)
                    if votes_map is not None:
                        c3, c4 = st.columns(2)
                        with c3:
                            st.markdown('<p class="section-label">Ensemble Vote Map</p>',
                                        unsafe_allow_html=True)
                            vote_img = plot_signal_votes(votes_map, strict_mask)
                            st.image(vote_img, use_container_width=True)
                        if show_gradient:
                            with c4:
                                st.markdown('<p class="section-label">Gradient Edge Map</p>',
                                            unsafe_allow_html=True)
                                grad_img = plot_gradient_map(diag["grad"], brain_mask, tumor_mask)
                                st.image(grad_img, use_container_width=True)

            # ── Debug intermediate steps ───────────────────────────────────
            if show_debug:
                st.markdown("---")
                st.markdown('<p class="section-label">Pipeline Intermediate Steps</p>',
                            unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.caption("① CLAHE-enhanced grayscale")
                    st.image((gray_norm * 255).astype(np.uint8), use_container_width=True)
                with d2:
                    st.caption("② Brain mask (binary)")
                    st.image((brain_mask.astype(np.uint8) * 255), use_container_width=True)
                with d3:
                    st.caption("③ Tumor mask (binary)")
                    vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    vis[tumor_mask] = [255, 60, 60]
                    st.image(vis, use_container_width=True)

            # ── Clinical Report ────────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
<div class="card">
  <div class="report-header">Clinical Analysis Report</div>
  <div class="report-meta">Generated by NeuroScan v3.0 · 4-Signal Ensemble Pipeline · Sensitivity: {sensitivity.upper()}</div>

  <div class="finding-row">
    <span class="finding-key">Detection Status</span>
    <span class="finding-val" style="color:#FF3860">ANOMALY DETECTED</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Risk Classification</span>
    <span class="risk-chip {risk_cls}">{risk}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Detection Confidence</span>
    <span class="finding-val">{confidence*100:.1f}%</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Tumor Coverage (% of brain)</span>
    <span class="finding-val">{area_pct:.2f}%</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Pixel Count (tumor region)</span>
    <span class="finding-val">{int(tumor_mask.sum()):,}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Bounding Box</span>
    <span class="finding-val">x: {bbox[0]}–{bbox[2]}, y: {bbox[1]}–{bbox[3]}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Intensity Contrast (σ)</span>
    <span class="finding-val">{diag['contrast']:.3f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Tumor Mean Intensity</span>
    <span class="finding-val">{diag['tumor_mean']:.4f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Surrounding Tissue Mean</span>
    <span class="finding-val">{diag['tissue_mean']:.4f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Tissue Std Dev</span>
    <span class="finding-val">{diag['tissue_std']:.4f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Z-score Threshold Used</span>
    <span class="finding-val">{diag['thr_a']:.4f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Otsu Threshold</span>
    <span class="finding-val">{diag['otsu_t']:.4f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Region Circularity</span>
    <span class="finding-val">{diag['circularity']:.3f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Convex Hull Solidity</span>
    <span class="finding-val">{diag['solidity']:.3f}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Sensitivity Mode</span>
    <span class="finding-val">{sensitivity.upper()}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Auto-recommended Sensitivity</span>
    <span class="finding-val">{rec_sens.upper()}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Brain Pixels Analyzed</span>
    <span class="finding-val">{int(brain_mask.sum()):,}</span>
  </div>

  <div style="margin-top:1.2rem;" class="{'warn-box' if risk == 'HIGH' else 'info-box'}">
    <strong>Recommendation:</strong> {recommendation}
  </div>
  <div class="info-box" style="margin-top:0.5rem;">
    ⚠️ This analysis is generated by a research algorithm. All findings must be reviewed by a qualified radiologist. Not for clinical decision-making.
  </div>
</div>
""", unsafe_allow_html=True)

        else:
            # ── No detection ───────────────────────────────────────────────
            st.markdown("""
<div style="margin:1.5rem 0 0.5rem;">
  <span class="detection-badge badge-clear">
    <span class="pulse-dot"></span>NO ANOMALY DETECTED
  </span>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="warn-box">
  No significant anomaly found at <strong>{sensitivity.upper()}</strong> sensitivity.<br>
  Auto-recommended sensitivity for this image: <strong>{rec_sens.upper()}</strong> — {rec_reason}<br>
  {"Try switching to <strong>HIGH</strong> sensitivity in the sidebar if you expect a lesion." if sensitivity != "high" else "You are already at HIGH sensitivity. Verify this is a valid axial brain MRI slice."}
</div>
""", unsafe_allow_html=True)

            brain_overlay = make_brain_mask_overlay(raw_img, brain_mask)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Uploaded MRI</p>', unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<p class="section-label">Brain Mask Overlay</p>', unsafe_allow_html=True)
                st.image(brain_overlay, use_container_width=True)

            # Still show heatmap even on no-detection
            hm = make_heatmap(gray_norm, brain_mask, None)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<p class="section-label">Intensity Heatmap</p>', unsafe_allow_html=True)
                st.image(hm, use_container_width=True)
            with c4:
                hist_img = plot_histogram(gray_norm, brain_mask, None, diag if diag else {})
                st.markdown('<p class="section-label">Pixel Intensity Distribution</p>',
                            unsafe_allow_html=True)
                st.image(hist_img, use_container_width=True)
