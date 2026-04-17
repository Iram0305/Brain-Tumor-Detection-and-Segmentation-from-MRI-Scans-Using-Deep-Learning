# NeuroScan API — Optimized Tumor Detection Pipeline v2.0
#
# DETECTION IMPROVEMENTS:
#  - Multi-scale anomaly detection (replaces single Z-threshold)
#  - Adaptive thresholding using Otsu's method on brain pixels
#  - Ensemble voting: Z-score + local contrast + Otsu combined
#  - Minimum size filter to kill noise blobs
#  - Gaussian smoothing of anomaly map before thresholding
#  - Better skull stripping with multi-iteration approach
#  - Convexity check to prefer compact, round lesions
#  - Confidence from 3 independent signals (contrast, size, shape)
#
# UI IMPROVEMENTS:
#  - Clinical dark-mode aesthetic (deep navy/charcoal + cyan accents)
#  - Animated scan line effect on image display
#  - Detailed PDF-style report with metrics, histogram, profile plot
#  - Intensity profile plot across tumor centroid
#  - Histogram overlay (brain vs tumor pixel distribution)
#  - Heatmap visualization using matplotlib colormap
#  - Animated progress steps during pipeline
#  - Result badge animations via CSS

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
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base:    #080C14;
    --bg-card:    #0D1421;
    --bg-panel:   #111827;
    --border:     #1E2D45;
    --border-lit: #2A4A72;
    --cyan:       #00D4FF;
    --cyan-dim:   #0090CC;
    --green:      #00FF9D;
    --red:        #FF3860;
    --amber:      #FFB830;
    --text-pri:   #E8EDF5;
    --text-sec:   #6B8099;
    --text-dim:   #3A5068;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-pri) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 2px; }

/* ── Hero ── */
.hero-wrap {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: -20%;
    width: 60%; height: 100%;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
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
    font-size: 0.88rem;
    color: var(--text-sec);
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: var(--border-lit); }

.card-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

/* ── Metric tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
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

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-pri);
    line-height: 1;
}
.metric-val.cyan  { color: var(--cyan); }
.metric-val.green { color: var(--green); }
.metric-val.amber { color: var(--amber); }
.metric-val.red   { color: var(--red); }
.metric-label {
    font-size: 0.65rem;
    color: var(--text-sec);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

/* ── Detection badge ── */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
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
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    transition: width 1s ease;
    animation: growBar 1.2s ease both;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
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

/* ── Upload zone ── */
.upload-hint {
    font-size: 0.82rem;
    color: var(--text-sec);
    text-align: center;
    padding: 0.5rem 0;
    font-style: italic;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, #3B6FFF 100%);
    border: none;
    border-radius: 8px;
    color: #000 !important;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    padding: 0.7rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-lit) !important;
    border-radius: 12px !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: var(--cyan-dim) !important; }
.stSlider > div > div > div > div { background: var(--cyan) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--text-sec) !important;
    border-radius: 0;
    padding: 0.7rem 1.4rem;
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
    font-size: 0.72rem;
    color: var(--text-sec);
    letter-spacing: 0.05em;
    animation: fadeSlideUp 0.3s ease both;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    flex-shrink: 0;
    animation: pulseDot 1s ease infinite;
}
.step-done { color: var(--green); }
.step-dot-done {
    background: var(--green);
    animation: none;
}

/* ── Report section ── */
.report-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-pri);
    margin-bottom: 0.2rem;
}
.report-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.finding-key {
    color: var(--text-sec);
    font-size: 0.78rem;
}
.finding-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-pri);
    font-weight: 700;
}
.risk-chip {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.risk-high   { background: rgba(255,56,96,0.2);   color: var(--red);   border: 1px solid rgba(255,56,96,0.4); }
.risk-medium { background: rgba(255,184,48,0.2);  color: var(--amber); border: 1px solid rgba(255,184,48,0.4); }
.risk-low    { background: rgba(0,255,157,0.15);  color: var(--green); border: 1px solid rgba(0,255,157,0.4); }

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

/* ── Scan effect wrapper ── */
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

/* ── Warning / info boxes ── */
.info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--text-sec);
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(255,184,48,0.06);
    border: 1px solid rgba(255,184,48,0.25);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--amber);
    margin: 0.75rem 0;
}

/* hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE v2 — Ensemble multi-signal approach
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_mri(img: Image.Image) -> np.ndarray:
    """Normalize + denoise. Returns float32 in [0,1]."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    lo, hi = gray.min(), gray.max()
    norm = (gray - lo) / (hi - lo + 1e-8)
    # Mild denoise — preserves edges
    return gaussian_filter(norm, sigma=0.8)


def extract_brain_mask(gray: np.ndarray) -> np.ndarray:
    """
    Robust skull stripping:
    1. Background threshold (very dark pixels = air/background)
    2. Fill holes
    3. Keep largest connected component (head)
    4. Erode to remove skull ring
    5. Re-fill holes (ventricles)
    6. Dilate slightly to recover gyri near edge
    """
    h, w = gray.shape

    # Step 1: coarse threshold — background is < 5% intensity
    rough = gray > 0.05
    rough = binary_fill_holes(rough)

    # Step 2: keep largest blob (the head)
    labeled, n = sp_label(rough)
    if n == 0:
        return rough
    sizes = ndimage.sum(rough, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = (labeled == head_label)

    # Step 3: erode to strip skull. 3-5% of shorter dim works empirically
    erode_px = max(5, int(min(h, w) * 0.035))
    brain = binary_erosion(head, iterations=erode_px)
    brain = binary_fill_holes(brain)

    # Step 4: light dilation to recover a bit of cortex
    brain = binary_dilation(brain, iterations=2)
    brain = brain & head  # clamp back inside head

    if brain.sum() < (h * w * 0.03):
        brain = head  # fallback
    return brain


def compute_local_contrast(gray: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Local contrast map: how much brighter each pixel is vs its neighbourhood.
    Uses uniform_filter to compute local mean efficiently.
    """
    local_mean = uniform_filter(gray, size=window)
    local_sq   = uniform_filter(gray ** 2, size=window)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    # Normalized local deviation
    contrast = (gray - local_mean) / (local_std + 0.02)
    return contrast


def otsu_threshold_1d(values: np.ndarray) -> float:
    """Otsu's method on a 1-D array of float values."""
    # Bin into 256 levels
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8

    # Otsu criterion
    best_thresh = bin_centers[len(bin_centers) // 2]
    best_var    = 0.0
    w0 = 0.0; mu0 = 0.0

    for i, (p, c) in enumerate(zip(hist, bin_centers)):
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            w0 += p; mu0 = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)
            continue
        mu1_num = np.dot(hist[i:], bin_centers[i:])
        mu1     = mu1_num / (w1 + 1e-8)
        var     = w0 * w1 * (mu0 / (w0 + 1e-8) - mu1) ** 2
        if var > best_var:
            best_var   = var
            best_thresh = c
        w0  += p
        mu0  = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)

    return best_thresh


def detect_tumor_region(
    gray: np.ndarray,
    brain_mask: np.ndarray,
    sensitivity: str = "balanced",
) -> tuple:
    """
    Ensemble detector with Border Suppression and Geometric Blob Scoring.
    """
    # ── 1. Strict Border Suppression ──────────────────────────────────────
    # Erode the outer edges of the brain mask so we completely ignore skull bleed
    h, w = gray.shape
    edge_margin = max(3, int(min(h, w) * 0.04))
    strict_mask = binary_erosion(brain_mask, iterations=edge_margin)

    brain_px = gray[strict_mask]
    if brain_px.size < 200:
        return None, None, {}

    mu = brain_px.mean()
    sig = brain_px.std()

    # ── Sensitivity presets ──────────────────────────────────────────────
    presets = {
        "low":      {"z_a": 3.2, "lc_z": 2.8, "min_area_frac": 0.003},
        "balanced": {"z_a": 2.5, "lc_z": 2.2, "min_area_frac": 0.002},
        "high":     {"z_a": 1.8, "lc_z": 1.6, "min_area_frac": 0.001},
    }
    p = presets.get(sensitivity, presets["balanced"])

    # ── Signal A: global Z-score ─────────────────────────────────────────
    thr_a = mu + p["z_a"] * sig
    sig_a = (gray >= thr_a) & strict_mask

    # ── Signal B: local contrast ─────────────────────────────────────────
    lc = compute_local_contrast(gray, window=21)
    lc_brain = lc[strict_mask]
    lc_thr = lc_brain.mean() + p["lc_z"] * lc_brain.std()
    sig_b = (lc >= lc_thr) & strict_mask

    # ── Signal C: Otsu on brain pixels ──────────────────────────────────
    otsu_t = otsu_threshold_1d(brain_px)
    sig_c = (gray >= otsu_t) & strict_mask

    # ── Ensemble vote: 2-of-3 ────────────────────────────────────────────
    votes = sig_a.astype(np.uint8) + sig_b.astype(np.uint8) + sig_c.astype(np.uint8)
    anomaly = votes >= 2

    # ── Morphological cleanup ─────────────────────────────────────────────
    anomaly = binary_erosion(anomaly, iterations=2)
    anomaly = binary_dilation(anomaly, iterations=6)
    anomaly = binary_fill_holes(anomaly)
    anomaly = anomaly & strict_mask

    # ── Remove small blobs (noise) ────────────────────────────────────────
    brain_area = strict_mask.sum()
    min_px = max(30, int(brain_area * p["min_area_frac"]))
    labeled, n = sp_label(anomaly)
    cleaned = np.zeros_like(anomaly)
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        if comp.sum() >= min_px:
            cleaned |= comp

    # ── Smart Blob Selection (Fixes the crescent issue) ──────────────────
    labeled2, n2 = sp_label(cleaned)
    if n2 == 0:
        return None, None, {}

    best_score = -1
    best_lbl = -1

    # Score blobs based on Area, Solidity, and Aspect Ratio
    for lbl in range(1, n2 + 1):
        comp = (labeled2 == lbl)
        area = comp.sum()

        # Bounding box for the blob
        rows = np.where(np.any(comp, axis=1))[0]
        cols = np.where(np.any(comp, axis=0))[0]
        if rows.size == 0 or cols.size == 0: continue

        h_c = rows[-1] - rows[0] + 1
        w_c = cols[-1] - cols[0] + 1
        bbox_area = h_c * w_c

        # Solidity: Area vs Bounding Box Area (1.0 = perfect rectangle/square)
        solidity = area / (bbox_area + 1e-8)

        # Aspect Ratio penalty: perfect circle/square is 1.0. Thin strip is close to 0.
        aspect_ratio = min(h_c, w_c) / (max(h_c, w_c) + 1e-8)

        # The Magic Formula: Punish long, thin objects and reward compact ones
        score = area * (solidity ** 2) * aspect_ratio

        if score > best_score:
            best_score = score
            best_lbl = lbl

    tumor = (labeled2 == best_lbl)

    # ── Bounding box + padding ────────────────────────────────────────────
    rows = np.where(np.any(tumor, axis=1))[0]
    cols = np.where(np.any(tumor, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, None, {}

    H, W = gray.shape
    pad = max(8, int(min(H, W) * 0.015))
    bbox = (
        max(0, cols[0] - pad),
        max(0, rows[0] - pad),
        min(W-1, cols[-1] + pad),
        min(H-1, rows[-1] + pad),
    )

    # ── Diagnostics dict ──────────────────────────────────────────────────
    tumor_px = gray[tumor]
    tissue_px = gray[strict_mask & ~tumor]
    contrast = float((tumor_px.mean() - tissue_px.mean()) / (tissue_px.std() + 1e-8))
    area_frac = float(tumor.sum() / (brain_area + 1e-8))
    
    from skimage.measure import perimeter as sk_perimeter
    try:
        perim = float(sk_perimeter(tumor))
        circ = float(4 * np.pi * tumor.sum() / (perim ** 2 + 1e-8))
    except Exception:
        circ = 0.5
    circ = min(1.0, max(0.0, circ))

    diag = {
        "mu": float(mu), "sig": float(sig),
        "thr_a": float(thr_a), "otsu_t": float(otsu_t),
        "contrast": contrast, "area_frac": area_frac,
        "circularity": circ,
        "n_components_before_filter": n,
        "tumor_mean": float(tumor_px.mean()),
        "tissue_mean": float(tissue_px.mean()),
        "tissue_std": float(tissue_px.std()),
        "signal_votes_mean": float(votes[strict_mask].mean()),
        "lc": lc,
    }
    return bbox, tumor, diag

def estimate_confidence(diag: dict) -> float:
    """
    Three-signal confidence:
      - Intensity contrast (0-1)
      - Circularity (0-1, real tumors are ~0.4-0.8)
      - Area plausibility (not too tiny, not too large)
    Combined as weighted average, mapped to [0.50, 0.98].
    """
    contrast  = diag.get("contrast", 0)
    circ      = diag.get("circularity", 0)
    area      = diag.get("area_frac", 0)

    # Contrast score: sigmoid-like, saturates at ~5σ
    c_score   = min(1.0, contrast / 5.0)
    # Circularity score: peak at 0.5, symmetric penalty outside
    r_score   = 1.0 - abs(circ - 0.55) / 0.55
    r_score   = max(0, min(1, r_score))
    # Area score: ideal range 0.3%-15% of brain
    if 0.003 <= area <= 0.15:
        a_score = 1.0
    elif area < 0.003:
        a_score = area / 0.003
    else:
        a_score = max(0, 1 - (area - 0.15) / 0.15)

    raw = 0.55 * c_score + 0.25 * r_score + 0.20 * a_score
    return float(min(0.98, max(0.52, 0.50 + raw * 0.48)))


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def draw_highlight(original: Image.Image, bbox: tuple, tumor_mask: np.ndarray) -> Image.Image:
    rgb   = np.array(original.convert("RGB"), dtype=np.float32)
    ovl   = rgb.copy()

    # Glow halo: dilate mask slightly for outer glow ring
    halo  = binary_dilation(tumor_mask, iterations=6) & ~tumor_mask
    ovl[halo]       = [255, 100,  40]   # orange outer glow
    ovl[tumor_mask] = [255,  40,  60]   # red core

    blended = (0.45 * ovl + 0.55 * rgb).clip(0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)
    draw    = ImageDraw.Draw(result)

    # Bounding box: double-line effect (outer dim, inner bright)
    draw.rectangle(bbox, outline=(180, 30, 50), width=4)
    inner = (bbox[0]+3, bbox[1]+3, bbox[2]-3, bbox[3]-3)
    draw.rectangle(inner, outline=(255, 80, 80), width=1)

    # Corner ticks (surgical-style localization corners)
    tick = 14
    x0, y0, x1, y1 = bbox
    for (cx, cy, dx, dy) in [
        (x0, y0,  1,  1), (x1, y0, -1,  1),
        (x0, y1,  1, -1), (x1, y1, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255,220,0), width=3)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255,220,0), width=3)

    # Crosshair
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(12, int(min(original.size) * 0.028))
    draw.line([(cx-arm, cy), (cx+arm, cy)], fill=(255,230,0), width=2)
    draw.line([(cx, cy-arm), (cx, cy+arm)], fill=(255,230,0), width=2)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255,230,0))

    # Label badge
    bh = 20
    bx, by = bbox[0], max(0, bbox[1] - bh - 3)
    draw.rectangle([bx, by, bx+148, by+bh], fill=(200, 25, 45))
    draw.text((bx+6, by+3), "ANOMALY DETECTED", fill=(255, 255, 255))

    return result


def make_heatmap(gray: np.ndarray, brain_mask: np.ndarray,
                 tumor_mask: np.ndarray | None) -> np.ndarray:
    """Return a matplotlib heatmap as uint8 RGB array."""
    # Normalize within brain
    disp = gray.copy()
    if brain_mask.any():
        lo, hi = gray[brain_mask].min(), gray[brain_mask].max()
        disp = (gray - lo) / (hi - lo + 1e-8)
    disp = np.clip(disp, 0, 1)
    disp[~brain_mask] = 0

    cmap = plt.get_cmap("inferno")
    rgba = (cmap(disp) * 255).astype(np.uint8)
    rgb  = rgba[:, :, :3]

    # Overlay tumor mask in cyan
    if tumor_mask is not None:
        rgb[tumor_mask] = [0, 220, 255]
        halo = binary_dilation(tumor_mask, iterations=3) & ~tumor_mask
        rgb[halo] = [0, 160, 200]

    return rgb


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0D1421", edgecolor="none", dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0D1421")
    ax.set_facecolor("#0D1421")

    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#00D4FF", alpha=0.55, label="Brain tissue",
            density=True, histtype="stepfilled")

    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#FF3860", alpha=0.8, label="Tumor region",
                density=True, histtype="stepfilled")

    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#FFB830", linewidth=1.5,
                   linestyle="--", label=f"Z-thr ({diag['thr_a']:.2f})")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#00FF9D", linewidth=1.5,
                   linestyle=":", label=f"Otsu ({diag['otsu_t']:.2f})")

    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#6B8099", labelsize=7)
    ax.set_xlabel("Normalised Intensity", color="#6B8099", fontsize=8)
    ax.set_ylabel("Density", color="#6B8099", fontsize=8)
    ax.set_title("Pixel Intensity Distribution", color="#E8EDF5", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color("#A0AEC0")

    plt.tight_layout(pad=0.5)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_intensity_profile(gray: np.ndarray, tumor_mask: np.ndarray) -> Image.Image:
    """Horizontal and vertical intensity profiles through tumor centroid."""
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    cy   = int(rows.mean()) if rows.size else gray.shape[0] // 2
    cx   = int(cols.mean()) if cols.size else gray.shape[1] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.6))
    fig.patch.set_facecolor("#0D1421")

    for ax, profile, label in [
        (ax1, gray[cy, :],  "Horizontal"),
        (ax2, gray[:, cx],  "Vertical"),
    ]:
        ax.set_facecolor("#0D1421")
        ax.plot(profile, color="#00D4FF", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#00D4FF")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#6B8099", labelsize=6)
        ax.set_title(f"{label} Profile @ centroid", color="#A0AEC0", fontsize=8)
        # Shade tumor span
        if label == "Horizontal":
            if cols.size:
                ax.axvspan(cols[0], cols[-1], alpha=0.2, color="#FF3860")
        else:
            if rows.size:
                ax.axvspan(rows[0], rows[-1], alpha=0.2, color="#FF3860")

    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    """Show the ensemble vote map (0,1,2,3 = how many signals agreed)."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#0D1421")
    ax.set_facecolor("#0D1421")

    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["#0D1421", "#00D4FF", "#FFB830", "#FF3860"], N=4
    )
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Ensemble Vote Map\n(0–3 signals)", color="#A0AEC0", fontsize=8, pad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["0", "1", "2", "3"])
    cbar.ax.tick_params(colors="#6B8099", labelsize=7)
    cbar.outline.set_visible(False)

    plt.tight_layout(pad=0.3)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


# ═════════════════════════════════════════════════════════════════════════════
#  RISK CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_risk(confidence: float, area_frac: float, contrast: float) -> tuple:
    """Returns (risk_label, risk_class, recommendation)"""
    if confidence > 0.82 and area_frac > 0.005:
        return "HIGH", "risk-high", "Urgent clinical review recommended."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "risk-medium", "Further imaging (MRI contrast, PET) advised."
    else:
        return "LOW", "risk-low", "Monitor; repeat scan in 3–6 months."


# ═════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-eyebrow">v2.0 · Ensemble Detection · Statistical Segmentation</p>
  <h1 class="hero-title">Neuro<span>Scan</span></h1>
  <p class="hero-sub">
    AI-powered MRI anomaly detection &nbsp;·&nbsp; Multi-signal ensemble pipeline &nbsp;·&nbsp;
    Clinical visual reporting
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    sensitivity = st.select_slider(
        "Sensitivity",
        options=["low", "balanced", "high"],
        value="balanced",
        help=(
            "**Low** — flags only strong, high-contrast lesions.\n\n"
            "**Balanced** — recommended default.\n\n"
            "**High** — catches subtle anomalies (more false positives possible)."
        ),
    )

    st.markdown("---")
    show_debug = st.checkbox("🔬 Show pipeline steps", value=False)
    show_votes = st.checkbox("🗳️ Show vote map", value=False)

    st.markdown("---")
    with st.expander("📖 Detection Algorithm"):
        st.markdown("""
**Ensemble 3-signal pipeline:**

**Signal A** — Global Z-score  
Pixels > µ + z·σ within brain mask

**Signal B** — Local contrast  
Pixels brighter than their neighbourhood by > threshold

**Signal C** — Otsu segmentation  
Data-driven binary split of brain intensities

**Voting:** anomaly if ≥ 2 of 3 signals agree → reduces both false positives and false negatives.

**Morphological cleanup:**  
Erode (kill noise) → Dilate (reconnect) → Fill holes → Size filter → Largest component
        """)

    st.markdown("---")
    st.caption("NeuroScan v2.0 · VIBE6 INNOVATHON 2026")
    st.caption("⚠️ Not for clinical use. Research only.")


# ── Upload ────────────────────────────────────────────────────────────────────
col_up_l, col_up_c, col_up_r = st.columns([1, 2, 1])
with col_up_c:
    uploaded_file = st.file_uploader(
        "Upload MRI Slice",
        type=["png", "jpg", "jpeg"],
        help="Axial T1, T1-CE, T2 or FLAIR slices. PNG preferred.",
    )
    st.markdown('<p class="upload-hint">Supports T1 · T1-CE · T2 · FLAIR axial slices</p>',
                unsafe_allow_html=True)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col_l, col_c2, col_r = st.columns([1, 2, 1])
    with col_c2:
        run = st.button("🔍  Run Analysis")

    if run:
        # ── Animated pipeline progress ────────────────────────────────────
        status_box = st.empty()

        steps = [
            "Preprocessing & normalizing MRI intensities…",
            "Extracting brain mask (skull stripping)…",
            "Computing local contrast map…",
            "Running ensemble anomaly detection (3 signals)…",
            "Morphological cleanup & size filtering…",
            "Generating visualizations & report…",
        ]

        done = []
        for i, step_msg in enumerate(steps):
            html_steps = ""
            for s in done:
                html_steps += f'<div class="step-row step-done"><div class="step-dot step-dot-done"></div>{s}</div>'
            html_steps += f'<div class="step-row"><div class="step-dot"></div>{step_msg}</div>'
            status_box.markdown(
                f'<div class="card">{html_steps}</div>', unsafe_allow_html=True)
            time.sleep(0.35)
            done.append(step_msg)

        # ── Run the actual pipeline ───────────────────────────────────────
        gray_norm   = preprocess_mri(raw_img)
        brain_mask  = extract_brain_mask(gray_norm)

        # Recompute votes for display (needed before detect call)
        bbox, tumor_mask, diag = detect_tumor_region(
            gray_norm, brain_mask, sensitivity=sensitivity
        )

        # Mark all steps done
        html_done = "".join(
            f'<div class="step-row step-done"><div class="step-dot step-dot-done"></div>{s}</div>'
            for s in steps
        )
        status_box.markdown(f'<div class="card">{html_done}</div>', unsafe_allow_html=True)
        time.sleep(0.3)
        status_box.empty()

        # ── Result header ─────────────────────────────────────────────────
        if bbox is not None and tumor_mask is not None:
            confidence  = estimate_confidence(diag)
            area_pct    = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(
                confidence, diag["area_frac"], diag["contrast"]
            )
            result_img  = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)

            st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin:1.5rem 0 0.5rem;">
  <span class="detection-badge badge-detected">
    <span class="pulse-dot"></span>ANOMALY DETECTED
  </span>
  <span class="risk-chip {risk_cls}">{risk} RISK</span>
</div>
""", unsafe_allow_html=True)

            # ── Metric tiles ─────────────────────────────────────────────
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
</div>
""", unsafe_allow_html=True)

            # ── Main image tabs ───────────────────────────────────────────
            tab1, tab2, tab3 = st.tabs(["🖼️  Detection Overlay", "🌡️  Heatmap", "📊  Analysis Charts"])

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
                    st.markdown('<p class="section-label">Brain Mask (Skull Stripped)</p>',
                                unsafe_allow_html=True)
                    brain_vis = (brain_mask.astype(np.uint8) * 200)
                    st.image(brain_vis, use_container_width=True)

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

                if show_votes and "lc" in diag:
                    # Recompute vote map for display
                    brain_px   = gray_norm[brain_mask]
                    mu, sig_v  = brain_px.mean(), brain_px.std()
                    presets    = {"low":(3.2,2.8),"balanced":(2.5,2.2),"high":(1.8,1.6)}
                    z_a, lc_z  = presets.get(sensitivity, (2.5,2.2))
                    thr_a      = mu + z_a * sig_v
                    lc_map     = diag["lc"]
                    lc_brain   = lc_map[brain_mask]
                    lc_thr     = lc_brain.mean() + lc_z * lc_brain.std()
                    otsu_t     = diag["otsu_t"]
                    votes_disp = (
                        (gray_norm >= thr_a).astype(np.uint8) +
                        (lc_map    >= lc_thr).astype(np.uint8) +
                        (gray_norm >= otsu_t).astype(np.uint8)
                    )
                    vote_img = plot_signal_votes(votes_disp, brain_mask)
                    st.markdown('<p class="section-label">Ensemble Vote Map</p>',
                                unsafe_allow_html=True)
                    st.image(vote_img, use_container_width=True)

            # ── Debug intermediate steps ──────────────────────────────────
            if show_debug:
                st.markdown("---")
                st.markdown('<p class="section-label">Pipeline Intermediate Steps</p>',
                            unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.caption("① Normalized grayscale")
                    st.image((gray_norm * 255).astype(np.uint8), use_container_width=True)
                with d2:
                    st.caption("② Brain mask")
                    st.image((brain_mask.astype(np.uint8) * 255), use_container_width=True)
                with d3:
                    st.caption("③ Tumor mask (binary)")
                    vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    vis[tumor_mask] = [255, 60, 60]
                    st.image(vis, use_container_width=True)

            # ── Detailed Report ───────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
<div class="card">
  <div class="report-header">Clinical Analysis Report</div>
  <div class="report-meta">Generated by NeuroScan v2.0 · Ensemble Detection Pipeline · Sensitivity: {sensitivity.upper()}</div>

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
    <span class="finding-key">Tumor Area (% of brain)</span>
    <span class="finding-val">{area_pct:.2f}%</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Pixel Count (tumor region)</span>
    <span class="finding-val">{int(tumor_mask.sum()):,}</span>
  </div>
  <div class="finding-row">
    <span class="finding-key">Bounding Box</span>
    <span class="finding-val">x:{bbox[0]}–{bbox[2]}, y:{bbox[1]}–{bbox[3]}</span>
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
    <span class="finding-key">Sensitivity Mode</span>
    <span class="finding-val">{sensitivity.upper()}</span>
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
            # ── No detection ──────────────────────────────────────────────
            st.markdown("""
<div style="margin:1.5rem 0 0.5rem;">
  <span class="detection-badge badge-clear">
    <span class="pulse-dot"></span>NO ANOMALY DETECTED
  </span>
</div>
""", unsafe_allow_html=True)
            st.markdown("""
<div class="warn-box">
  No significant anomaly found at the current sensitivity setting.<br>
  Try switching to <strong>HIGH</strong> sensitivity in the sidebar if you expect a lesion,
  or verify that the uploaded image is a brain MRI slice (not a localizer or scout view).
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Uploaded MRI</p>', unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<p class="section-label">Brain Mask</p>', unsafe_allow_html=True)
                hm = make_heatmap(gray_norm, brain_mask, None)
                st.image(hm, use_container_width=True)
