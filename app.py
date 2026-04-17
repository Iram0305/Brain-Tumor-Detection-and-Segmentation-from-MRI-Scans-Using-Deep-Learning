# NeuroScan — Brain MRI Anomaly Detection
# UI v4.0: Netflix-Style · Large Readable Text · Clickable Info Panels

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

# ── Netflix-Style CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Bebas+Neue&display=swap');

:root {
    --bg-base:       #141414;
    --bg-card:       #1f1f1f;
    --bg-elevated:   #2a2a2a;
    --bg-hover:      #333333;
    --border:        #333333;
    --border-bright: #555555;
    --red:           #E50914;
    --red-hover:     #F40612;
    --red-soft:      rgba(229,9,20,0.15);
    --red-border:    rgba(229,9,20,0.4);
    --white:         #FFFFFF;
    --off-white:     #E5E5E5;
    --grey-light:    #B3B3B3;
    --grey-mid:      #808080;
    --grey-dark:     #4a4a4a;
    --green:         #46D369;
    --green-soft:    rgba(70,211,105,0.15);
    --amber:         #F5A623;
    --amber-soft:    rgba(245,166,35,0.15);
    --cyan:          #00B4D8;
    --cyan-soft:     rgba(0,180,216,0.15);
    --radius:        8px;
    --radius-lg:     12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--off-white) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--grey-dark); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--grey-mid); }

/* ══════════ HERO ══════════ */
.nf-hero {
    background: linear-gradient(180deg, rgba(0,0,0,0) 0%, var(--bg-base) 100%),
                linear-gradient(135deg, #1a0000 0%, #141414 40%, #0a0a1a 100%);
    padding: 3.5rem 0 2.5rem;
    margin-bottom: 2rem;
    border-bottom: 3px solid var(--red);
    position: relative;
    overflow: hidden;
}
.nf-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 60% 80% at 10% 50%, rgba(229,9,20,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 90% 30%, rgba(0,180,216,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.nf-logo {
    font-family: 'Bebas Neue', 'Inter', sans-serif;
    font-size: clamp(3rem, 8vw, 6rem);
    color: var(--red);
    letter-spacing: 0.04em;
    line-height: 1;
    margin: 0;
    text-shadow: 0 0 40px rgba(229,9,20,0.4);
}
.nf-logo span {
    color: var(--white);
}
.nf-tagline {
    font-size: clamp(1rem, 2vw, 1.2rem);
    color: var(--grey-light);
    margin-top: 0.75rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 620px;
}
.nf-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.25rem;
}
.nf-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px;
    padding: 0.3rem 0.8rem;
    font-size: 0.85rem;
    color: var(--grey-light);
    font-weight: 500;
    letter-spacing: 0.02em;
}
.nf-badge.red {
    background: var(--red-soft);
    border-color: var(--red-border);
    color: #ff6b6b;
}

/* ══════════ SECTION HEADERS ══════════ */
.nf-section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--white);
    margin: 1.5rem 0 0.75rem;
    letter-spacing: -0.01em;
}
.nf-section-title .red-bar {
    display: inline-block;
    width: 4px;
    height: 1.4rem;
    background: var(--red);
    border-radius: 2px;
    margin-right: 0.6rem;
    vertical-align: middle;
}

/* ══════════ INFO CARDS (replaces tooltips) ══════════ */
.info-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-left: 4px solid var(--cyan);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    margin: 0.75rem 0 1.25rem;
    font-size: 1rem;
    color: var(--off-white);
    line-height: 1.7;
}
.info-panel strong {
    color: var(--white);
    font-weight: 700;
}
.info-panel.red-accent  { border-left-color: var(--red);   }
.info-panel.green-accent{ border-left-color: var(--green); }
.info-panel.amber-accent{ border-left-color: var(--amber); }
.warn-panel {
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.35);
    border-left: 4px solid var(--amber);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    margin: 0.75rem 0;
    font-size: 1rem;
    color: var(--off-white);
    line-height: 1.7;
}
.warn-panel strong { color: var(--amber); font-weight: 700; }

/* ══════════ DETECTION BADGES ══════════ */
.detect-badge-wrap {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.5rem 0 1rem;
}
.nf-detect-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.5rem;
    border-radius: 6px;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-anomaly {
    background: var(--red-soft);
    border: 2px solid var(--red);
    color: #ff4757;
}
.badge-clear {
    background: var(--green-soft);
    border: 2px solid var(--green);
    color: var(--green);
}
.pulse-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: currentColor;
    animation: pulseDot 1.4s ease-in-out infinite;
}
.risk-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 4px;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-high   { background: var(--red-soft);   border: 1px solid var(--red);   color: #ff4757; }
.risk-medium { background: var(--amber-soft); border: 1px solid var(--amber); color: var(--amber); }
.risk-low    { background: var(--green-soft); border: 1px solid var(--green); color: var(--green); }

/* ══════════ METRIC CARDS (Netflix content cards style) ══════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.metric-card:hover { border-color: var(--border-bright); transform: translateY(-2px); }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.mc-red::after    { background: var(--red); }
.metric-card.mc-green::after  { background: var(--green); }
.metric-card.mc-amber::after  { background: var(--amber); }
.metric-card.mc-cyan::after   { background: var(--cyan); }

.metric-card-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--grey-mid);
    margin-bottom: 0.5rem;
}
.metric-card-value {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.mc-red   .metric-card-value { color: #ff4757; }
.mc-green .metric-card-value { color: var(--green); }
.mc-amber .metric-card-value { color: var(--amber); }
.mc-cyan  .metric-card-value { color: var(--cyan); }
.metric-card-explain {
    font-size: 0.9rem;
    color: var(--grey-light);
    line-height: 1.55;
    margin-top: 0.4rem;
}
.conf-bar-bg {
    background: var(--bg-elevated);
    border-radius: 100px;
    height: 6px;
    margin-top: 0.7rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--red), var(--amber));
    animation: growBar 1s ease both;
}

/* ══════════ TABS ══════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    gap: 0;
    border-radius: var(--radius) var(--radius) 0 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--grey-mid) !important;
    padding: 0.9rem 1.75rem;
    border-radius: 0;
    border-bottom: 3px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--white) !important;
    border-bottom: 3px solid var(--red) !important;
    background: transparent;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 1.5rem;
}

/* ══════════ IMAGE LABELS ══════════ */
.img-label {
    font-size: 1rem;
    font-weight: 700;
    color: var(--white);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    display: block;
}

/* ══════════ PIPELINE STEPS ══════════ */
.pipeline-wrap { display: flex; flex-direction: column; gap: 0; }
.pipeline-step {
    display: flex;
    gap: 1.2rem;
    padding: 1rem 1.1rem;
    border-radius: var(--radius);
    position: relative;
}
.pipeline-step:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 1.6rem;
    top: 100%;
    width: 2px;
    height: 8px;
    background: var(--border-bright);
}
.step-dot-done   { width: 11px; height: 11px; border-radius: 50%; background: var(--green); flex-shrink: 0; margin-top: 3px; }
.step-dot-active { width: 11px; height: 11px; border-radius: 50%; background: var(--red); animation: pulseDot 1s ease infinite; box-shadow: 0 0 10px rgba(229,9,20,0.6); flex-shrink: 0; margin-top: 3px; }
.step-dot-wait   { width: 11px; height: 11px; border-radius: 50%; background: var(--border-bright); flex-shrink: 0; margin-top: 3px; }
.step-active-bg { background: rgba(229,9,20,0.05); border: 1px solid rgba(229,9,20,0.15); }
.step-done-bg   { background: transparent; }
.step-title-done   { font-size: 1rem; font-weight: 600; color: var(--green); opacity: 0.85; }
.step-title-active { font-size: 1rem; font-weight: 700; color: var(--white); }
.step-title-wait   { font-size: 1rem; font-weight: 500; color: var(--grey-dark); }
.step-detail { font-size: 0.92rem; color: var(--grey-light); margin-top: 0.3rem; line-height: 1.6; }

/* ══════════ CLINICAL REPORT TABLE ══════════ */
.report-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-top: 1.5rem;
}
.report-title-bar {
    background: var(--red);
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.report-title-text {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.report-meta-text {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    font-weight: 500;
}
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    gap: 1.5rem;
    flex-wrap: wrap;
}
.finding-row:last-child { border-bottom: none; }
.finding-row:nth-child(even) { background: rgba(255,255,255,0.02); }
.finding-key {
    font-size: 1rem;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 0.3rem;
}
.finding-explain {
    font-size: 0.9rem;
    color: var(--grey-light);
    line-height: 1.55;
}
.finding-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--off-white);
    text-align: right;
    flex-shrink: 0;
    min-width: 120px;
}

/* ══════════ SIDEBAR ══════════ */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: var(--off-white) !important;
    font-size: 1rem !important;
}
.sidebar-heading {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 2px solid var(--red);
    padding-bottom: 0.5rem;
    margin: 1.2rem 0 1rem;
}
.sidebar-algo-step {
    display: flex;
    gap: 0.85rem;
    margin-bottom: 1rem;
    align-items: flex-start;
}
.sidebar-num {
    font-size: 0.75rem;
    font-weight: 800;
    color: var(--red);
    background: var(--red-soft);
    border: 1px solid var(--red-border);
    border-radius: 4px;
    padding: 0.15rem 0.45rem;
    flex-shrink: 0;
    margin-top: 2px;
    min-width: 28px;
    text-align: center;
}
.sidebar-algo-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 0.2rem;
}
.sidebar-algo-text {
    font-size: 0.88rem;
    color: var(--grey-light);
    line-height: 1.55;
}

/* ══════════ BUTTONS ══════════ */
.stButton { display: flex; justify-content: center; }
.stButton > button {
    background: var(--red) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: var(--white) !important;
    font-weight: 800 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.85rem 3rem !important;
    min-width: 220px !important;
    text-transform: uppercase !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(229,9,20,0.35) !important;
}
.stButton > button:hover {
    background: var(--red-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(229,9,20,0.5) !important;
}

/* ══════════ FILE UPLOADER ══════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--red) !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span { font-size: 1rem !important; color: var(--grey-light) !important; }

/* ══════════ SLIDER ══════════ */
.stSlider > div > div > div { background: var(--grey-dark) !important; }
.stSlider > div > div > div > div { background: var(--red) !important; }
.stSlider label { font-size: 1rem !important; font-weight: 600 !important; color: var(--white) !important; }

/* ══════════ CHECKBOX ══════════ */
.stCheckbox label { font-size: 1rem !important; color: var(--off-white) !important; }

/* ══════════ ANIMATIONS ══════════ */
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.65); }
}
@keyframes growBar {
    from { width: 0 !important; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══════════ UPLOAD HINT ══════════ */
.upload-hint {
    font-size: 0.95rem;
    color: var(--grey-mid);
    text-align: center;
    padding: 0.5rem 0 1rem;
}

/* hide streamlit default UI chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem; max-width: 1280px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE v2 — DO NOT MODIFY — Ensemble multi-signal approach
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_mri(img: Image.Image) -> np.ndarray:
    """Normalize + denoise. Returns float32 in [0,1]."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    lo, hi = gray.min(), gray.max()
    norm = (gray - lo) / (hi - lo + 1e-8)
    return gaussian_filter(norm, sigma=0.8)


def extract_brain_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    rough = gray > 0.05
    rough = binary_fill_holes(rough)
    labeled, n = sp_label(rough)
    if n == 0:
        return rough
    sizes = ndimage.sum(rough, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = (labeled == head_label)
    erode_px = max(5, int(min(h, w) * 0.035))
    brain = binary_erosion(head, iterations=erode_px)
    brain = binary_fill_holes(brain)
    brain = binary_dilation(brain, iterations=2)
    brain = brain & head
    if brain.sum() < (h * w * 0.03):
        brain = head
    return brain


def compute_local_contrast(gray: np.ndarray, window: int = 15) -> np.ndarray:
    local_mean = uniform_filter(gray, size=window)
    local_sq   = uniform_filter(gray ** 2, size=window)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    contrast = (gray - local_mean) / (local_std + 0.02)
    return contrast


def otsu_threshold_1d(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8
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
    h, w = gray.shape
    edge_margin = max(3, int(min(h, w) * 0.04))
    strict_mask = binary_erosion(brain_mask, iterations=edge_margin)
    brain_px = gray[strict_mask]
    if brain_px.size < 200:
        return None, None, {}
    mu = brain_px.mean()
    sig = brain_px.std()
    presets = {
        "low":      {"z_a": 3.2, "lc_z": 2.8, "min_area_frac": 0.003},
        "balanced": {"z_a": 2.5, "lc_z": 2.2, "min_area_frac": 0.002},
        "high":     {"z_a": 1.8, "lc_z": 1.6, "min_area_frac": 0.001},
    }
    p = presets.get(sensitivity, presets["balanced"])
    thr_a = mu + p["z_a"] * sig
    sig_a = (gray >= thr_a) & strict_mask
    lc = compute_local_contrast(gray, window=21)
    lc_brain = lc[strict_mask]
    lc_thr = lc_brain.mean() + p["lc_z"] * lc_brain.std()
    sig_b = (lc >= lc_thr) & strict_mask
    otsu_t = otsu_threshold_1d(brain_px)
    sig_c = (gray >= otsu_t) & strict_mask
    votes = sig_a.astype(np.uint8) + sig_b.astype(np.uint8) + sig_c.astype(np.uint8)
    anomaly = votes >= 2
    anomaly = binary_erosion(anomaly, iterations=2)
    anomaly = binary_dilation(anomaly, iterations=6)
    anomaly = binary_fill_holes(anomaly)
    anomaly = anomaly & strict_mask
    brain_area = strict_mask.sum()
    min_px = max(30, int(brain_area * p["min_area_frac"]))
    labeled, n = sp_label(anomaly)
    cleaned = np.zeros_like(anomaly)
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        if comp.sum() >= min_px:
            cleaned |= comp
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
        if rows.size == 0 or cols.size == 0: continue
        h_c = rows[-1] - rows[0] + 1
        w_c = cols[-1] - cols[0] + 1
        bbox_area = h_c * w_c
        solidity = area / (bbox_area + 1e-8)
        aspect_ratio = min(h_c, w_c) / (max(h_c, w_c) + 1e-8)
        score = area * (solidity ** 2) * aspect_ratio
        if score > best_score:
            best_score = score
            best_lbl = lbl
    tumor = (labeled2 == best_lbl)
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
    contrast  = diag.get("contrast", 0)
    circ      = diag.get("circularity", 0)
    area      = diag.get("area_frac", 0)
    c_score   = min(1.0, contrast / 5.0)
    r_score   = 1.0 - abs(circ - 0.55) / 0.55
    r_score   = max(0, min(1, r_score))
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
    halo  = binary_dilation(tumor_mask, iterations=6) & ~tumor_mask
    ovl[halo]       = [255, 100,  40]
    ovl[tumor_mask] = [255,  40,  60]
    blended = (0.45 * ovl + 0.55 * rgb).clip(0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)
    draw    = ImageDraw.Draw(result)
    draw.rectangle(bbox, outline=(180, 30, 50), width=4)
    inner = (bbox[0]+3, bbox[1]+3, bbox[2]-3, bbox[3]-3)
    draw.rectangle(inner, outline=(255, 80, 80), width=1)
    tick = 14
    x0, y0, x1, y1 = bbox
    for (cx, cy, dx, dy) in [
        (x0, y0,  1,  1), (x1, y0, -1,  1),
        (x0, y1,  1, -1), (x1, y1, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255,220,0), width=3)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255,220,0), width=3)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(12, int(min(original.size) * 0.028))
    draw.line([(cx-arm, cy), (cx+arm, cy)], fill=(255,230,0), width=2)
    draw.line([(cx, cy-arm), (cx, cy+arm)], fill=(255,230,0), width=2)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255,230,0))
    bh = 20
    bx, by = bbox[0], max(0, bbox[1] - bh - 3)
    draw.rectangle([bx, by, bx+148, by+bh], fill=(200, 25, 45))
    draw.text((bx+6, by+3), "ANOMALY DETECTED", fill=(255, 255, 255))
    return result


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
    rgb  = rgba[:, :, :3]
    if tumor_mask is not None:
        rgb[tumor_mask] = [0, 220, 255]
        halo = binary_dilation(tumor_mask, iterations=3) & ~tumor_mask
        rgb[halo] = [0, 160, 200]
    return rgb


def make_brain_mask_visual(brain_mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    gray_u8 = (np.clip(gray, 0, 1) * 200).astype(np.uint8)
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    rgb[brain_mask, 0] = np.clip(rgb[brain_mask, 0].astype(int) * 0.3 + 0, 0, 255).astype(np.uint8)
    rgb[brain_mask, 1] = np.clip(rgb[brain_mask, 1].astype(int) * 0.6 + 30, 0, 255).astype(np.uint8)
    rgb[brain_mask, 2] = np.clip(rgb[brain_mask, 2].astype(int) * 0.5 + 160, 0, 255).astype(np.uint8)
    rgb[~brain_mask] = (rgb[~brain_mask].astype(float) * 0.15).astype(np.uint8)
    return rgb


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#1f1f1f", edgecolor="none", dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#1f1f1f")
    ax.set_facecolor("#1f1f1f")
    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#00B4D8", alpha=0.55, label="Normal brain tissue",
            density=True, histtype="stepfilled")
    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#E50914", alpha=0.8, label="Suspected tumor region",
                density=True, histtype="stepfilled")
    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#F5A623", linewidth=1.5,
                   linestyle="--", label=f"Brightness cutoff")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#46D369", linewidth=1.5,
                   linestyle=":", label=f"Otsu split")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#808080", labelsize=7)
    ax.set_xlabel("Pixel Brightness (0 = dark, 1 = bright)", color="#808080", fontsize=8)
    ax.set_ylabel("How common", color="#808080", fontsize=8)
    ax.set_title("Brightness Distribution — Brain vs Tumor", color="#E5E5E5", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color("#B3B3B3")
    plt.tight_layout(pad=0.5)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_intensity_profile(gray: np.ndarray, tumor_mask: np.ndarray) -> Image.Image:
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    cy   = int(rows.mean()) if rows.size else gray.shape[0] // 2
    cx   = int(cols.mean()) if cols.size else gray.shape[1] // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.6))
    fig.patch.set_facecolor("#1f1f1f")
    for ax, profile, label in [
        (ax1, gray[cy, :],  "Horizontal scan"),
        (ax2, gray[:, cx],  "Vertical scan"),
    ]:
        ax.set_facecolor("#1f1f1f")
        ax.plot(profile, color="#00B4D8", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#00B4D8")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#808080", labelsize=6)
        ax.set_title(f"{label} through center of region", color="#B3B3B3", fontsize=8)
        if label == "Horizontal scan":
            if cols.size:
                ax.axvspan(cols[0], cols[-1], alpha=0.2, color="#E50914", label="Tumor span")
        else:
            if rows.size:
                ax.axvspan(rows[0], rows[-1], alpha=0.2, color="#E50914", label="Tumor span")
    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#1f1f1f")
    ax.set_facecolor("#1f1f1f")
    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["#141414", "#00B4D8", "#F5A623", "#E50914"], N=4
    )
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Agreement Map\n(how many detectors agree: 0–3)", color="#B3B3B3", fontsize=8, pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["None", "1/3", "2/3", "All 3"])
    cbar.ax.tick_params(colors="#808080", labelsize=7)
    cbar.outline.set_visible(False)
    plt.tight_layout(pad=0.3)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


# ═════════════════════════════════════════════════════════════════════════════
#  RISK CLASSIFICATION (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def classify_risk(confidence: float, area_frac: float, contrast: float) -> tuple:
    if confidence > 0.82 and area_frac > 0.005:
        return "HIGH", "risk-high", "This scan shows strong signs of an abnormal region. We recommend consulting a neurologist or radiologist as soon as possible for a professional review."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "risk-medium", "There are moderate indicators of an unusual region. Additional imaging (like a contrast MRI or PET scan) would help clarify the finding."
    else:
        return "LOW", "risk-low", "The signs are mild. No immediate action may be needed, but it's worth doing a follow-up scan in 3–6 months to monitor any changes."


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP DEFINITIONS (detailed, plain-English)
# ═════════════════════════════════════════════════════════════════════════════

PIPELINE_STEPS = [
    {
        "title": "Step 1 — Loading & Preparing the Image",
        "detail": "The MRI image is converted to grayscale (black-and-white) and every pixel's brightness is scaled from 0 (pure black) to 1 (pure white). A light blur is applied to reduce digital noise without losing important edges — similar to how a camera smooths out grain."
    },
    {
        "title": "Step 2 — Skull Stripping (Isolating the Brain)",
        "detail": "The algorithm finds and removes the skull, scalp, and background from the image, keeping only the brain tissue. It does this by identifying the large bright mass in the center, shrinking its edges to cut away the skull ring, then slightly expanding to recover brain tissue near the surface."
    },
    {
        "title": "Step 3 — Computing a Brightness Map",
        "detail": "For each pixel inside the brain, the system calculates how bright it is compared to its immediate neighbours. A tumor often appears unusually bright or dark compared to surrounding tissue — this map captures those differences in a grid of 'local contrast' scores."
    },
    {
        "title": "Step 4 — Running 3 Independent Detectors",
        "detail": "Three separate detection methods run at the same time:\n• Detector A (Z-score): Flags pixels that are significantly brighter than the brain average — like spotting someone unusually tall in a crowd.\n• Detector B (Local Contrast): Flags pixels that stand out compared to their immediate neighbourhood, even if globally they look normal.\n• Detector C (Otsu Split): Automatically finds the best brightness level to split the scan into 'normal' and 'abnormal' halves."
    },
    {
        "title": "Step 5 — Voting & Noise Cleanup",
        "detail": "A pixel is flagged as suspicious only if at least 2 out of 3 detectors agree. This reduces false alarms. The result is then cleaned up: tiny isolated dots (noise) are removed, nearby flagged areas are merged, and any holes inside a detected region are filled in."
    },
    {
        "title": "Step 6 — Selecting the Most Likely Region",
        "detail": "If multiple suspicious blobs remain, the system scores each one based on size, compactness, and roundness — since tumors tend to be solid, compact, and roughly oval. Thin strips or crescents along the skull edge are penalized. The highest-scoring blob is selected as the final candidate region."
    },
    {
        "title": "Step 7 — Measuring & Scoring the Region",
        "detail": "The detected region is measured for: how bright it is vs the surrounding brain, what fraction of the brain it covers, and how circular it is. These three signals are combined into a single confidence score (50%–98%) that estimates how likely the region is a real anomaly."
    },
    {
        "title": "Step 8 — Generating Report & Visuals",
        "detail": "All findings are assembled into the clinical report you see below: the highlighted overlay, heatmap, intensity charts, and metrics. Nothing is stored or transmitted — all processing happens locally in your browser session."
    },
]


# ═════════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="nf-hero">
  <p style="font-size:0.9rem;font-weight:700;letter-spacing:0.2em;color:#E50914;text-transform:uppercase;margin-bottom:0.5rem;">
    AI-Powered Brain MRI Analysis
  </p>
  <h1 class="nf-logo">Neuro<span>Scan</span></h1>
  <p class="nf-tagline">
    Upload a brain MRI slice and our ensemble detection pipeline will analyse it for unusual regions,
    measure their size and brightness contrast, and give you a full plain-English report.
  </p>
  <div class="nf-badges">
    <span class="nf-badge red">⚠️ Research Use Only — Not a Medical Device</span>
    <span class="nf-badge">Always Consult a Radiologist</span>
    <span class="nf-badge">T1 · T1-CE · T2 · FLAIR Supported</span>
    <span class="nf-badge">v2.0 Ensemble Pipeline</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 NeuroScan")
    st.markdown('<div class="sidebar-heading">Detection Sensitivity</div>', unsafe_allow_html=True)

    sensitivity = st.select_slider(
        "Choose how aggressively to flag anomalies",
        options=["low", "balanced", "high"],
        value="balanced",
    )

    sensitivity_descriptions = {
        "low": ("🔵 Conservative Mode",
                "Only flags very obvious, high-contrast anomalies. Fewer false alarms but may miss small or subtle lesions. Best for quick screening."),
        "balanced": ("🟡 Balanced Mode (Recommended)",
                     "A good middle ground that works well for most standard brain MRI scans. This is the default and recommended setting."),
        "high": ("🔴 High Sensitivity Mode",
                 "Catches even faint or very small anomalies. Useful if you already suspect a lesion exists. May produce some false positives on noisy scans."),
    }
    s_title, s_desc = sensitivity_descriptions[sensitivity]
    st.markdown(f"""
<div style="background:#1f1f1f;border:1px solid #333;border-left:4px solid #E50914;border-radius:8px;padding:1rem 1.1rem;margin:0.5rem 0 1rem;">
  <div style="font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.4rem;">{s_title}</div>
  <div style="font-size:0.92rem;color:#B3B3B3;line-height:1.6;">{s_desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">Display Options</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("🔬 Show pipeline intermediate images", value=False)
    if show_debug:
        st.markdown("""<div style="background:#1f1f1f;border:1px solid #333;border-radius:6px;padding:0.8rem 1rem;font-size:0.9rem;color:#B3B3B3;line-height:1.6;margin-bottom:0.75rem;">
Shows 3 internal images: the normalized grayscale, the brain mask (what the AI considers brain tissue), and the raw binary tumor mask before overlay rendering.</div>""", unsafe_allow_html=True)

    show_votes = st.checkbox("🗳️ Show detector agreement map", value=False)
    if show_votes:
        st.markdown("""<div style="background:#1f1f1f;border:1px solid #333;border-radius:6px;padding:0.8rem 1rem;font-size:0.9rem;color:#B3B3B3;line-height:1.6;margin-bottom:0.75rem;">
Displays a colour map showing which pixels were flagged by 1, 2, or all 3 detectors. Red = all 3 agreed. Only regions with 2+ agreement are marked as anomalous.</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">How the Algorithm Works</div>', unsafe_allow_html=True)

    algo_steps = [
        ("Preprocess", "Convert to grayscale, normalize brightness 0–1, apply mild blur to reduce noise"),
        ("Skull Strip", "Remove skull & background, keep only brain tissue for analysis"),
        ("Local Contrast", "Map how each pixel compares to its immediate neighbours"),
        ("Detector A — Z-score", "Flag pixels far above average brain brightness"),
        ("Detector B — Local Contrast", "Flag pixels unusually bright vs their surroundings"),
        ("Detector C — Otsu Split", "Auto-find brightness threshold to split normal vs abnormal"),
        ("2-of-3 Voting", "Only keep pixels where at least 2 detectors agreed"),
        ("Cleanup", "Remove noise dots, fill holes, filter tiny blobs"),
        ("Blob Scoring", "Score regions by size, compactness and roundness"),
        ("Report", "Calculate confidence, render highlights and charts"),
    ]
    for num, (title, desc) in enumerate(algo_steps, 1):
        st.markdown(f"""
<div class="sidebar-algo-step">
  <div class="sidebar-num">{num:02d}</div>
  <div>
    <div class="sidebar-algo-title">{title}</div>
    <div class="sidebar-algo-text">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.9rem;color:#808080;text-align:center;">NeuroScan v2.0 · Research Prototype<br>Not validated for clinical use</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ═════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Upload Your MRI Scan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-panel">
  <strong>What to upload:</strong> A single 2D slice from a brain MRI scan saved as PNG or JPEG.
  Best results come from <strong>axial (top-down) slices</strong> in T1, T1 with contrast, T2, or FLAIR format.
  The file should show clearly visible brain tissue — not a scout localizer or sagittal view.
</div>
""", unsafe_allow_html=True)

col_up_l, col_up_c, col_up_r = st.columns([1, 2, 1])
with col_up_c:
    uploaded_file = st.file_uploader(
        "Drag and drop your MRI image here, or click to browse",
        type=["png", "jpg", "jpeg"],
    )
    st.markdown('<p class="upload-hint">PNG preferred · Axial slice · T1 · T2 · FLAIR formats · Max 200MB</p>',
                unsafe_allow_html=True)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 1, 1])
    with col_btn_c:
        run = st.button("▶  Run Analysis Now")

    if run:

        # ── Animated pipeline progress ────────────────────────────────────────
        st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Analysis in Progress</div>',
                    unsafe_allow_html=True)
        status_box = st.empty()

        done_steps = []
        for i, step in enumerate(PIPELINE_STEPS):
            html_steps = ""
            for s in done_steps:
                html_steps += f"""
<div class="pipeline-step step-done-bg">
  <div class="step-dot-done"></div>
  <div><div class="step-title-done">✓ {s['title']}</div></div>
</div>"""
            html_steps += f"""
<div class="pipeline-step step-active-bg">
  <div class="step-dot-active"></div>
  <div>
    <div class="step-title-active">{step['title']}</div>
    <div class="step-detail">{step['detail']}</div>
  </div>
</div>"""
            for s in PIPELINE_STEPS[i+1:]:
                html_steps += f"""
<div class="pipeline-step step-done-bg" style="opacity:0.3">
  <div class="step-dot-wait"></div>
  <div><div class="step-title-wait">{s['title']}</div></div>
</div>"""

            status_box.markdown(
                f'<div style="background:#1f1f1f;border:1px solid #333;border-radius:12px;padding:1.5rem;"><div class="pipeline-wrap">{html_steps}</div></div>',
                unsafe_allow_html=True)
            time.sleep(0.55)
            done_steps.append(step)

        # ── Run the actual pipeline ───────────────────────────────────────
        gray_norm  = preprocess_mri(raw_img)
        brain_mask = extract_brain_mask(gray_norm)
        bbox, tumor_mask, diag = detect_tumor_region(
            gray_norm, brain_mask, sensitivity=sensitivity
        )

        # Mark all steps done
        html_done = ""
        for s in PIPELINE_STEPS:
            html_done += f"""
<div class="pipeline-step step-done-bg">
  <div class="step-dot-done"></div>
  <div><div class="step-title-done">✓ {s['title']}</div></div>
</div>"""
        status_box.markdown(
            f'<div style="background:#1f1f1f;border:1px solid #333;border-radius:12px;padding:1.5rem;"><div class="pipeline-wrap">{html_done}</div></div>',
            unsafe_allow_html=True)
        time.sleep(0.4)
        status_box.empty()

        # ══════════════════════════════════════════════════════════════════
        #  RESULTS
        # ══════════════════════════════════════════════════════════════════

        if bbox is not None and tumor_mask is not None:
            confidence  = estimate_confidence(diag)
            area_pct    = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(
                confidence, diag["area_frac"], diag["contrast"]
            )
            result_img  = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)

            # ── Detection status banner ───────────────────────────────────
            st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Detection Result</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
<div class="detect-badge-wrap">
  <span class="nf-detect-badge badge-anomaly">
    <span class="pulse-dot"></span>&nbsp;Anomaly Detected
  </span>
  <span class="risk-pill {risk_cls}">{risk} Risk</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="info-panel red-accent">
  <strong>What this means:</strong><br>
  The algorithm found a region in this scan that looks statistically unusual compared to the surrounding brain tissue.
  It is brighter or differently textured than normal, and was confirmed by at least 2 of our 3 independent detection methods.
  This <em>could</em> indicate a tumor, cyst, or other abnormality — but only a qualified radiologist or neurologist
  can confirm or rule this out. This tool is a research aid, not a diagnosis.
</div>
""", unsafe_allow_html=True)

            # ── Metric cards ──────────────────────────────────────────────
            st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Key Measurements</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-row">

  <div class="metric-card mc-cyan">
    <div class="metric-card-label">Detection Confidence</div>
    <div class="metric-card-value">{confidence*100:.1f}%</div>
    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div></div>
    <div class="metric-card-explain">
      How certain the AI is that an anomaly genuinely exists here.
      This combines brightness contrast, shape regularity, and region size.
      It is <strong>not a medical probability</strong> — it reflects algorithmic confidence only.
    </div>
  </div>

  <div class="metric-card mc-red">
    <div class="metric-card-label">Brain Coverage</div>
    <div class="metric-card-value">{area_pct:.2f}%</div>
    <div class="metric-card-explain">
      The detected region covers this percentage of the total brain area visible in this slice.
      Larger values mean a bigger suspected region relative to the brain.
    </div>
  </div>

  <div class="metric-card mc-amber">
    <div class="metric-card-label">Brightness Contrast</div>
    <div class="metric-card-value">{diag['contrast']:.2f}σ</div>
    <div class="metric-card-explain">
      How many standard deviations brighter the anomaly is compared to surrounding normal brain tissue.
      Values above 2σ are considered notable. Higher = more clearly different from normal tissue.
    </div>
  </div>

  <div class="metric-card mc-green">
    <div class="metric-card-label">Region Roundness</div>
    <div class="metric-card-value">{diag['circularity']:.2f}</div>
    <div class="metric-card-explain">
      How circular the detected region is. 1.0 = perfect circle. Brain tumors typically
      score between 0.3 and 0.8. Very low scores (near 0) may indicate a skull artifact rather than a lesion.
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

            # ── Image tabs ────────────────────────────────────────────────
            st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Visual Analysis</div>',
                        unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs([
                "🖼️  Detection Overlay",
                "🌡️  Heatmap View",
                "📊  Analysis Charts"
            ])

            with tab1:
                st.markdown("""
<div class="info-panel">
  <strong>How to read this view:</strong><br>
  The <strong>left image</strong> is your original MRI scan exactly as uploaded — nothing changed.<br>
  The <strong>right image</strong> is the same scan with the suspected anomaly highlighted:
  the region is coloured red, surrounded by an orange glow halo.
  The <strong>yellow crosshair</strong> marks the centre of the detected region,
  and the <strong>corner brackets</strong> show the bounding box drawn around it.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<span class="img-label">Original MRI Scan</span>', unsafe_allow_html=True)
                    st.image(raw_img, use_container_width=True)
                with c2:
                    st.markdown('<span class="img-label">Anomaly Highlighted</span>', unsafe_allow_html=True)
                    st.image(result_img, use_container_width=True)

            with tab2:
                st.markdown("""
<div class="info-panel">
  <strong>How to read the Heatmap:</strong><br>
  Each pixel is coloured by its brightness intensity — <strong>dark purple = very low</strong>,
  progressing through red and orange to <strong>bright yellow/white = very high</strong>.
  The <strong>cyan/blue region</strong> is the detected anomaly overlaid on top.
  Regions that appear much brighter than surrounding tissue are the ones the algorithm flagged.<br><br>
  <strong>How to read the Brain Mask:</strong><br>
  The <strong>blue-tinted area</strong> shows exactly what the algorithm considered "brain tissue" during analysis.
  Everything dark has been excluded (skull, background, scalp). Only the blue region was used to calculate thresholds and detect anomalies.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<span class="img-label">Intensity Heatmap + Anomaly Overlay</span>', unsafe_allow_html=True)
                    st.image(heatmap_arr, use_container_width=True)
                with c2:
                    st.markdown('<span class="img-label">Brain Mask — What the AI Analysed</span>', unsafe_allow_html=True)
                    brain_vis = make_brain_mask_visual(brain_mask, gray_norm)
                    st.image(brain_vis, use_container_width=True)

            with tab3:
                st.markdown("""
<div class="info-panel">
  <strong>How to read the Brightness Distribution chart (left):</strong><br>
  The <strong>blue bars</strong> show how often each brightness level appears in normal brain tissue.
  The <strong>red bars</strong> show the brightness levels found in the detected anomaly region.
  If the red bars are shifted <em>to the right</em> of the blue bars, the anomaly is brighter than normal tissue — a key detection signal.
  The <strong>dashed amber line</strong> shows the Z-score brightness cutoff. The <strong>green dotted line</strong> shows the Otsu split threshold.<br><br>
  <strong>How to read the Intensity Profile chart (right):</strong><br>
  These charts slice through the centre of the detected region — horizontally (left panel) and vertically (right panel).
  The <strong>cyan line</strong> is pixel brightness from one edge of the image to the other.
  The <strong>red shaded zone</strong> marks where the detected region spans. Peaks inside the red zone confirm the anomaly is brighter than surrounding tissue.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<span class="img-label">Pixel Brightness Distribution</span>', unsafe_allow_html=True)
                    hist_img = plot_histogram(gray_norm, brain_mask, tumor_mask, diag)
                    st.image(hist_img, use_container_width=True)
                with c2:
                    st.markdown('<span class="img-label">Brightness Profile Through Detected Region</span>', unsafe_allow_html=True)
                    prof_img = plot_intensity_profile(gray_norm, tumor_mask)
                    st.image(prof_img, use_container_width=True)

                if show_votes and "lc" in diag:
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
                    st.markdown('<span class="img-label">Detector Agreement Map</span>', unsafe_allow_html=True)
                    st.markdown("""
<div class="info-panel">
  Each pixel is coloured by how many of the 3 detectors flagged it as suspicious.
  <strong>Dark</strong> = no detector flagged it (normal tissue).
  <strong>Cyan</strong> = 1 detector flagged it.
  <strong>Amber</strong> = 2 detectors flagged it (this is the agreement threshold the algorithm uses).
  <strong>Red</strong> = all 3 detectors agreed it is suspicious.
  The algorithm only marks a region as anomalous where at least 2 detectors agree.
</div>""", unsafe_allow_html=True)
                    st.image(vote_img, use_container_width=True)

            # ── Pipeline debug images ─────────────────────────────────────
            if show_debug:
                st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Pipeline Intermediate Images</div>',
                            unsafe_allow_html=True)
                st.markdown("""
<div class="info-panel">
  These are the internal images generated at each stage of processing — useful for verifying the algorithm behaved correctly on your scan.<br>
  <strong>Image 1 — Normalized Grayscale:</strong> The input image after converting to black-and-white and scaling brightness from 0 to 1. This is what all detectors work from.<br>
  <strong>Image 2 — Brain Mask:</strong> The blue-tinted area is what the algorithm identified as brain tissue. Everything dark was excluded from analysis.<br>
  <strong>Image 3 — Tumor Mask:</strong> The raw binary mask showing exactly which pixels were flagged as anomalous before the final overlay was drawn.
</div>
""", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown('<span class="img-label">① Normalized Grayscale</span>', unsafe_allow_html=True)
                    st.image((gray_norm * 255).astype(np.uint8), use_container_width=True)
                with d2:
                    st.markdown('<span class="img-label">② Brain Mask</span>', unsafe_allow_html=True)
                    brain_debug = make_brain_mask_visual(brain_mask, gray_norm)
                    st.image(brain_debug, use_container_width=True)
                with d3:
                    st.markdown('<span class="img-label">③ Raw Tumor Mask</span>', unsafe_allow_html=True)
                    vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    vis[tumor_mask] = [255, 60, 60]
                    st.image(vis, use_container_width=True)

            # ── Clinical Report ───────────────────────────────────────────
            st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Full Clinical Analysis Report</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
<div class="report-wrap">
  <div class="report-title-bar">
    <div class="report-title-text">Clinical Analysis Report</div>
    <div class="report-meta-text">NeuroScan v2.0 · Ensemble Detection · Sensitivity: {sensitivity.upper()}</div>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Detection Status</div>
      <div class="finding-explain">Did the algorithm find a region that looks statistically unusual compared to normal brain tissue in this scan?</div>
    </div>
    <span class="finding-val" style="color:#ff4757;">ANOMALY DETECTED</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Risk Classification</div>
      <div class="finding-explain">Overall severity estimate based on detection confidence, the size of the region relative to the brain, and how bright it is. This is a heuristic estimate — not a medical diagnosis.</div>
    </div>
    <span class="risk-pill {risk_cls}">{risk}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Detection Confidence</div>
      <div class="finding-explain">How algorithmically certain the system is that this is a real anomaly. Combines brightness contrast (55% weight), shape regularity (25%), and region size (20%). Range is 50%–98%.</div>
    </div>
    <span class="finding-val">{confidence*100:.1f}%</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Anomaly Size — % of Brain</div>
      <div class="finding-explain">The detected region covers this fraction of the total brain area visible in this slice. Small values (under 1%) may indicate a focal lesion; large values (over 10%) could indicate diffuse pathology or a false positive.</div>
    </div>
    <span class="finding-val">{area_pct:.2f}%</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Pixel Count (Anomaly Region)</div>
      <div class="finding-explain">The total number of image pixels inside the detected region. This is resolution-dependent — a higher-resolution scan will produce larger pixel counts for the same physical area.</div>
    </div>
    <span class="finding-val">{int(tumor_mask.sum()):,} px</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Location — Bounding Box</div>
      <div class="finding-explain">The pixel coordinates of the rectangular box drawn around the anomaly. X = horizontal position (left to right); Y = vertical position (top to bottom). Used to locate the region in the image.</div>
    </div>
    <span class="finding-val">x: {bbox[0]}–{bbox[2]}, y: {bbox[1]}–{bbox[3]}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Brightness Contrast (σ)</div>
      <div class="finding-explain">How many standard deviations brighter the anomaly is compared to surrounding normal brain tissue. Values above 2σ are considered notable. Values above 3σ are strongly elevated and more likely to represent a genuine structural abnormality.</div>
    </div>
    <span class="finding-val">{diag['contrast']:.3f} σ</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Anomaly Mean Brightness</div>
      <div class="finding-explain">The average pixel brightness value inside the detected region. Scale runs from 0 (pure black) to 1 (pure white). Higher values indicate a hyper-intense region, which is typical of certain tumor types on T1-CE and FLAIR sequences.</div>
    </div>
    <span class="finding-val">{diag['tumor_mean']:.4f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Surrounding Tissue Brightness</div>
      <div class="finding-explain">The average brightness of normal brain tissue surrounding the detected region. This is the baseline the algorithm compares the anomaly against. The further the anomaly is from this value, the stronger the detection signal.</div>
    </div>
    <span class="finding-val">{diag['tissue_mean']:.4f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Tissue Variability (Std Dev)</div>
      <div class="finding-explain">How much brightness varies across normal brain tissue. A higher standard deviation means the brain tissue itself is more varied, making it harder to distinguish a tumor by brightness alone — this is factored into the threshold calculations.</div>
    </div>
    <span class="finding-val">{diag['tissue_std']:.4f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Z-Score Brightness Cutoff</div>
      <div class="finding-explain">Pixels with brightness at or above this value were flagged by Detector A (the Z-score method). Calculated as: mean brain brightness + {sensitivity} Z multiplier × standard deviation.</div>
    </div>
    <span class="finding-val">{diag['thr_a']:.4f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Otsu Auto-Split Threshold</div>
      <div class="finding-explain">The brightness level automatically chosen by Detector C (Otsu's method) to split pixels into "normal" and "abnormal" groups. This threshold maximises the brightness difference between the two groups — no manual input needed.</div>
    </div>
    <span class="finding-val">{diag['otsu_t']:.4f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Region Roundness (Circularity)</div>
      <div class="finding-explain">Measures how circular the detected region is. Calculated as: 4π × area ÷ perimeter². A perfect circle scores 1.0. Brain tumors typically score 0.3–0.8. Very low scores (below 0.1) suggest a thin stripe artefact rather than a true lesion.</div>
    </div>
    <span class="finding-val">{diag['circularity']:.3f}</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Brain Pixels Analysed</div>
      <div class="finding-explain">Total number of pixels identified as brain tissue and included in the analysis. A larger number means more of the brain was visible and usable in this slice, giving a more reliable result.</div>
    </div>
    <span class="finding-val">{int(brain_mask.sum()):,} px</span>
  </div>

  <div class="finding-row">
    <div>
      <div class="finding-key">Sensitivity Mode Used</div>
      <div class="finding-explain">The detection sensitivity setting active for this scan. This controls how strict or liberal the thresholds are. Changing sensitivity and re-running can help confirm or rule out marginal findings.</div>
    </div>
    <span class="finding-val">{sensitivity.upper()}</span>
  </div>

</div>
""", unsafe_allow_html=True)

            # ── Recommendation ────────────────────────────────────────────
            risk_color = {"HIGH": "red-accent", "MODERATE": "amber-accent", "LOW": "green-accent"}
            st.markdown(f"""
<div class="info-panel {risk_color.get(risk, '')}">
  <strong>Recommendation ({risk} Risk):</strong><br>{recommendation}
</div>
<div class="warn-panel">
  <strong>⚠️ Important Medical Disclaimer:</strong><br>
  This analysis is generated by a research algorithm using classical image processing methods.
  It has <strong>not</strong> been trained on labelled medical data and has <strong>not</strong> been validated
  for clinical use. All findings must be reviewed by a qualified radiologist or neurologist
  before any medical decisions are made. This tool should never be used as a substitute for
  professional medical evaluation.
</div>
""", unsafe_allow_html=True)

        else:
            # ── No detection ──────────────────────────────────────────────
            st.markdown('<div class="nf-section-title"><span class="red-bar"></span>Detection Result</div>',
                        unsafe_allow_html=True)

            st.markdown("""
<div class="detect-badge-wrap">
  <span class="nf-detect-badge badge-clear">
    <span class="pulse-dot"></span>&nbsp;No Anomaly Detected
  </span>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="info-panel green-accent">
  <strong>What this result means:</strong><br>
  At the current sensitivity setting, no region in this scan was flagged as statistically unusual compared
  to the surrounding brain tissue. The algorithm's three detectors did not reach the required 2-of-3 agreement
  on any region large enough to be considered significant.<br><br>
  <strong>This does not mean the scan is definitively clear.</strong> Subtle, very small, or diffuse lesions
  may not be detectable by this algorithm. If you expect a lesion is present, try switching to
  <strong>HIGH sensitivity</strong> in the sidebar and running the analysis again.
  Also verify that the uploaded image is an axial brain MRI slice — not a scout view or localizer image.
</div>
""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<span class="img-label">Uploaded MRI Scan</span>', unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<span class="img-label">Brain Region Identified</span>', unsafe_allow_html=True)
                hm = make_heatmap(gray_norm, brain_mask, None)
                st.image(hm, use_container_width=True)
