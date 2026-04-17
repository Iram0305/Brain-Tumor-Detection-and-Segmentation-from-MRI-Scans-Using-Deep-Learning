# NeuroScan — Brain MRI Anomaly Detection
# UI v3.0: Plain-English Dashboard · Detailed Pipeline · Accessible Design

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --bg-base:      #05080F;
    --bg-card:      #090E1A;
    --bg-panel:     #0B1120;
    --bg-glass:     rgba(9,14,26,0.85);
    --border:       #16243A;
    --border-lit:   #1E3557;
    --border-glow:  rgba(0,200,255,0.25);
    --cyan:         #00C8FF;
    --cyan-dim:     #007BAA;
    --cyan-glow:    rgba(0,200,255,0.12);
    --green:        #00F5A0;
    --green-dim:    rgba(0,245,160,0.12);
    --red:          #FF3F6C;
    --red-dim:      rgba(255,63,108,0.12);
    --amber:        #FFAA00;
    --amber-dim:    rgba(255,170,0,0.12);
    --violet:       #A78BFA;
    --text-pri:     #ECF2FC;
    --text-sec:     #5E7A99;
    --text-dim:     #2D4460;
    --text-hint:    #8BA4C0;
    --radius-card:  14px;
    --radius-sm:    8px;
    --shadow-card:  0 4px 32px rgba(0,0,0,0.4);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-pri) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 2px; }

/* ═══════════════ HERO ═══════════════ */
.hero-wrap {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-grid-bg {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}
.hero-glow {
    position: absolute;
    top: -60px; left: -10%;
    width: 55%; height: 200%;
    background: radial-gradient(ellipse at 25% 40%, rgba(0,200,255,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-glow-r {
    position: absolute;
    top: 0; right: -5%;
    width: 40%; height: 100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(167,139,250,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.65rem;
    opacity: 0.85;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4.2rem);
    font-weight: 900;
    color: var(--text-pri);
    line-height: 1;
    letter-spacing: -0.04em;
    margin: 0;
}
.hero-title .accent {
    background: linear-gradient(100deg, var(--cyan) 0%, #60A5FA 45%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: clamp(0.8rem, 1.5vw, 0.92rem);
    color: var(--text-sec);
    margin-top: 0.8rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    line-height: 1.6;
    max-width: 600px;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.2rem;
}
.hero-tag {
    background: rgba(0,200,255,0.07);
    border: 1px solid var(--border-lit);
    border-radius: 100px;
    padding: 0.22rem 0.8rem;
    font-size: 0.68rem;
    color: var(--text-hint);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* ═══════════════ CARDS ═══════════════ */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-card);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.3s ease;
}
.card:hover { border-color: var(--border-lit); }

.card-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.7rem;
    margin-bottom: 1.1rem;
}

/* ═══════════════ METRIC TILES ═══════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.85rem;
    margin: 1.5rem 0;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-card);
    padding: 1.1rem 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
    cursor: help;
}
.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
}
.metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-tile.cyan::before  { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.metric-tile.cyan::after   { background: var(--cyan); }
.metric-tile.cyan          { box-shadow: 0 0 20px rgba(0,200,255,0.06); }

.metric-tile.green::before { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.metric-tile.green::after  { background: var(--green); }
.metric-tile.green         { box-shadow: 0 0 20px rgba(0,245,160,0.06); }

.metric-tile.amber::before { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
.metric-tile.amber::after  { background: var(--amber); }
.metric-tile.amber         { box-shadow: 0 0 20px rgba(255,170,0,0.06); }

.metric-tile.red::before   { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.metric-tile.red::after    { background: var(--red); }
.metric-tile.red           { box-shadow: 0 0 20px rgba(255,63,108,0.06); }

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.4rem, 3vw, 1.9rem);
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-val.cyan  { color: var(--cyan); }
.metric-val.green { color: var(--green); }
.metric-val.amber { color: var(--amber); }
.metric-val.red   { color: var(--red); }

.metric-label {
    font-size: 0.65rem;
    color: var(--text-sec);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}
.metric-hint {
    font-size: 0.6rem;
    color: var(--text-dim);
    margin-top: 0.35rem;
    font-style: italic;
    font-family: 'Outfit', sans-serif;
    line-height: 1.3;
    letter-spacing: 0;
}

/* ═══════════════ TOOLTIP ═══════════════ */
.has-tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 1px dashed var(--border-lit);
}
.has-tooltip .tooltip-box {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #0F1E33;
    border: 1px solid var(--border-lit);
    border-radius: var(--radius-sm);
    padding: 0.65rem 0.85rem;
    width: 220px;
    font-size: 0.72rem;
    color: var(--text-hint);
    line-height: 1.5;
    z-index: 9999;
    pointer-events: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    transition: opacity 0.2s ease, visibility 0.2s ease;
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    text-align: left;
    white-space: normal;
}
.has-tooltip .tooltip-box::after {
    content: '';
    position: absolute;
    top: 100%; left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: var(--border-lit);
}
.has-tooltip:hover .tooltip-box {
    visibility: visible;
    opacity: 1;
}

/* ═══════════════ BADGE ═══════════════ */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    font-weight: 700;
    animation: badgePop 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
.badge-detected {
    background: var(--red-dim);
    border: 1px solid rgba(255,63,108,0.45);
    color: var(--red);
}
.badge-clear {
    background: var(--green-dim);
    border: 1px solid rgba(0,245,160,0.4);
    color: var(--green);
}
.pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    animation: pulseDot 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

/* ═══════════════ CONFIDENCE BAR ═══════════════ */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 5px;
    margin-top: 0.55rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    animation: growBar 1.2s ease both;
}

/* ═══════════════ SECTION LABELS ═══════════════ */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-sec);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ═══════════════ UPLOAD ZONE ═══════════════ */
.upload-hint {
    font-size: 0.78rem;
    color: var(--text-sec);
    text-align: center;
    padding: 0.4rem 0;
    font-style: italic;
}

/* ═══════════════ SIDEBAR ═══════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: var(--text-pri) !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: var(--text-sec) !important;
}
.sidebar-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.2rem 0 0.8rem;
}
.sidebar-algo-step {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.85rem;
    align-items: flex-start;
}
.sidebar-algo-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--cyan);
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.sidebar-algo-text {
    font-size: 0.78rem;
    color: var(--text-hint);
    line-height: 1.55;
}
.sidebar-algo-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-pri);
    margin-bottom: 0.15rem;
}

/* ═══════════════ BUTTONS ═══════════════ */
.stButton { display: flex; justify-content: center; }
.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, #3B7EFF 100%);
    border: none;
    border-radius: var(--radius-sm);
    color: #030810 !important;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    font-size: 0.88rem;
    letter-spacing: 0.04em;
    padding: 0.75rem 2.5rem;
    width: auto;
    min-width: 200px;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 0 24px rgba(0,200,255,0.25);
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-2px);
    box-shadow: 0 4px 32px rgba(0,200,255,0.4);
}

/* ═══════════════ FILE UPLOADER ═══════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-lit) !important;
    border-radius: var(--radius-card) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan-dim) !important;
}

/* ═══════════════ SLIDER ═══════════════ */
.stSlider > div > div > div { background: var(--cyan-dim) !important; }
.stSlider > div > div > div > div { background: var(--cyan) !important; }

/* ═══════════════ TABS ═══════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 0;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: var(--text-sec) !important;
    border-radius: 0;
    padding: 0.75rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan);
    background: transparent;
}

/* ═══════════════ PIPELINE STEPS ═══════════════ */
.pipeline-wrap {
    display: flex;
    flex-direction: column;
    gap: 0;
}
.pipeline-step {
    display: flex;
    gap: 1rem;
    padding: 0.9rem 1rem;
    border-radius: var(--radius-sm);
    animation: fadeSlideUp 0.35s ease both;
    position: relative;
}
.pipeline-step:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 1.5rem;
    top: 100%;
    width: 2px;
    height: 8px;
    background: var(--border-lit);
}
.pipeline-step-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
    padding-top: 2px;
}
.step-icon-active {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--cyan);
    animation: pulseDot 1s ease infinite;
    box-shadow: 0 0 8px rgba(0,200,255,0.6);
}
.step-icon-done {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--green);
    flex-shrink: 0;
}
.step-icon-wait {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--border-lit);
    flex-shrink: 0;
}
.step-content-main {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-pri);
    line-height: 1.3;
}
.step-content-sub {
    font-size: 0.72rem;
    color: var(--text-sec);
    margin-top: 0.25rem;
    line-height: 1.5;
}
.step-content-sub.done { color: var(--green); opacity: 0.7; }
.step-active-bg { background: rgba(0,200,255,0.04); border: 1px solid rgba(0,200,255,0.1); }
.step-done-bg   { background: transparent; }

/* ═══════════════ REPORT ═══════════════ */
.report-header {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(1.1rem, 2.5vw, 1.45rem);
    font-weight: 700;
    color: var(--text-pri);
    margin-bottom: 0.2rem;
}
.report-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
    gap: 1rem;
    flex-wrap: wrap;
}
.finding-left { flex: 1; min-width: 0; }
.finding-key {
    color: var(--text-sec);
    font-size: 0.8rem;
    font-weight: 500;
}
.finding-explain {
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-top: 0.15rem;
    font-style: italic;
    line-height: 1.4;
}
.finding-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-pri);
    font-weight: 600;
    text-align: right;
    flex-shrink: 0;
}

/* ═══════════════ RISK CHIPS ═══════════════ */
.risk-chip {
    display: inline-block;
    padding: 0.18rem 0.75rem;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.risk-high   { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,63,108,0.4); }
.risk-medium { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,170,0,0.4); }
.risk-low    { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,245,160,0.4); }

/* ═══════════════ INFO / WARN BOXES ═══════════════ */
.info-box {
    background: var(--cyan-glow);
    border: 1px solid rgba(0,200,255,0.18);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--text-hint);
    margin: 0.75rem 0;
    line-height: 1.55;
}
.warn-box {
    background: var(--amber-dim);
    border: 1px solid rgba(255,170,0,0.25);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--amber);
    margin: 0.75rem 0;
    line-height: 1.55;
}
.plain-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--text-hint);
    margin: 0.75rem 0;
    line-height: 1.6;
}

/* ═══════════════ ANIMATIONS ═══════════════ */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    0%   { transform: scale(0.75); opacity: 0; }
    60%  { transform: scale(1.04); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.75); }
}
@keyframes growBar {
    from { width: 0 !important; }
}
@keyframes scanLine {
    0%   { top: 0; opacity: 0.6; }
    50%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* ═══════════════ SCAN WRAP ═══════════════ */
.scan-wrap {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius-sm);
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

/* ═══════════════ CENTERING ═══════════════ */
.center-wrap { display: flex; justify-content: center; }

/* hide streamlit default UI chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1200px; }
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
    """
    Render brain mask as a proper visualization:
    - Brain region shown as a translucent blue overlay on the grayscale MRI
    - Background is dark
    """
    # Build grayscale base (uint8)
    gray_u8 = (np.clip(gray, 0, 1) * 200).astype(np.uint8)
    # Create RGB image from grayscale
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    # Tint the brain region with a blue overlay
    rgb[brain_mask, 0] = np.clip(rgb[brain_mask, 0].astype(int) * 0.3 + 0, 0, 255).astype(np.uint8)
    rgb[brain_mask, 1] = np.clip(rgb[brain_mask, 1].astype(int) * 0.6 + 30, 0, 255).astype(np.uint8)
    rgb[brain_mask, 2] = np.clip(rgb[brain_mask, 2].astype(int) * 0.5 + 160, 0, 255).astype(np.uint8)
    # Darken everything outside brain
    rgb[~brain_mask] = (rgb[~brain_mask].astype(float) * 0.15).astype(np.uint8)
    return rgb


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#090E1A", edgecolor="none", dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#090E1A")
    ax.set_facecolor("#090E1A")
    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#00C8FF", alpha=0.55, label="Normal brain tissue",
            density=True, histtype="stepfilled")
    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#FF3F6C", alpha=0.8, label="Suspected tumor region",
                density=True, histtype="stepfilled")
    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#FFAA00", linewidth=1.5,
                   linestyle="--", label=f"Brightness cutoff")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#00F5A0", linewidth=1.5,
                   linestyle=":", label=f"Otsu split")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#5E7A99", labelsize=7)
    ax.set_xlabel("Pixel Brightness (0 = dark, 1 = bright)", color="#5E7A99", fontsize=8)
    ax.set_ylabel("How common", color="#5E7A99", fontsize=8)
    ax.set_title("Brightness Distribution — Brain vs Tumor", color="#ECF2FC", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color("#8BA4C0")
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
    fig.patch.set_facecolor("#090E1A")
    for ax, profile, label in [
        (ax1, gray[cy, :],  "Horizontal scan"),
        (ax2, gray[:, cx],  "Vertical scan"),
    ]:
        ax.set_facecolor("#090E1A")
        ax.plot(profile, color="#00C8FF", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#00C8FF")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#5E7A99", labelsize=6)
        ax.set_title(f"{label} through center of region", color="#8BA4C0", fontsize=8)
        if label == "Horizontal scan":
            if cols.size:
                ax.axvspan(cols[0], cols[-1], alpha=0.2, color="#FF3F6C", label="Tumor span")
        else:
            if rows.size:
                ax.axvspan(rows[0], rows[-1], alpha=0.2, color="#FF3F6C", label="Tumor span")
    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#090E1A")
    ax.set_facecolor("#090E1A")
    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["#0D1421", "#00C8FF", "#FFAA00", "#FF3F6C"], N=4
    )
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Agreement Map\n(how many detectors agree: 0–3)", color="#8BA4C0", fontsize=8, pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["None", "1/3", "2/3", "All 3"])
    cbar.ax.tick_params(colors="#5E7A99", labelsize=7)
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
<div class="hero-wrap">
  <div class="hero-grid-bg"></div>
  <div class="hero-glow"></div>
  <div class="hero-glow-r"></div>
  <p class="hero-eyebrow">v2.0 · Ensemble Detection · Statistical Segmentation</p>
  <h1 class="hero-title">Neuro<span class="accent">Scan</span></h1>
  <p class="hero-sub">
    AI-powered MRI anomaly detection using a multi-signal ensemble pipeline —
    designed to flag unusual regions in brain scans and present findings in plain, understandable language.
  </p>
  <div class="hero-tags">
    <span class="hero-tag">⚠️ Research Use Only</span>
    <span class="hero-tag">Not a Medical Device</span>
    <span class="hero-tag">Always Consult a Radiologist</span>
    <span class="hero-tag">T1 · T1-CE · T2 · FLAIR</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 NeuroScan Settings")
    st.markdown('<div class="sidebar-section-title">Detection Sensitivity</div>', unsafe_allow_html=True)

    sensitivity = st.select_slider(
        "Sensitivity level",
        options=["low", "balanced", "high"],
        value="balanced",
    )

    sensitivity_help = {
        "low": "🔵 **Conservative** — Only flags very obvious, high-contrast anomalies. Fewer false alarms, but may miss subtle lesions.",
        "balanced": "🟡 **Recommended** — A good middle ground. Works well for most standard brain MRI scans.",
        "high": "🔴 **Sensitive** — Catches even faint or small anomalies. Useful if you already suspect a lesion, but may produce some false alarms.",
    }
    st.info(sensitivity_help[sensitivity])

    st.markdown('<div class="sidebar-section-title">Display Options</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("🔬 Show pipeline intermediate images", value=False,
                              help="Shows the grayscale, brain mask, and tumor mask images produced at each stage.")
    show_votes = st.checkbox("🗳️ Show detector agreement map", value=False,
                              help="Displays which pixels were flagged by 1, 2, or all 3 detectors.")

    st.markdown('<div class="sidebar-section-title">How it Works — Algorithm Overview</div>', unsafe_allow_html=True)

    algo_steps = [
        ("Preprocess", "Convert to grayscale, normalize brightness 0–1, apply mild blur"),
        ("Skull Strip", "Remove skull & background, keep only brain tissue"),
        ("Local Contrast", "Map how each pixel compares to its neighbours"),
        ("Z-score (A)", "Flag pixels far above average brain brightness"),
        ("Local Contrast (B)", "Flag pixels unusually bright vs their surroundings"),
        ("Otsu Split (C)", "Auto-find threshold to split normal vs abnormal"),
        ("2-of-3 Vote", "Only keep pixels where ≥ 2 detectors agreed"),
        ("Cleanup", "Remove noise, fill holes, filter small blobs"),
        ("Blob Scoring", "Score remaining regions by size, compactness & roundness"),
        ("Report", "Calculate confidence, render highlights & charts"),
    ]
    for num, (title, desc) in enumerate(algo_steps, 1):
        st.markdown(f"""
<div class="sidebar-algo-step">
  <div class="sidebar-algo-num">{num:02d}</div>
  <div>
    <div class="sidebar-algo-title">{title}</div>
    <div class="sidebar-algo-text">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("NeuroScan v2.0 · Research prototype · Not for clinical use")


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

col_up_l, col_up_c, col_up_r = st.columns([1, 2, 1])
with col_up_c:
    uploaded_file = st.file_uploader(
        "Upload a Brain MRI Slice",
        type=["png", "jpg", "jpeg"],
        help="Upload a single 2D slice from a brain MRI scan. Best results with T1, T1-CE, T2, or FLAIR axial slices.",
    )
    st.markdown('<p class="upload-hint">PNG preferred · Axial slice · T1 · T2 · FLAIR formats supported</p>',
                unsafe_allow_html=True)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 1, 1])
    with col_btn_c:
        run = st.button("🔍  Run Analysis")

    if run:

        # ── Animated detailed pipeline steps ─────────────────────────────────
        status_box = st.empty()

        done_steps = []
        for i, step in enumerate(PIPELINE_STEPS):
            html_steps = ""
            for s in done_steps:
                html_steps += f"""
<div class="pipeline-step step-done-bg">
  <div class="pipeline-step-left"><div class="step-icon-done"></div></div>
  <div>
    <div class="step-content-main" style="color:var(--green);opacity:0.8">✓ {s['title']}</div>
  </div>
</div>"""
            html_steps += f"""
<div class="pipeline-step step-active-bg">
  <div class="pipeline-step-left"><div class="step-icon-active"></div></div>
  <div>
    <div class="step-content-main">{step['title']}</div>
    <div class="step-content-sub">{step['detail']}</div>
  </div>
</div>"""
            for s in PIPELINE_STEPS[i+1:]:
                html_steps += f"""
<div class="pipeline-step step-done-bg" style="opacity:0.3">
  <div class="pipeline-step-left"><div class="step-icon-wait"></div></div>
  <div><div class="step-content-main">{s['title']}</div></div>
</div>"""

            status_box.markdown(
                f'<div class="card"><div class="pipeline-wrap">{html_steps}</div></div>',
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
  <div class="pipeline-step-left"><div class="step-icon-done"></div></div>
  <div><div class="step-content-main" style="color:var(--green);opacity:0.85">✓ {s['title']}</div></div>
</div>"""
        status_box.markdown(
            f'<div class="card"><div class="pipeline-wrap">{html_done}</div></div>',
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

            # ── Result header ─────────────────────────────────────────────
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin:1.5rem 0 0.5rem;flex-wrap:wrap;">
  <span class="detection-badge badge-detected">
    <span class="pulse-dot"></span>ANOMALY DETECTED
  </span>
  <span class="risk-chip {risk_cls}">{risk} RISK</span>
</div>
<div class="plain-box">
  <strong>What does this mean?</strong><br>
  The algorithm found a region in the scan that looks statistically unusual — it's brighter or differently textured
  than the surrounding brain tissue, and it was confirmed by at least 2 of our 3 independent detection methods.
  This could indicate a tumor, cyst, or other abnormality — but <strong>only a qualified radiologist or neurologist
  can confirm this.</strong>
</div>
""", unsafe_allow_html=True)

            # ── Metric tiles ──────────────────────────────────────────────
            st.markdown(f"""
<div class="metric-grid">
  <div class="metric-tile cyan" title="How certain is the algorithm that this is a real anomaly? Higher = more sure.">
    <div class="metric-val cyan">{confidence*100:.1f}%</div>
    <div class="metric-label">Confidence</div>
    <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div></div>
    <div class="metric-hint">How sure the AI is that an anomaly exists — not a medical certainty</div>
  </div>
  <div class="metric-tile red" title="What percentage of the brain does the detected region take up?">
    <div class="metric-val red">{area_pct:.2f}%</div>
    <div class="metric-label">Brain Coverage</div>
    <div class="metric-hint">How much of the brain the flagged region occupies</div>
  </div>
  <div class="metric-tile amber" title="How much brighter is the anomaly vs. surrounding normal brain tissue, measured in standard deviations.">
    <div class="metric-val amber">{diag['contrast']:.2f}σ</div>
    <div class="metric-label">Brightness Contrast</div>
    <div class="metric-hint">Higher = more clearly different from normal tissue around it</div>
  </div>
  <div class="metric-tile green" title="How round/compact is the detected region? 1.0 = perfect circle. Tumors tend to be round.">
    <div class="metric-val green">{diag['circularity']:.2f}</div>
    <div class="metric-label">Roundness</div>
    <div class="metric-hint">How circular the region is (1.0 = perfect circle, 0 = irregular)</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Main image tabs ───────────────────────────────────────────
            tab1, tab2, tab3 = st.tabs([
                "🖼️  Detection Overlay",
                "🌡️  Heatmap View",
                "📊  Analysis Charts"
            ])

            with tab1:
                st.markdown("""
<div class="info-box">
  <strong>How to read this view:</strong> The left image is the original MRI scan as uploaded.
  The right image shows the same scan with the suspected anomaly highlighted in red, surrounded by an orange glow.
  The yellow crosshair and corner brackets pinpoint the centre and bounding box of the detected region.
</div>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Original MRI Scan</p>', unsafe_allow_html=True)
                    st.image(raw_img, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Anomaly Highlighted</p>', unsafe_allow_html=True)
                    st.image(result_img, use_container_width=True)

            with tab2:
                st.markdown("""
<div class="info-box">
  <strong>How to read this view:</strong> The heatmap colours each pixel by brightness — dark purple = low intensity, 
  yellow/white = high intensity. The bright <span style="color:#00C8FF">cyan/blue</span> region is the detected anomaly overlay.
  The right panel shows the <em>brain mask</em>: the blue-tinted area is what the algorithm considers "brain tissue";
  everything dark outside has been excluded from analysis.
</div>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Intensity Heatmap + Anomaly Overlay</p>',
                                unsafe_allow_html=True)
                    st.image(heatmap_arr, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Brain Mask — What the AI "sees"</p>',
                                unsafe_allow_html=True)
                    brain_vis = make_brain_mask_visual(brain_mask, gray_norm)
                    st.image(brain_vis, use_container_width=True)

            with tab3:
                st.markdown("""
<div class="info-box">
  <strong>How to read these charts:</strong><br>
  <strong>Left chart:</strong> Shows how many pixels fall at each brightness level.
  Blue bars = normal brain tissue. Red bars = the detected anomaly region.
  If the red bars are shifted to the right of blue bars, it means the anomaly is brighter than normal brain tissue.
  The dashed lines show where the detection thresholds were set.<br><br>
  <strong>Right chart:</strong> Shows a slice through the centre of the detected region.
  The cyan line is pixel brightness across the image from left to right (top chart) and top to bottom (bottom chart).
  The red shaded zone marks where the suspected region sits.
</div>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="section-label">Pixel Brightness Distribution</p>',
                                unsafe_allow_html=True)
                    hist_img = plot_histogram(gray_norm, brain_mask, tumor_mask, diag)
                    st.image(hist_img, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">Brightness Profile Through Detected Region</p>',
                                unsafe_allow_html=True)
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
                    st.markdown('<p class="section-label">Detector Agreement Map</p>',
                                unsafe_allow_html=True)
                    st.markdown("""
<div class="plain-box">
  Each pixel is coloured by how many of the 3 detectors flagged it. 
  <strong>Dark blue</strong> = no detector flagged it (normal). 
  <strong>Cyan</strong> = 1 detector. <strong>Amber</strong> = 2 detectors. 
  <strong>Red</strong> = all 3 agreed. The algorithm only marks a region as anomalous where at least 2 agree.
</div>""", unsafe_allow_html=True)
                    st.image(vote_img, use_container_width=True)

            # ── Pipeline debug images ─────────────────────────────────────
            if show_debug:
                st.markdown("---")
                st.markdown('<p class="section-label">Pipeline Intermediate Images</p>',
                            unsafe_allow_html=True)
                st.markdown("""
<div class="plain-box">
  These are the internal images generated during processing.
  <strong>①</strong> is the normalized grayscale (what the algorithm works from).
  <strong>②</strong> is the brain mask — blue = brain tissue kept for analysis, dark = excluded.
  <strong>③</strong> is the binary tumor mask — red pixels are the detected anomaly region.
</div>""", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.caption("① Normalized grayscale — brightness scaled 0–1")
                    st.image((gray_norm * 255).astype(np.uint8), use_container_width=True)
                with d2:
                    st.caption("② Brain mask — blue = brain region used for detection")
                    brain_debug = make_brain_mask_visual(brain_mask, gray_norm)
                    st.image(brain_debug, use_container_width=True)
                with d3:
                    st.caption("③ Tumor mask — red = detected anomaly pixels")
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
    <div class="finding-left">
      <div class="finding-key">Detection Status</div>
      <div class="finding-explain">Did the algorithm find an unusual region?</div>
    </div>
    <span class="finding-val" style="color:var(--red)">ANOMALY DETECTED</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Risk Classification</div>
      <div class="finding-explain">Overall severity estimate based on confidence, size, and brightness</div>
    </div>
    <span class="risk-chip {risk_cls}">{risk}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Detection Confidence</div>
      <div class="finding-explain">How certain is the AI? Combines brightness contrast, shape, and size signals.</div>
    </div>
    <span class="finding-val">{confidence*100:.1f}%</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Anomaly Size (% of brain)</div>
      <div class="finding-explain">The detected region covers this percentage of total brain area</div>
    </div>
    <span class="finding-val">{area_pct:.2f}%</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Pixel Count (anomaly region)</div>
      <div class="finding-explain">Total number of image pixels inside the detected region</div>
    </div>
    <span class="finding-val">{int(tumor_mask.sum()):,} px</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Location (bounding box)</div>
      <div class="finding-explain">X and Y pixel coordinates of the box drawn around the anomaly</div>
    </div>
    <span class="finding-val">x: {bbox[0]}–{bbox[2]}, y: {bbox[1]}–{bbox[3]}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Brightness Contrast (σ)</div>
      <div class="finding-explain">How many standard deviations brighter the anomaly is vs normal brain tissue. Above 2σ is notable.</div>
    </div>
    <span class="finding-val">{diag['contrast']:.3f} σ</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Anomaly Mean Brightness</div>
      <div class="finding-explain">Average pixel brightness inside the detected region (0 = black, 1 = white)</div>
    </div>
    <span class="finding-val">{diag['tumor_mean']:.4f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Surrounding Tissue Brightness</div>
      <div class="finding-explain">Average brightness of normal brain tissue around the anomaly</div>
    </div>
    <span class="finding-val">{diag['tissue_mean']:.4f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Tissue Variability (Std Dev)</div>
      <div class="finding-explain">How much the brightness varies across normal brain tissue — used to judge what counts as "too bright"</div>
    </div>
    <span class="finding-val">{diag['tissue_std']:.4f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Brightness Cutoff Used (Z-threshold)</div>
      <div class="finding-explain">Pixels brighter than this value were flagged by Detector A (Z-score method)</div>
    </div>
    <span class="finding-val">{diag['thr_a']:.4f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Auto-Split Threshold (Otsu)</div>
      <div class="finding-explain">Brightness level automatically chosen to separate "normal" from "abnormal" pixels by Detector C</div>
    </div>
    <span class="finding-val">{diag['otsu_t']:.4f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Region Roundness (Circularity)</div>
      <div class="finding-explain">1.0 = perfect circle. Tumors typically score 0.3–0.8. Very low scores may indicate artifacts.</div>
    </div>
    <span class="finding-val">{diag['circularity']:.3f}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Sensitivity Mode</div>
      <div class="finding-explain">The detection setting used for this scan — adjusts how strictly anomalies are flagged</div>
    </div>
    <span class="finding-val">{sensitivity.upper()}</span>
  </div>

  <div class="finding-row">
    <div class="finding-left">
      <div class="finding-key">Brain Pixels Analysed</div>
      <div class="finding-explain">Total number of pixels identified as brain tissue and included in the analysis</div>
    </div>
    <span class="finding-val">{int(brain_mask.sum()):,} px</span>
  </div>

  <div style="margin-top:1.2rem;" class="{'warn-box' if risk == 'HIGH' else 'info-box'}">
    <strong>Recommendation:</strong> {recommendation}
  </div>
  <div class="warn-box" style="margin-top:0.5rem;">
    ⚠️ <strong>Important:</strong> This analysis is generated by a research algorithm using classical image processing —
    it is <strong>not</strong> a trained medical AI and has <strong>not</strong> been validated for clinical use.
    All findings must be reviewed by a qualified radiologist or neurologist before any medical decisions are made.
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
<div class="plain-box">
  <strong>What does this mean?</strong><br>
  At the current sensitivity setting, no region in this scan was flagged as statistically unusual.
  This does <em>not</em> mean the scan is definitively clear — subtle or very small lesions may not be
  detectable at this sensitivity. If you expect a lesion, try switching to <strong>HIGH</strong> sensitivity in the sidebar,
  or verify that the uploaded image is an axial brain MRI slice (not a scout or localizer view).
</div>
""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-label">Uploaded MRI Scan</p>', unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<p class="section-label">Brain Region Identified</p>', unsafe_allow_html=True)
                hm = make_heatmap(gray_norm, brain_mask, None)
                st.image(hm, use_container_width=True)
