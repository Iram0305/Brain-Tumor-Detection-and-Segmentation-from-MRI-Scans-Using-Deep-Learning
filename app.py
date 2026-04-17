# NeuroScan — Brain MRI Anomaly Detection
# UI v5.0: Cohesive Medical-Green · Netflix Energy · One Unified Design System

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

st.set_page_config(
    page_title="NeuroScan · AI MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Medical Green · Unified · Netflix Energy
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800;900&family=Syne:wght@700;800&display=swap');

/* ── TOKENS ── */
:root {
    --g900: #0a1a0f;
    --g800: #0d2116;
    --g700: #112b1c;
    --g600: #163522;
    --g500: #1c4a2e;
    --g400: #1f5c36;
    --g300: #246b3e;
    --accent: #22c55e;
    --accent-dim: rgba(34,197,94,0.12);
    --accent-glow: rgba(34,197,94,0.25);
    --accent-border: rgba(34,197,94,0.35);
    --accent-hover: #16a34a;
    --surface: #0f1a14;
    --surface-2: #141f18;
    --surface-3: #192519;
    --surface-4: #1e2c1e;
    --border-subtle: rgba(34,197,94,0.10);
    --border-mid:    rgba(34,197,94,0.20);
    --border-strong: rgba(34,197,94,0.35);
    --text-primary:  #f0fdf4;
    --text-secondary:#86efac;
    --text-muted:    #4ade80;
    --text-dim:      rgba(134,239,172,0.5);
    --white:         #ffffff;
    --amber:         #f59e0b;
    --amber-soft:    rgba(245,158,11,0.12);
    --amber-border:  rgba(245,158,11,0.3);
    --red:           #ef4444;
    --red-soft:      rgba(239,68,68,0.12);
    --red-border:    rgba(239,68,68,0.3);
    --radius-sm:     6px;
    --radius:        10px;
    --radius-lg:     14px;
    --radius-xl:     20px;
    --shadow:        0 4px 24px rgba(0,0,0,0.5);
    --shadow-glow:   0 0 32px rgba(34,197,94,0.12);
}

/* ── BASE ── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--g400); border-radius: 3px; }

/* ════════════════════════════
   HERO — full cinematic band
   ════════════════════════════ */
.ns-hero {
    background:
        linear-gradient(180deg, var(--g800) 0%, var(--surface) 100%);
    border-bottom: 1px solid var(--border-mid);
    padding: 3.5rem 2rem 2.5rem;
    margin: -0.5rem -1rem 2.5rem;
    position: relative;
    overflow: hidden;
}
.ns-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(ellipse 70% 120% at 5% 60%, rgba(34,197,94,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 80% at 95% 20%, rgba(34,197,94,0.04) 0%, transparent 55%);
    pointer-events: none;
}
.ns-hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}
.ns-hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.ns-hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: var(--accent);
    border-radius: 1px;
}
.ns-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 800;
    color: var(--text-primary);
    line-height: 0.95;
    letter-spacing: -0.03em;
    margin: 0 0 1rem;
    position: relative;
    z-index: 1;
}
.ns-hero-title .hi {
    color: var(--accent);
    text-shadow: 0 0 40px rgba(34,197,94,0.4);
}
.ns-hero-desc {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.65;
    max-width: 580px;
    font-weight: 400;
    position: relative;
    z-index: 1;
    margin-bottom: 1.5rem;
}
.ns-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    position: relative;
    z-index: 1;
}
.ns-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid var(--border-mid);
    background: var(--accent-dim);
    color: var(--text-secondary);
    letter-spacing: 0.01em;
}
.ns-badge.warn {
    border-color: var(--amber-border);
    background: var(--amber-soft);
    color: var(--amber);
}

/* ════════════════════════════
   SECTION HEADER
   ════════════════════════════ */
.ns-section {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.1rem;
}
.ns-section-bar {
    width: 4px;
    height: 1.6rem;
    background: var(--accent);
    border-radius: 2px;
    flex-shrink: 0;
    box-shadow: 0 0 12px var(--accent-glow);
}
.ns-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0;
}

/* ════════════════════════════
   INFO CARDS — replaces every tooltip
   ════════════════════════════ */
.ns-info {
    background: var(--surface-3);
    border: 1px solid var(--border-mid);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.15rem 1.4rem;
    margin: 0.75rem 0 1.25rem;
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.75;
    animation: fadeUp 0.3s ease both;
}
.ns-info strong { color: var(--text-primary); font-weight: 700; }
.ns-info em { color: var(--accent); font-style: normal; font-weight: 600; }
.ns-info.amber {
    border-left-color: var(--amber);
    background: rgba(245,158,11,0.06);
    color: rgba(253,230,138,0.85);
}
.ns-info.amber strong { color: var(--amber); }
.ns-info.red {
    border-left-color: var(--red);
    background: var(--red-soft);
    color: rgba(252,165,165,0.9);
}
.ns-info.red strong { color: var(--red); }

/* ════════════════════════════
   METRIC CARDS
   ════════════════════════════ */
.ns-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0;
}
.ns-metric {
    background: var(--surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
    animation: fadeUp 0.4s ease both;
    cursor: default;
}
.ns-metric:hover {
    border-color: var(--border-strong);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
}
.ns-metric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.6;
}
.ns-metric.amber::before { background: var(--amber); }
.ns-metric.red::before   { background: var(--red); }
.ns-metric.white::before { background: var(--text-secondary); }
.ns-metric-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    margin-bottom: 0.6rem;
}
.ns-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    color: var(--accent);
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.ns-metric.amber .ns-metric-value { color: var(--amber); }
.ns-metric.red   .ns-metric-value { color: var(--red); }
.ns-metric.white .ns-metric-value { color: var(--text-primary); }
.ns-metric-desc {
    font-size: 0.9rem;
    color: var(--text-dim);
    line-height: 1.6;
}
.ns-conf-track {
    height: 5px;
    background: var(--surface-4);
    border-radius: 100px;
    margin: 0.7rem 0 0.4rem;
    overflow: hidden;
}
.ns-conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent-hover), var(--accent));
    animation: growBar 1.2s cubic-bezier(0.25,1,0.5,1) both;
}
@keyframes growBar { from { width: 0 !important; } }

/* ════════════════════════════
   DETECTION BADGES
   ════════════════════════════ */
.ns-result-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.5rem 0 1rem;
}
.ns-detect {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.65rem 1.6rem;
    border-radius: var(--radius);
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    animation: badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
.ns-detect.found {
    background: var(--red-soft);
    border: 2px solid var(--red-border);
    color: #fca5a5;
}
.ns-detect.clear {
    background: var(--accent-dim);
    border: 2px solid var(--accent-border);
    color: var(--accent);
}
.ns-risk {
    display: inline-block;
    padding: 0.45rem 1.1rem;
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    animation: badgePop 0.5s 0.1s cubic-bezier(0.34,1.56,0.64,1) both;
}
.ns-risk.high   { background: var(--red-soft);   border: 1px solid var(--red-border);   color: #fca5a5; }
.ns-risk.medium { background: var(--amber-soft); border: 1px solid var(--amber-border); color: var(--amber); }
.ns-risk.low    { background: var(--accent-dim); border: 1px solid var(--accent-border);color: var(--accent); }

/* pulse dot */
.pdot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: currentColor;
    animation: pdotAnim 1.5s ease infinite;
    flex-shrink: 0;
}
@keyframes pdotAnim {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.6); }
}

/* ════════════════════════════
   IMAGE LABELS
   ════════════════════════════ */
.ns-img-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ns-img-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}

/* ════════════════════════════
   TABS
   ════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface-2) !important;
    border-bottom: 1px solid var(--border-mid) !important;
    gap: 0 !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    padding: 0 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-dim) !important;
    padding: 0.85rem 1.5rem !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1.75rem !important;
}

/* ════════════════════════════
   PIPELINE STEPS
   ════════════════════════════ */
.pipeline-wrap { display: flex; flex-direction: column; }
.pipe-step {
    display: flex;
    gap: 1.1rem;
    padding: 0.9rem 1rem;
    border-radius: var(--radius-sm);
    position: relative;
    align-items: flex-start;
}
.pipe-step:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 1.45rem;
    top: 100%;
    width: 2px;
    height: 10px;
    background: var(--border-subtle);
}
.pdot-done   { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); flex-shrink: 0; margin-top: 4px; }
.pdot-active { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); animation: pdotAnim 0.9s ease infinite; box-shadow: 0 0 10px var(--accent-glow); flex-shrink: 0; margin-top: 4px; }
.pdot-wait   { width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--border-mid); flex-shrink: 0; margin-top: 4px; }
.pipe-active { background: var(--accent-dim); border: 1px solid var(--accent-border); }
.pipe-title-done   { font-size: 0.97rem; font-weight: 600; color: var(--accent); }
.pipe-title-active { font-size: 0.97rem; font-weight: 700; color: var(--text-primary); }
.pipe-title-wait   { font-size: 0.97rem; font-weight: 500; color: var(--text-dim); }
.pipe-detail { font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.3rem; line-height: 1.65; opacity: 0.85; }

/* ════════════════════════════
   CLINICAL REPORT
   ════════════════════════════ */
.ns-report {
    background: var(--surface-2);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-top: 1.5rem;
    box-shadow: var(--shadow);
}
.ns-report-header {
    background: linear-gradient(135deg, var(--g500), var(--g400));
    border-bottom: 1px solid var(--accent-border);
    padding: 1.15rem 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.ns-report-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: 0.01em;
    text-transform: uppercase;
}
.ns-report-meta {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-weight: 500;
    letter-spacing: 0.04em;
}
.ns-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 1rem 1.75rem;
    border-bottom: 1px solid var(--border-subtle);
    gap: 2rem;
    flex-wrap: wrap;
    transition: background 0.15s;
}
.ns-row:hover { background: rgba(34,197,94,0.03); }
.ns-row:last-child { border-bottom: none; }
.ns-row-key {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.ns-row-explain {
    font-size: 0.88rem;
    color: var(--text-dim);
    line-height: 1.6;
    max-width: 500px;
}
.ns-row-val {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    text-align: right;
    flex-shrink: 0;
    min-width: 130px;
    font-variant-numeric: tabular-nums;
}

/* ════════════════════════════
   SIDEBAR
   ════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--g900) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    font-size: 0.97rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.sb-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    padding: 1rem 0 0.25rem;
}
.sb-logo span { color: var(--accent); }
.sb-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--accent-border), transparent);
    margin: 0.5rem 0 1.25rem;
}
.sb-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1.5rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sb-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}
.sb-mode-card {
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    margin: 0.5rem 0 1rem;
}
.sb-mode-title {
    font-size: 1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.sb-mode-body {
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.sb-algo-step {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 0.95rem;
    align-items: flex-start;
}
.sb-num {
    font-size: 0.7rem;
    font-weight: 800;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    flex-shrink: 0;
    margin-top: 2px;
    min-width: 28px;
    text-align: center;
}
.sb-algo-title { font-size: 0.92rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.15rem; }
.sb-algo-body  { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.55; }
.sb-option-card {
    background: var(--surface-3);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    margin: 0.4rem 0 0.5rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ════════════════════════════
   BUTTONS
   ════════════════════════════ */
.stButton { display: flex; justify-content: center; }
.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #052e10 !important;
    font-weight: 800 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.9rem 3.2rem !important;
    min-width: 240px !important;
    text-transform: uppercase !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}
.stButton > button:hover {
    background: #16a34a !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 30px var(--accent-glow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ════════════════════════════
   FILE UPLOADER
   ════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--surface-3) !important;
    border: 2px dashed var(--border-mid) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-border) !important;
    background: var(--accent-dim) !important;
}
[data-testid="stFileUploader"] label { font-size: 1rem !important; font-weight: 600 !important; }

/* ════════════════════════════
   SLIDER
   ════════════════════════════ */
.stSlider > div > div > div { background: var(--border-mid) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSlider label { font-size: 1rem !important; font-weight: 600 !important; color: var(--text-primary) !important; }

/* ════════════════════════════
   CHECKBOX
   ════════════════════════════ */
.stCheckbox label span:last-child { font-size: 1rem !important; font-weight: 500 !important; }

/* ════════════════════════════
   UPLOAD HINT
   ════════════════════════════ */
.ns-upload-hint {
    font-size: 0.9rem;
    color: var(--text-dim);
    text-align: center;
    padding: 0.5rem 0 0.25rem;
}

/* ════════════════════════════
   ANIMATIONS
   ════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    0%   { transform: scale(0.8); opacity: 0; }
    65%  { transform: scale(1.04); }
    100% { transform: scale(1); opacity: 1; }
}

/* ════════════════════════════
   HIDE STREAMLIT CHROME
   ════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; max-width: 1300px; }
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
                facecolor="#141f18", edgecolor="none", dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#141f18")
    ax.set_facecolor("#141f18")
    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#22c55e", alpha=0.5, label="Normal brain tissue",
            density=True, histtype="stepfilled")
    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#ef4444", alpha=0.8, label="Suspected tumor region",
                density=True, histtype="stepfilled")
    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#f59e0b", linewidth=1.5,
                   linestyle="--", label="Brightness cutoff")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#86efac", linewidth=1.5,
                   linestyle=":", label="Otsu split")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#4ade80", labelsize=7)
    ax.set_xlabel("Pixel Brightness (0 = dark, 1 = bright)", color="#4ade80", fontsize=8)
    ax.set_ylabel("How common", color="#4ade80", fontsize=8)
    ax.set_title("Brightness Distribution — Brain vs Tumor", color="#f0fdf4", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color("#86efac")
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
    fig.patch.set_facecolor("#141f18")
    for ax, profile, label in [
        (ax1, gray[cy, :],  "Horizontal scan"),
        (ax2, gray[:, cx],  "Vertical scan"),
    ]:
        ax.set_facecolor("#141f18")
        ax.plot(profile, color="#22c55e", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#22c55e")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#4ade80", labelsize=6)
        ax.set_title(f"{label} through center of region", color="#86efac", fontsize=8)
        if label == "Horizontal scan":
            if cols.size:
                ax.axvspan(cols[0], cols[-1], alpha=0.2, color="#ef4444", label="Tumor span")
        else:
            if rows.size:
                ax.axvspan(rows[0], rows[-1], alpha=0.2, color="#ef4444", label="Tumor span")
    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#141f18")
    ax.set_facecolor("#141f18")
    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["#0a1a0f", "#22c55e", "#f59e0b", "#ef4444"], N=4
    )
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Agreement Map — Detectors 0–3", color="#86efac", fontsize=8, pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["None", "1 of 3", "2 of 3", "All 3"])
    cbar.ax.tick_params(colors="#4ade80", labelsize=7)
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
        return "HIGH", "high", "This scan shows strong signs of an abnormal region. We recommend consulting a neurologist or radiologist as soon as possible for a professional review."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "medium", "There are moderate indicators of an unusual region. Additional imaging (like a contrast MRI or PET scan) would help clarify the finding."
    else:
        return "LOW", "low", "The signs are mild. No immediate action may be needed, but it's worth doing a follow-up scan in 3–6 months to monitor any changes."


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP DEFINITIONS (detailed, plain-English)
# ═════════════════════════════════════════════════════════════════════════════

PIPELINE_STEPS = [
    {"title": "Step 1 — Loading & Preparing the Image",
     "detail": "The MRI image is converted to grayscale and every pixel's brightness is scaled from 0 (pure black) to 1 (pure white). A light blur reduces digital noise without losing important edges."},
    {"title": "Step 2 — Skull Stripping (Isolating the Brain)",
     "detail": "The algorithm removes the skull, scalp, and background — keeping only brain tissue. It finds the large bright central mass, shrinks edges to cut the skull ring, then expands slightly to recover surface tissue."},
    {"title": "Step 3 — Computing a Brightness Map",
     "detail": "For each brain pixel, the system calculates how bright it is compared to its immediate neighbours. Tumors often appear unusually bright compared to surrounding tissue — this map captures those differences."},
    {"title": "Step 4 — Running 3 Independent Detectors",
     "detail": "Detector A (Z-score) flags pixels significantly brighter than the brain average. Detector B (Local Contrast) flags pixels that stand out from their neighbourhood. Detector C (Otsu) auto-finds a brightness split between normal and abnormal."},
    {"title": "Step 5 — Voting & Noise Cleanup",
     "detail": "A pixel is only flagged if at least 2 of 3 detectors agree. Then: tiny isolated dots are removed, nearby areas are merged, and holes inside detected regions are filled."},
    {"title": "Step 6 — Selecting the Most Likely Region",
     "detail": "If multiple blobs remain, each is scored by size, compactness, and roundness — since tumors tend to be solid and roughly oval. Thin strips along the skull edge are penalised."},
    {"title": "Step 7 — Measuring & Scoring the Region",
     "detail": "The region is measured for brightness contrast vs surrounding brain, its fraction of total brain area, and how circular it is. These three signals combine into a confidence score (50–98%)."},
    {"title": "Step 8 — Generating Report & Visuals",
     "detail": "All findings are assembled into the clinical report: the highlighted overlay, heatmap, intensity charts, and metrics. Nothing is stored or transmitted — all processing is local."},
]


# ═════════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="ns-hero">
  <div class="ns-hero-eyebrow">AI-Powered Brain MRI Analysis · v2.0 Ensemble Pipeline</div>
  <h1 class="ns-hero-title">Neuro<span class="hi">Scan</span></h1>
  <p class="ns-hero-desc">
    Upload a brain MRI slice and our multi-signal ensemble pipeline will detect unusual regions,
    measure their size and brightness contrast, and deliver a full plain-English clinical report.
  </p>
  <div class="ns-badge-row">
    <span class="ns-badge warn">⚠ Research Use Only — Not a Medical Device</span>
    <span class="ns-badge">Always Consult a Radiologist</span>
    <span class="ns-badge">T1 · T1-CE · T2 · FLAIR</span>
    <span class="ns-badge">3-Detector Ensemble</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-logo">Neuro<span>Scan</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-rule"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">Detection Sensitivity</div>', unsafe_allow_html=True)

    sensitivity = st.select_slider(
        "Adjust how aggressively anomalies are flagged",
        options=["low", "balanced", "high"],
        value="balanced",
    )

    _modes = {
        "low":      ("🔵  Conservative", "Only flags very obvious, high-contrast anomalies. Fewer false alarms but may miss subtle or small lesions."),
        "balanced": ("🟡  Balanced — Recommended", "The default setting. Works well for most standard brain MRI scans and provides a reliable result."),
        "high":     ("🔴  High Sensitivity", "Catches faint or small anomalies. Useful when you already suspect a lesion is present. May produce some false positives."),
    }
    _mt, _mb = _modes[sensitivity]
    st.markdown(f'<div class="sb-mode-card"><div class="sb-mode-title">{_mt}</div><div class="sb-mode-body">{_mb}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">Display Options</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("🔬  Show pipeline intermediate images", value=False)
    if show_debug:
        st.markdown('<div class="sb-option-card">Shows 3 internal processing images: the normalized grayscale input, the brain mask showing exactly what tissue was analysed, and the raw binary tumor mask before the final overlay is drawn.</div>', unsafe_allow_html=True)

    show_votes = st.checkbox("🗳️  Show detector agreement map", value=False)
    if show_votes:
        st.markdown('<div class="sb-option-card">Shows a colour map of which pixels were flagged by 1, 2, or all 3 detectors. Green = 1 detector. Amber = 2 detectors (the detection threshold). Red = all 3 agreed.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">How the Algorithm Works</div>', unsafe_allow_html=True)

    _algo = [
        ("Preprocess",         "Convert to grayscale, normalize 0–1, apply mild noise-reduction blur"),
        ("Skull Strip",        "Remove skull and background; keep only brain tissue"),
        ("Local Contrast Map", "Calculate how each pixel compares to its immediate neighbours"),
        ("Detector A — Z-score", "Flag pixels far above average brain brightness"),
        ("Detector B — Local", "Flag pixels unusually bright vs their neighbourhood"),
        ("Detector C — Otsu",  "Auto-threshold to separate normal from abnormal pixels"),
        ("2-of-3 Voting",      "Only keep pixels where ≥ 2 detectors agreed"),
        ("Cleanup",            "Remove noise, fill holes, filter tiny blobs"),
        ("Blob Scoring",       "Score regions by size, compactness, and roundness"),
        ("Report",             "Calculate confidence, render highlights and full report"),
    ]
    for n, (t, d) in enumerate(_algo, 1):
        st.markdown(f"""<div class="sb-algo-step">
  <div class="sb-num">{n:02d}</div>
  <div><div class="sb-algo-title">{t}</div><div class="sb-algo-body">{d}</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.82rem;color:rgba(134,239,172,0.4);text-align:center;line-height:1.6;">NeuroScan v2.0 · Research Prototype<br>Not validated for clinical use</p>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Upload Your MRI Scan</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ns-info">
  <strong>What to upload:</strong> A single 2D slice from a brain MRI scan, saved as PNG or JPEG.
  Best results come from <em>axial (top-down) slices</em> in T1, T1-with-contrast, T2, or FLAIR format.
  The image should show clearly visible brain tissue — not a scout localizer or a sagittal/coronal view.
</div>
""", unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    uploaded_file = st.file_uploader(
        "Drag and drop your MRI image here, or click to browse",
        type=["png", "jpg", "jpeg"],
    )
    st.markdown('<p class="ns-upload-hint">PNG preferred · Axial slice · T1 · T2 · FLAIR formats supported</p>',
                unsafe_allow_html=True)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col_l2, col_c2, col_r2 = st.columns([1, 1, 1])
    with col_c2:
        run = st.button("▶  Run Analysis Now")

    if run:

        # ── Animated pipeline ──────────────────────────────────────────────
        st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Analysis in Progress</h2>
</div>
""", unsafe_allow_html=True)
        status_box = st.empty()

        done = []
        for i, step in enumerate(PIPELINE_STEPS):
            html = ""
            for s in done:
                html += f'<div class="pipe-step"><div class="pdot-done"></div><div><div class="pipe-title-done">✓ {s["title"]}</div></div></div>'
            html += f'<div class="pipe-step pipe-active"><div class="pdot-active"></div><div><div class="pipe-title-active">{step["title"]}</div><div class="pipe-detail">{step["detail"]}</div></div></div>'
            for s in PIPELINE_STEPS[i+1:]:
                html += f'<div class="pipe-step" style="opacity:0.35"><div class="pdot-wait"></div><div><div class="pipe-title-wait">{s["title"]}</div></div></div>'
            status_box.markdown(
                f'<div style="background:var(--surface-2);border:1px solid var(--border-mid);border-radius:var(--radius-lg);padding:1.5rem 1.75rem;"><div class="pipeline-wrap">{html}</div></div>',
                unsafe_allow_html=True)
            time.sleep(0.55)
            done.append(step)

        # ── Run detection ──────────────────────────────────────────────────
        gray_norm  = preprocess_mri(raw_img)
        brain_mask = extract_brain_mask(gray_norm)
        bbox, tumor_mask, diag = detect_tumor_region(gray_norm, brain_mask, sensitivity=sensitivity)

        html_done = "".join(
            f'<div class="pipe-step"><div class="pdot-done"></div><div><div class="pipe-title-done">✓ {s["title"]}</div></div></div>'
            for s in PIPELINE_STEPS
        )
        status_box.markdown(
            f'<div style="background:var(--surface-2);border:1px solid var(--border-mid);border-radius:var(--radius-lg);padding:1.5rem 1.75rem;"><div class="pipeline-wrap">{html_done}</div></div>',
            unsafe_allow_html=True)
        time.sleep(0.4)
        status_box.empty()

        # ══════════════════════════════════════════════════════════════════
        #  RESULTS — ANOMALY FOUND
        # ══════════════════════════════════════════════════════════════════

        if bbox is not None and tumor_mask is not None:
            confidence = estimate_confidence(diag)
            area_pct   = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(confidence, diag["area_frac"], diag["contrast"])
            result_img  = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)

            # ── Status banner ──────────────────────────────────────────────
            st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Detection Result</h2>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="ns-result-row">
  <span class="ns-detect found"><span class="pdot"></span>Anomaly Detected</span>
  <span class="ns-risk {risk_cls}">{risk} Risk</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="ns-info red">
  <strong>What this means:</strong>
  The algorithm found a region that looks statistically unusual compared to the surrounding brain tissue —
  it was brighter or differently textured, and confirmed by at least 2 of our 3 independent detectors.
  This <em>could</em> indicate a tumor, cyst, or other abnormality.
  <strong>Only a qualified radiologist or neurologist can confirm this finding.</strong>
  This tool is a research aid — not a clinical diagnosis.
</div>
""", unsafe_allow_html=True)

            # ── Metric cards ───────────────────────────────────────────────
            st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Key Measurements</h2>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="ns-metric-grid">

  <div class="ns-metric">
    <div class="ns-metric-label">Detection Confidence</div>
    <div class="ns-metric-value">{confidence*100:.1f}%</div>
    <div class="ns-conf-track"><div class="ns-conf-fill" style="width:{confidence*100:.1f}%"></div></div>
    <div class="ns-metric-desc">How algorithmically certain the system is that a real anomaly exists here.
    Combines brightness contrast (55%), shape regularity (25%), and size (20%). Not a medical probability.</div>
  </div>

  <div class="ns-metric red">
    <div class="ns-metric-label">Brain Coverage</div>
    <div class="ns-metric-value">{area_pct:.2f}%</div>
    <div class="ns-metric-desc">The detected region covers this percentage of the brain area visible in this slice.
    Small values (&lt;1%) suggest a focal lesion. Very large values (&gt;10%) may indicate a false positive.</div>
  </div>

  <div class="ns-metric amber">
    <div class="ns-metric-label">Brightness Contrast</div>
    <div class="ns-metric-value">{diag['contrast']:.2f}σ</div>
    <div class="ns-metric-desc">How many standard deviations brighter the anomaly is versus normal brain tissue.
    Values above 2σ are notable. Higher = more clearly different from surrounding tissue.</div>
  </div>

  <div class="ns-metric white">
    <div class="ns-metric-label">Region Roundness</div>
    <div class="ns-metric-value">{diag['circularity']:.2f}</div>
    <div class="ns-metric-desc">How circular the detected region is. 1.0 = perfect circle.
    Tumors typically score 0.3–0.8. Values near 0 suggest a skull artifact rather than a true lesion.</div>
  </div>

</div>
""", unsafe_allow_html=True)

            # ── Image tabs ─────────────────────────────────────────────────
            st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Visual Analysis</h2>
</div>
""", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["🖼  Detection Overlay", "🌡  Heatmap View", "📊  Analysis Charts"])

            with tab1:
                st.markdown("""
<div class="ns-info">
  <strong>Left — Original Scan:</strong> Your uploaded MRI image, unchanged.<br>
  <strong>Right — Anomaly Highlighted:</strong> Same scan with the detected region coloured red and surrounded by an orange glow.
  The <em>yellow crosshair</em> marks the region's centre. The <em>corner brackets</em> show the bounding box used to measure its position and size.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="ns-img-label">Original MRI Scan</div>', unsafe_allow_html=True)
                    st.image(raw_img, use_container_width=True)
                with c2:
                    st.markdown('<div class="ns-img-label">Anomaly Highlighted</div>', unsafe_allow_html=True)
                    st.image(result_img, use_container_width=True)

            with tab2:
                st.markdown("""
<div class="ns-info">
  <strong>Left — Intensity Heatmap:</strong> Each pixel is coloured by brightness —
  dark purple = low intensity, progressing through red/orange to bright yellow/white = very high.
  The <em>cyan region</em> is the detected anomaly overlaid on top.<br>
  <strong>Right — Brain Mask:</strong> The blue-tinted area is the exact tissue the algorithm analysed.
  Everything dark (skull, background, scalp) was excluded. Only the blue region was used to calculate thresholds and run detection.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="ns-img-label">Intensity Heatmap + Anomaly Overlay</div>', unsafe_allow_html=True)
                    st.image(heatmap_arr, use_container_width=True)
                with c2:
                    st.markdown('<div class="ns-img-label">Brain Mask — What the AI Analysed</div>', unsafe_allow_html=True)
                    st.image(make_brain_mask_visual(brain_mask, gray_norm), use_container_width=True)

            with tab3:
                st.markdown("""
<div class="ns-info">
  <strong>Left — Brightness Distribution:</strong> Green bars = normal brain tissue pixel counts. Red bars = the anomaly region's pixel counts.
  If red bars are shifted right of green, the anomaly is brighter than normal tissue — a core detection signal.
  The <em>dashed amber line</em> is the Z-score cutoff. The <em>dotted line</em> is the Otsu auto-threshold.<br>
  <strong>Right — Intensity Profile:</strong> Brightness sliced horizontally and vertically through the region centre.
  The <em>red shaded zone</em> marks where the detected region spans. Peaks inside that zone confirm the anomaly is brighter than surroundings.
</div>
""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="ns-img-label">Pixel Brightness Distribution</div>', unsafe_allow_html=True)
                    st.image(plot_histogram(gray_norm, brain_mask, tumor_mask, diag), use_container_width=True)
                with c2:
                    st.markdown('<div class="ns-img-label">Brightness Profile Through Region</div>', unsafe_allow_html=True)
                    st.image(plot_intensity_profile(gray_norm, tumor_mask), use_container_width=True)

                if show_votes and "lc" in diag:
                    brain_px  = gray_norm[brain_mask]
                    mu, sig_v = brain_px.mean(), brain_px.std()
                    _pre = {"low":(3.2,2.8),"balanced":(2.5,2.2),"high":(1.8,1.6)}
                    z_a, lc_z = _pre.get(sensitivity, (2.5,2.2))
                    lc_map    = diag["lc"]
                    lc_thr    = lc_map[brain_mask].mean() + lc_z * lc_map[brain_mask].std()
                    votes_disp = (
                        (gray_norm >= mu + z_a*sig_v).astype(np.uint8) +
                        (lc_map    >= lc_thr).astype(np.uint8) +
                        (gray_norm >= diag["otsu_t"]).astype(np.uint8)
                    )
                    st.markdown('<div class="ns-img-label">Detector Agreement Map</div>', unsafe_allow_html=True)
                    st.markdown("""
<div class="ns-info">
  Each pixel is coloured by how many of the 3 detectors flagged it.
  <strong>Dark</strong> = no detector (normal tissue). <strong>Green</strong> = 1 detector.
  <strong>Amber</strong> = 2 detectors — this is the agreement threshold the algorithm uses.
  <strong>Red</strong> = all 3 detectors agreed. Only regions reaching amber or red are marked as anomalous.
</div>""", unsafe_allow_html=True)
                    st.image(plot_signal_votes(votes_disp, brain_mask), use_container_width=True)

            # ── Pipeline debug images ──────────────────────────────────────
            if show_debug:
                st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Pipeline Intermediate Images</h2>
</div>
""", unsafe_allow_html=True)
                st.markdown("""
<div class="ns-info">
  <strong>Image 1 — Normalized Grayscale:</strong> The input after converting to black-and-white and scaling brightness 0–1. All three detectors work from this image.<br>
  <strong>Image 2 — Brain Mask:</strong> The blue-tinted region is what was identified as brain tissue and used for analysis. Everything else was excluded.<br>
  <strong>Image 3 — Raw Tumor Mask:</strong> The binary mask of flagged pixels before the final overlay rendering. Red pixels = the detected anomaly region.
</div>
""", unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown('<div class="ns-img-label">① Normalized Grayscale</div>', unsafe_allow_html=True)
                    st.image((gray_norm * 255).astype(np.uint8), use_container_width=True)
                with d2:
                    st.markdown('<div class="ns-img-label">② Brain Mask</div>', unsafe_allow_html=True)
                    st.image(make_brain_mask_visual(brain_mask, gray_norm), use_container_width=True)
                with d3:
                    st.markdown('<div class="ns-img-label">③ Raw Tumor Mask</div>', unsafe_allow_html=True)
                    vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    vis[tumor_mask] = [255, 60, 60]
                    st.image(vis, use_container_width=True)

            # ── Clinical Report ────────────────────────────────────────────
            st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Full Clinical Analysis Report</h2>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="ns-report">
  <div class="ns-report-header">
    <div class="ns-report-title">Clinical Analysis Report</div>
    <div class="ns-report-meta">NeuroScan v2.0 · Ensemble Detection · Sensitivity: {sensitivity.upper()}</div>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Detection Status</div>
      <div class="ns-row-explain">Did the algorithm find a region that looks statistically unusual in this scan?</div>
    </div>
    <span class="ns-row-val" style="color:#fca5a5;">ANOMALY DETECTED</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Risk Classification</div>
      <div class="ns-row-explain">Overall severity estimate based on detection confidence, region size relative to the brain, and brightness contrast. A heuristic — not a clinical diagnosis.</div>
    </div>
    <span class="ns-risk {risk_cls}">{risk}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Detection Confidence</div>
      <div class="ns-row-explain">Algorithmic certainty that this is a genuine anomaly. Combines contrast (55%), shape (25%), and size (20%). Range 50–98%. Not a medical probability score.</div>
    </div>
    <span class="ns-row-val">{confidence*100:.1f}%</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Anomaly Size — % of Brain</div>
      <div class="ns-row-explain">The detected region covers this fraction of the total brain area in this slice. Under 1% suggests a focal lesion. Over 10% may indicate diffuse pathology or a false positive.</div>
    </div>
    <span class="ns-row-val">{area_pct:.2f}%</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Pixel Count — Anomaly Region</div>
      <div class="ns-row-explain">Total pixels inside the detected region. Resolution-dependent — a higher-resolution scan produces larger counts for the same physical area.</div>
    </div>
    <span class="ns-row-val">{int(tumor_mask.sum()):,} px</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Location — Bounding Box</div>
      <div class="ns-row-explain">Pixel coordinates of the box drawn around the anomaly. X = horizontal (left to right). Y = vertical (top to bottom).</div>
    </div>
    <span class="ns-row-val">x: {bbox[0]}–{bbox[2]}, y: {bbox[1]}–{bbox[3]}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Brightness Contrast (σ)</div>
      <div class="ns-row-explain">Standard deviations above normal brain brightness. Above 2σ is notable. Above 3σ strongly suggests a genuine structural difference from normal tissue.</div>
    </div>
    <span class="ns-row-val">{diag['contrast']:.3f} σ</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Anomaly Mean Brightness</div>
      <div class="ns-row-explain">Average pixel brightness inside the detected region. 0 = pure black, 1 = pure white. Higher values indicate a hyper-intense region, typical of some tumor types on T1-CE and FLAIR.</div>
    </div>
    <span class="ns-row-val">{diag['tumor_mean']:.4f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Surrounding Tissue Brightness</div>
      <div class="ns-row-explain">Average brightness of normal brain tissue around the anomaly. This is the baseline the algorithm compares against. The further the anomaly deviates from this, the stronger the detection signal.</div>
    </div>
    <span class="ns-row-val">{diag['tissue_mean']:.4f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Tissue Variability (Std Dev)</div>
      <div class="ns-row-explain">How much brightness varies across normal brain tissue. High variability means the brain is more heterogeneous, making threshold-based detection harder.</div>
    </div>
    <span class="ns-row-val">{diag['tissue_std']:.4f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Z-Score Brightness Cutoff</div>
      <div class="ns-row-explain">Pixels at or above this brightness were flagged by Detector A. Calculated as: mean brain brightness + Z-multiplier × standard deviation.</div>
    </div>
    <span class="ns-row-val">{diag['thr_a']:.4f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Otsu Auto-Split Threshold</div>
      <div class="ns-row-explain">The brightness level automatically chosen by Detector C (Otsu's method) to separate normal from abnormal pixels by maximising inter-class brightness variance.</div>
    </div>
    <span class="ns-row-val">{diag['otsu_t']:.4f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Region Roundness (Circularity)</div>
      <div class="ns-row-explain">4π × area ÷ perimeter². Perfect circle = 1.0. Tumors typically score 0.3–0.8. Very low values (below 0.1) suggest a thin skull-edge artifact rather than a true lesion.</div>
    </div>
    <span class="ns-row-val">{diag['circularity']:.3f}</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Brain Pixels Analysed</div>
      <div class="ns-row-explain">Total pixels identified as brain tissue and included in the analysis. More pixels = more brain visible in this slice, giving more reliable threshold estimates.</div>
    </div>
    <span class="ns-row-val">{int(brain_mask.sum()):,} px</span>
  </div>

  <div class="ns-row">
    <div>
      <div class="ns-row-key">Sensitivity Mode Used</div>
      <div class="ns-row-explain">The detection sensitivity active for this scan. Changing this and re-running can help confirm marginal findings or reduce false positives.</div>
    </div>
    <span class="ns-row-val">{sensitivity.upper()}</span>
  </div>

</div>
""", unsafe_allow_html=True)

            # ── Recommendation + Disclaimer ────────────────────────────────
            _risk_info_cls = {"HIGH": "red", "MODERATE": "amber", "LOW": ""}
            st.markdown(f"""
<div class="ns-info {_risk_info_cls.get(risk, '')}">
  <strong>Recommendation — {risk} Risk:</strong><br>{recommendation}
</div>
<div class="ns-info amber">
  <strong>⚠ Medical Disclaimer:</strong>
  This analysis uses classical image processing — it is not a trained medical AI and has not been validated for clinical use.
  <strong>All findings must be reviewed by a qualified radiologist or neurologist before any medical decisions are made.</strong>
  This tool must never replace professional medical evaluation.
</div>
""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        #  RESULTS — NO DETECTION
        # ══════════════════════════════════════════════════════════════════
        else:
            st.markdown("""
<div class="ns-section">
  <div class="ns-section-bar"></div>
  <h2 class="ns-section-title">Detection Result</h2>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="ns-result-row">
  <span class="ns-detect clear"><span class="pdot"></span>No Anomaly Detected</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="ns-info">
  <strong>What this means:</strong>
  At the current sensitivity setting, no region was flagged as statistically unusual.
  The algorithm's 3 detectors did not reach the required 2-of-3 agreement on any region large enough to be significant.<br><br>
  <strong>This does not mean the scan is definitively clear.</strong>
  Subtle, very small, or diffuse lesions may not be detectable by this algorithm.
  If you expect a lesion is present, try switching to <em>High Sensitivity</em> in the sidebar and re-running.
  Also confirm the uploaded image is an axial brain MRI slice — not a scout, localizer, or sagittal view.
</div>
""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="ns-img-label">Uploaded MRI Scan</div>', unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<div class="ns-img-label">Brain Region Identified</div>', unsafe_allow_html=True)
                st.image(make_heatmap(gray_norm, brain_mask, None), use_container_width=True)
