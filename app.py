# NeuroScan — Brain MRI Anomaly Detection
# UI v4.0: High-Legibility Dashboard · Clickable Explanations · No Tooltips

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

# ── Custom CSS (OVERHAULED FOR MAXIMUM LEGIBILITY) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --bg-base:      #020408;
    --bg-card:      #0D1322;
    --bg-panel:     #0A0F1C;
    --border:       #1F3252;
    --border-lit:   #2A4775;
    --cyan:         #00C8FF;
    --cyan-dim:     #007BAA;
    --cyan-glow:    rgba(0,200,255,0.12);
    --green:        #00F5A0;
    --green-dim:    rgba(0,245,160,0.12);
    --red:          #FF4A7A;
    --red-dim:      rgba(255,74,122,0.15);
    --amber:        #FFB31A;
    --amber-dim:    rgba(255,179,26,0.15);
    --text-pri:     #FFFFFF;
    --text-sec:     #A0B8D6;
    --text-dim:     #6D8AAA;
    --radius-card:  16px;
    --radius-sm:    10px;
    --shadow-card:  0 8px 32px rgba(0,0,0,0.6);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-pri) !important;
}

/* ── Typography Enhancements ── */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(3rem, 7vw, 5rem);
    font-weight: 900;
    color: var(--text-pri);
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin: 0 0 1rem 0;
}
.hero-title .accent {
    background: linear-gradient(100deg, var(--cyan) 0%, #60A5FA 45%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: clamp(1.2rem, 2vw, 1.4rem);
    color: var(--text-pri);
    font-weight: 400;
    line-height: 1.6;
    max-width: 800px;
    margin-bottom: 2rem;
}

/* ═══════════════ CARDS ═══════════════ */
.card {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: var(--radius-card);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-card);
}

/* ═══════════════ METRIC TILES ═══════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}
.metric-tile {
    background: var(--bg-panel);
    border: 2px solid var(--border);
    border-radius: var(--radius-card);
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-tile.cyan  { border-bottom: 4px solid var(--cyan); }
.metric-tile.green { border-bottom: 4px solid var(--green); }
.metric-tile.amber { border-bottom: 4px solid var(--amber); }
.metric-tile.red   { border-bottom: 4px solid var(--red); }

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.metric-val.cyan  { color: var(--cyan); }
.metric-val.green { color: var(--green); }
.metric-val.amber { color: var(--amber); }
.metric-val.red   { color: var(--red); }

.metric-label {
    font-size: 1.1rem;
    color: var(--text-pri);
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ═══════════════ BADGE ═══════════════ */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.8rem 1.8rem;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-detected {
    background: var(--red-dim);
    border: 2px solid var(--red);
    color: #FFA3B8;
}
.badge-clear {
    background: var(--green-dim);
    border: 2px solid var(--green);
    color: var(--green);
}

/* ═══════════════ SIDEBAR & INPUTS ═══════════════ */
.sidebar-section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--cyan);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem;
}
.stSlider > div > div > div > div { background: var(--cyan) !important; height: 16px !important; }
.stSlider div[data-testid="stThumbValue"] { font-size: 1.2rem !important; }

/* ═══════════════ PIPELINE STEPS ═══════════════ */
.pipeline-step {
    display: flex;
    gap: 1.5rem;
    padding: 1.5rem;
    border: 2px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 1rem;
    background: var(--bg-panel);
}
.step-content-main {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--cyan);
    margin-bottom: 0.5rem;
}
.step-content-sub {
    font-size: 1.1rem;
    color: var(--text-pri);
    line-height: 1.6;
}

/* ═══════════════ REPORT ═══════════════ */
.report-header {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-pri);
    margin-bottom: 1.5rem;
    border-bottom: 2px solid var(--border);
    padding-bottom: 1rem;
}
.finding-row {
    display: flex;
    flex-direction: column;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
    gap: 0.5rem;
}
.finding-key {
    color: var(--cyan);
    font-size: 1.2rem;
    font-weight: 700;
}
.finding-explain {
    font-size: 1rem;
    color: var(--text-sec);
    line-height: 1.5;
}
.finding-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    color: var(--text-pri);
    font-weight: 700;
    background: rgba(255,255,255,0.05);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    width: fit-content;
}

/* ═══════════════ INFO / WARN BOXES ═══════════════ */
.info-box, .warn-box, .plain-box {
    border-radius: var(--radius-sm);
    padding: 1.5rem;
    font-size: 1.15rem;
    margin: 1rem 0;
    line-height: 1.6;
}
.info-box { background: var(--cyan-glow); border: 2px solid var(--cyan-dim); color: var(--text-pri); }
.warn-box { background: var(--amber-dim); border: 2px solid var(--amber); color: #FFF; }
.plain-box{ background: var(--bg-panel); border: 2px solid var(--border); color: var(--text-pri); }

/* hide streamlit default UI chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1400px; }

/* Expander Overrides for bigger text */
.streamlit-expanderHeader {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: var(--cyan) !important;
}
.streamlit-expanderContent {
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE v2 — COMPLETELY UNTOUCHED
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_mri(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.float32)
    lo, hi = gray.min(), gray.max()
    norm = (gray - lo) / (hi - lo + 1e-8)
    return gaussian_filter(norm, sigma=0.8)

def extract_brain_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    rough = gray > 0.05
    rough = binary_fill_holes(rough)
    labeled, n = sp_label(rough)
    if n == 0: return rough
    sizes = ndimage.sum(rough, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = (labeled == head_label)
    erode_px = max(5, int(min(h, w) * 0.035))
    brain = binary_erosion(head, iterations=erode_px)
    brain = binary_fill_holes(brain)
    brain = binary_dilation(brain, iterations=2)
    brain = brain & head
    if brain.sum() < (h * w * 0.03): brain = head
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

def detect_tumor_region(gray: np.ndarray, brain_mask: np.ndarray, sensitivity: str = "balanced") -> tuple:
    h, w = gray.shape
    edge_margin = max(3, int(min(h, w) * 0.04))
    strict_mask = binary_erosion(brain_mask, iterations=edge_margin)
    brain_px = gray[strict_mask]
    if brain_px.size < 200: return None, None, {}
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
        if comp.sum() >= min_px: cleaned |= comp
    labeled2, n2 = sp_label(cleaned)
    if n2 == 0: return None, None, {}
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
    if rows.size == 0 or cols.size == 0: return None, None, {}
    H, W = gray.shape
    pad = max(8, int(min(H, W) * 0.015))
    bbox = (max(0, cols[0] - pad), max(0, rows[0] - pad), min(W-1, cols[-1] + pad), min(H-1, rows[-1] + pad))
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
        "mu": float(mu), "sig": float(sig), "thr_a": float(thr_a), "otsu_t": float(otsu_t),
        "contrast": contrast, "area_frac": area_frac, "circularity": circ,
        "n_components_before_filter": n, "tumor_mean": float(tumor_px.mean()),
        "tissue_mean": float(tissue_px.mean()), "tissue_std": float(tissue_px.std()),
        "signal_votes_mean": float(votes[strict_mask].mean()), "lc": lc,
    }
    return bbox, tumor, diag

def estimate_confidence(diag: dict) -> float:
    contrast  = diag.get("contrast", 0)
    circ      = diag.get("circularity", 0)
    area      = diag.get("area_frac", 0)
    c_score   = min(1.0, contrast / 5.0)
    r_score   = 1.0 - abs(circ - 0.55) / 0.55
    r_score   = max(0, min(1, r_score))
    if 0.003 <= area <= 0.15: a_score = 1.0
    elif area < 0.003: a_score = area / 0.003
    else: a_score = max(0, 1 - (area - 0.15) / 0.15)
    raw = 0.55 * c_score + 0.25 * r_score + 0.20 * a_score
    return float(min(0.98, max(0.52, 0.50 + raw * 0.48)))

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
    for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255,220,0), width=3)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255,220,0), width=3)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(12, int(min(original.size) * 0.028))
    draw.line([(cx-arm, cy), (cx+arm, cy)], fill=(255,230,0), width=2)
    draw.line([(cx, cy-arm), (cx, cy+arm)], fill=(255,230,0), width=2)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255,230,0))
    bh = 30
    bx, by = bbox[0], max(0, bbox[1] - bh - 5)
    draw.rectangle([bx, by, bx+190, by+bh], fill=(200, 25, 45))
    return result

def make_heatmap(gray: np.ndarray, brain_mask: np.ndarray, tumor_mask: np.ndarray | None) -> np.ndarray:
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
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0D1322", edgecolor="none", dpi=150)
    buf.seek(0)
    return Image.open(buf).copy()

def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray, tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0D1322")
    ax.set_facecolor("#0D1322")
    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#00C8FF", alpha=0.55, label="Normal tissue", density=True, histtype="stepfilled")
    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#FF4A7A", alpha=0.8, label="Suspected tumor", density=True, histtype="stepfilled")
    if "thr_a" in diag: ax.axvline(diag["thr_a"], color="#FFB31A", linewidth=2, linestyle="--", label="Brightness cutoff")
    if "otsu_t" in diag: ax.axvline(diag["otsu_t"], color="#00F5A0", linewidth=2, linestyle=":", label="Otsu split")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#A0B8D6", labelsize=10)
    ax.set_xlabel("Pixel Brightness", color="#A0B8D6", fontsize=12)
    ax.set_title("Brightness Distribution", color="#FFFFFF", fontsize=14, pad=10)
    leg = ax.legend(fontsize=10, framealpha=0)
    for t in leg.get_texts(): t.set_color("#FFFFFF")
    plt.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img

def plot_intensity_profile(gray: np.ndarray, tumor_mask: np.ndarray) -> Image.Image:
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    cy   = int(rows.mean()) if rows.size else gray.shape[0] // 2
    cx   = int(cols.mean()) if cols.size else gray.shape[1] // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor("#0D1322")
    for ax, profile, label in [(ax1, gray[cy, :], "Horizontal scan"), (ax2, gray[:, cx], "Vertical scan")]:
        ax.set_facecolor("#0D1322")
        ax.plot(profile, color="#00C8FF", linewidth=2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#00C8FF")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#A0B8D6", labelsize=10)
        ax.set_title(label, color="#FFFFFF", fontsize=12)
        if label == "Horizontal scan" and cols.size: ax.axvspan(cols[0], cols[-1], alpha=0.3, color="#FF4A7A")
        if label == "Vertical scan" and rows.size: ax.axvspan(rows[0], rows[-1], alpha=0.3, color="#FF4A7A")
    plt.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img

def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0D1322")
    ax.set_facecolor("#0D1322")
    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list("vote", ["#0A0F1C", "#00C8FF", "#FFB31A", "#FF4A7A"], N=4)
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Agreement Map", color="#FFFFFF", fontsize=14, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["0/3", "1/3", "2/3", "3/3"])
    cbar.ax.tick_params(colors="#FFFFFF", labelsize=12)
    plt.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img

def classify_risk(confidence: float, area_frac: float, contrast: float) -> tuple:
    if confidence > 0.82 and area_frac > 0.005:
        return "HIGH", "risk-high", "This scan shows strong signs of an abnormal region. Consult a medical professional immediately."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "risk-medium", "There are moderate indicators of an unusual region. Additional imaging is recommended."
    else:
        return "LOW", "risk-low", "Signs are mild. No immediate action required, but consider a follow-up scan."

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP DEFINITIONS (Detailed, Plain English)
# ═════════════════════════════════════════════════════════════════════════════
PIPELINE_STEPS = [
    {"title": "Step 1 — Loading & Preparing the Image", "detail": "The MRI image is converted to grayscale (black-and-white) and every pixel's brightness is scaled from 0 (pure black) to 1 (pure white). A light blur is applied to reduce digital noise without losing important edges."},
    {"title": "Step 2 — Skull Stripping (Isolating the Brain)", "detail": "The algorithm finds and removes the skull, scalp, and background from the image, keeping only the brain tissue."},
    {"title": "Step 3 — Computing a Brightness Map", "detail": "For each pixel inside the brain, the system calculates how bright it is compared to its immediate neighbours."},
    {"title": "Step 4 — Running 3 Independent Detectors", "detail": "Three separate detection methods run at the same time looking for statistical brightness outliers and local contrast spikes."},
    {"title": "Step 5 — Voting & Noise Cleanup", "detail": "A pixel is flagged as suspicious only if at least 2 out of 3 detectors agree. This vastly reduces false alarms."},
    {"title": "Step 6 — Selecting the Most Likely Region", "detail": "The highest-scoring blob is selected based on size, compactness, and roundness."},
    {"title": "Step 7 — Measuring & Scoring the Region", "detail": "These signals are combined into a single confidence score (50%–98%) that estimates how likely the region is a real anomaly."},
]

# ═════════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
  <h1 class="hero-title">Neuro<span class="accent">Scan</span></h1>
  <p class="hero-sub">
    AI-powered MRI anomaly detection. Upload a scan, and our ensemble pipeline will
    analyze it for unusual regions and present the findings in clear, readable language.
  </p>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR (Tooltips removed, Explanations Explanded)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ NeuroScan Control Panel")
    
    st.markdown('<div class="sidebar-section-title">Detection Sensitivity</div>', unsafe_allow_html=True)
    sensitivity = st.select_slider(
        "Select how strict the algorithm should be:",
        options=["low", "balanced", "high"],
        value="balanced",
    )
    
    with st.expander("ℹ️ Click here to understand Sensitivity Levels"):
        st.markdown("""
        * **Low (Conservative):** Only flags very obvious, high-contrast anomalies. Fewer false alarms, but may miss subtle lesions.
        * **Balanced (Recommended):** A good middle ground. Works well for most standard brain MRI scans.
        * **High (Sensitive):** Catches even faint or small anomalies. Useful if you already suspect a lesion, but may produce some false alarms.
        """)

    st.markdown('<div class="sidebar-section-title">Display Options</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("🔬 Show intermediate images")
    with st.expander("ℹ️ What does 'intermediate images' mean?"):
        st.write("Checking this box will display the raw grayscale, the isolated brain mask, and the raw tumor mask generated at each step of the algorithm before final processing.")
        
    show_votes = st.checkbox("🗳️ Show detector agreement map")
    with st.expander("ℹ️ What does the 'agreement map' do?"):
        st.write("Checking this box adds a chart to the results showing exactly which pixels were flagged by 1, 2, or all 3 of our mathematical detectors.")


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD & RUN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 📤 Upload Patient Scan")
uploaded_file = st.file_uploader("Select a Brain MRI Slice (PNG, JPG)", type=["png", "jpg", "jpeg"])

with st.expander("ℹ️ Click here for Upload Guidelines"):
    st.markdown("""
    * **Format:** PNG is highly preferred to avoid compression artifacts. JPG is acceptable.
    * **View:** Must be an **Axial slice** (looking from the top down).
    * **Type:** Best results are achieved with T1, T1-CE, T2, or FLAIR formats.
    """)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🔍 Run Full Anomaly Detection Analysis", use_container_width=True):
        
        status_box = st.empty()
        
        # ── Animated Pipeline Output ───────────────────────────────────────
        with st.spinner("Processing Scan..."):
            gray_norm  = preprocess_mri(raw_img)
            brain_mask = extract_brain_mask(gray_norm)
            bbox, tumor_mask, diag = detect_tumor_region(gray_norm, brain_mask, sensitivity=sensitivity)
            time.sleep(1) # Brief pause for effect
            
        status_box.empty()

        # ══════════════════════════════════════════════════════════════════
        #  RESULTS
        # ══════════════════════════════════════════════════════════════════
        if bbox is not None and tumor_mask is not None:
            confidence  = estimate_confidence(diag)
            area_pct    = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(confidence, diag["area_frac"], diag["contrast"])
            result_img  = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)

            st.markdown(f"""
            <div style="margin:2rem 0;">
                <span class="detection-badge badge-detected">⚠️ ANOMALY DETECTED</span>
            </div>
            <div class="info-box">
                <strong>What does this mean?</strong><br><br>
                The algorithm found a region in the scan that looks statistically unusual. It is brighter or differently textured
                than the surrounding brain tissue, and it was confirmed by at least 2 of our 3 independent mathematical detection methods.
                <br><br><strong>Note: Only a qualified radiologist or neurologist can confirm what this region actually is.</strong>
            </div>
            """, unsafe_allow_html=True)

            # ── Metric tiles (No Tooltips) ──────────────────────────────────────────────
            st.markdown(f"""
            <div class="metric-grid">
              <div class="metric-tile cyan">
                <div class="metric-val cyan">{confidence*100:.1f}%</div>
                <div class="metric-label">AI Confidence</div>
              </div>
              <div class="metric-tile red">
                <div class="metric-val red">{area_pct:.2f}%</div>
                <div class="metric-label">Brain Coverage</div>
              </div>
              <div class="metric-tile amber">
                <div class="metric-val amber">{diag['contrast']:.2f}σ</div>
                <div class="metric-label">Brightness Contrast</div>
              </div>
              <div class="metric-tile green">
                <div class="metric-val green">{diag['circularity']:.2f}</div>
                <div class="metric-label">Roundness Score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Click here to read detailed explanations of these four metrics"):
                st.markdown("""
                * **AI Confidence:** How mathematically certain the algorithm is that this is a real anomaly. Higher is more sure.
                * **Brain Coverage:** What exact percentage of the total brain volume the detected region takes up.
                * **Brightness Contrast:** How much brighter the anomaly is compared to the normal brain tissue around it, measured in standard deviations (σ).
                * **Roundness Score:** How compact the detected region is. A perfect circle is 1.0. Tumors tend to be round and compact.
                """)

            # ── Main image tabs ───────────────────────────────────────────
            tab1, tab2, tab3 = st.tabs(["🖼️ Detection Overlay", "🌡️ Heatmap View", "📊 Analysis Charts"])

            with tab1:
                st.markdown("### Visual Overlay Comparison")
                st.write("The left image is the original scan. The right image shows the anomaly highlighted in red with bounding crosshairs.")
                c1, c2 = st.columns(2)
                with c1: st.image(raw_img, use_container_width=True, caption="Original Uploaded MRI")
                with c2: st.image(result_img, use_container_width=True, caption="Detected Anomaly Highlighted")

            with tab2:
                st.markdown("### Intensity Heatmap")
                st.write("The heatmap colors pixels by brightness. Dark purple is low intensity, white/yellow is high. The cyan/blue overlay represents the anomaly.")
                c1, c2 = st.columns(2)
                with c1: st.image(heatmap_arr, use_container_width=True, caption="Heatmap with Overlay")
                with c2: 
                    brain_vis = make_brain_mask_visual(brain_mask, gray_norm)
                    st.image(brain_vis, use_container_width=True, caption="Isolated Brain Mask")

            with tab3:
                st.markdown("### Statistical Charts")
                st.write("These charts explain why the AI flagged the region based on pixel brightness and structural profile.")
                c1, c2 = st.columns(2)
                with c1:
                    hist_img = plot_histogram(gray_norm, brain_mask, tumor_mask, diag)
                    st.image(hist_img, use_container_width=True)
                with c2:
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
                    votes_disp = ((gray_norm >= thr_a).astype(np.uint8) + (lc_map >= lc_thr).astype(np.uint8) + (gray_norm >= otsu_t).astype(np.uint8))
                    vote_img = plot_signal_votes(votes_disp, brain_mask)
                    st.markdown("### Detector Agreement Map")
                    st.write("Shows how many of the 3 detectors flagged each pixel. Red means all 3 agreed.")
                    st.image(vote_img, use_container_width=True)

            if show_debug:
                st.markdown("---")
                st.markdown("### Pipeline Intermediate Images")
                st.write("Raw data outputs from the computer vision engine.")
                d1, d2, d3 = st.columns(3)
                with d1: st.image((gray_norm * 255).astype(np.uint8), use_container_width=True, caption="1. Normalized Grayscale")
                with d2: st.image(make_brain_mask_visual(brain_mask, gray_norm), use_container_width=True, caption="2. Brain Mask")
                with d3: 
                    vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    vis[tumor_mask] = [255, 60, 60]
                    st.image(vis, use_container_width=True, caption="3. Binary Tumor Mask")

            # ── Detailed Report ───────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
            <div class="card">
              <div class="report-header">Detailed Clinical Report Data</div>

              <div class="finding-row">
                <div class="finding-key">Overall Risk Assessment</div>
                <div class="finding-explain">Estimated severity based on size and contrast.</div>
                <div class="finding-val">{risk} RISK</div>
              </div>

              <div class="finding-row">
                <div class="finding-key">Anomaly Size (Pixels)</div>
                <div class="finding-explain">Total number of image pixels inside the detected region.</div>
                <div class="finding-val">{int(tumor_mask.sum()):,} px</div>
              </div>

              <div class="finding-row">
                <div class="finding-key">Anomaly Mean Brightness</div>
                <div class="finding-explain">Average pixel brightness inside the detected region (0 = black, 1 = white).</div>
                <div class="finding-val">{diag['tumor_mean']:.4f}</div>
              </div>

              <div class="finding-row">
                <div class="finding-key">Surrounding Tissue Brightness</div>
                <div class="finding-explain">Average brightness of normal brain tissue around the anomaly.</div>
                <div class="finding-val">{diag['tissue_mean']:.4f}</div>
              </div>

              <div class="warn-box" style="margin-top:2rem;">
                <strong>Recommendation:</strong> {recommendation}<br><br>
                ⚠️ <strong>Important:</strong> This analysis is generated by a research algorithm using classical image processing. It is not a trained medical AI and has not been validated for clinical use.
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Click here to view the step-by-step logic the AI used to find this anomaly"):
                for step in PIPELINE_STEPS:
                    st.markdown(f"""
                    <div class="pipeline-step">
                      <div>
                        <div class="step-content-main">{step['title']}</div>
                        <div class="step-content-sub">{step['detail']}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        else:
            # ── No detection ──────────────────────────────────────────────
            st.markdown("""
            <div style="margin:2rem 0;">
              <span class="detection-badge badge-clear">✅ NO ANOMALY DETECTED</span>
            </div>
            <div class="plain-box">
              <strong>What does this mean?</strong><br><br>
              At the current sensitivity setting, no region in this scan was flagged as statistically unusual. 
              This does <em>not</em> mean the scan is definitively clear. If you expect a lesion, try switching to <strong>HIGH</strong> sensitivity in the sidebar.
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.image(raw_img, use_container_width=True, caption="Original Uploaded MRI")
            with c2: st.image(make_heatmap(gray_norm, brain_mask, None), use_container_width=True, caption="Brain Region Analyzed")
