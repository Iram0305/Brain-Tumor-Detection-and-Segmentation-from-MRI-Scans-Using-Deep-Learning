# NeuroScan API — Fixed Tumor Detection Pipeline
#
# ═══════════════════════════════════════════════════════════════
#  WHAT WAS WRONG IN THE ORIGINAL CODE (and how it's fixed here)
# ═══════════════════════════════════════════════════════════════
#
#  BUG 1 ─ No skull stripping
#    Original scanned the *entire* image for bright pixels.
#    In T1 MRI the skull ring is always the brightest structure,
#    so the box always landed on the skull, never the tumor.
#    FIX: Extract a brain-only mask first; restrict all analysis
#         to pixels inside that mask.
#
#  BUG 2 ─ Naive global percentile threshold
#    np.percentile(gray, 98) on raw uint8 with no normalisation
#    is non-generalizable across scanners / MRI types.
#    FIX: Normalize to [0,1] first, then compute threshold as
#         brain_mean + z * brain_std (z-score approach).
#
#  BUG 3 ─ No statistical context
#    Threshold wasn't relative to surrounding tissue.
#    FIX: All statistics computed only over brain-masked pixels.
#
#  BUG 4 ─ O(N²) pixel-by-pixel DFS
#    Running DFS per pixel on the full image is both slow and wrong.
#    FIX: scipy.ndimage.label() does connected-component labeling
#         efficiently in one pass.
#
#  BUG 5 ─ Hardcoded 0.90 confidence
#    Meaningless placeholder.
#    FIX: Confidence derived from actual contrast ratio between
#         tumor region and surrounding brain tissue.
#
#  BUG 6 ─ Box only, no segmentation overlay
#    A bounding box alone doesn't confirm the highlighted region
#    matches real anatomy.
#    FIX: Semi-transparent colour overlay drawn on the tumor mask
#         in addition to the bounding box.
# ═══════════════════════════════════════════════════════════════

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label as sp_label,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan API", page_icon="🧠", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(120deg, #00E5FF 0%, #7B61FF 60%, #FF61DC 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.hero-sub { color: #718096; font-size: 0.95rem; margin-top: -0.5rem; }
.stButton > button {
    background: linear-gradient(135deg, #00E5FF, #7B61FF);
    border: none; border-radius: 8px;
    color: #000; font-weight: 700;
    width: 100%; padding: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.03em;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem; font-weight: 600;
    letter-spacing: .12em; text-transform: uppercase;
    color: #A0AEC0;
    border-bottom: 1px solid #2D3748;
    padding-bottom: .4rem; margin-bottom: 1rem;
}
.debug-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem; color: #718096;
    text-transform: uppercase; letter-spacing: .1em;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Preprocess: normalize + denoise
# ═════════════════════════════════════════════════════════════════════════════
def preprocess_mri(img: Image.Image) -> np.ndarray:
    """
    Convert to float32 grayscale normalized to [0, 1].
    Apply light Gaussian blur to suppress acquisition noise.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Min-max normalization so intensity ranges are comparable across scanners
    lo, hi = gray.min(), gray.max()
    normalized = (gray - lo) / (hi - lo + 1e-8)

    # Mild denoising; sigma=1.0 preserves edges while killing speckle noise
    denoised = gaussian_filter(normalized, sigma=1.0)
    return denoised


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Brain mask (rough skull stripping)
# ═════════════════════════════════════════════════════════════════════════════
def extract_brain_mask(gray_norm: np.ndarray) -> np.ndarray:
    """
    Separates the brain parenchyma from skull and background.

    Strategy:
      1. Threshold out near-black background.
      2. Fill holes → solid head shape.
      3. Keep only the largest connected region (the head).
      4. Erode to peel off skull (~3-5 % of image width works well).
      5. Fill holes again to get a solid brain-only mask.

    Returns a boolean (H, W) mask — True = brain tissue.
    """
    h, w = gray_norm.shape

    # 1. Background is very dark after normalization
    rough = gray_norm > 0.08

    # 2. Fill any internal holes (ventricles appear dark but are inside brain)
    filled = binary_fill_holes(rough)

    # 3. Keep largest connected component (avoids stray bright artefacts at edges)
    labeled, n = sp_label(filled)
    if n == 0:
        return filled  # fallback: return whatever we have
    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head_mask = labeled == head_label

    # 4. Erode to strip skull ring; iterations ≈ 4 % of the shorter image dimension
    erode_px = max(6, int(min(h, w) * 0.04))
    brain = binary_erosion(head_mask, iterations=erode_px)

    # 5. Re-fill ventricle holes that erosion exposed
    brain = binary_fill_holes(brain)

    # Safety: if erosion wiped everything out, back off to the head mask
    if brain.sum() < (h * w * 0.04):
        brain = head_mask

    return brain


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Anomaly / tumour detection
# ═════════════════════════════════════════════════════════════════════════════
def detect_tumor_region(
    gray_norm: np.ndarray,
    brain_mask: np.ndarray,
    z_threshold: float = 2.5,
) -> tuple:
    """
    Within the brain mask, flag voxels whose intensity is more than
    z_threshold standard deviations above the mean brain intensity.

    This works because:
      • T1-CE:   enhancing tumors are bright relative to white/grey matter.
      • T2/FLAIR: most tumors show hyperintensity relative to normal tissue.

    Pipeline:
      anomaly map → morphological cleanup → largest connected component
                  → bounding box (+ padding)

    Returns: (bbox, tumor_mask)  or  (None, None) if nothing found.
    """
    brain_px = gray_norm[brain_mask]
    if brain_px.size < 200:          # degenerate image guard
        return None, None

    mu  = brain_px.mean()
    sig = brain_px.std()

    # Z-score anomaly: pixels significantly brighter than average brain tissue
    thr = mu + z_threshold * sig
    anomaly = (gray_norm >= thr) & brain_mask

    # ── Morphological cleanup ──────────────────────────────────────────────
    # Erosion kills isolated noise speckles (small blobs that aren't real)
    anomaly = binary_erosion(anomaly,  iterations=2)
    # Dilation reconnects nearby fragments that belong to the same lesion
    anomaly = binary_dilation(anomaly, iterations=5)
    # Fill enclosed holes so the mask covers the full lesion core
    anomaly = binary_fill_holes(anomaly)
    # Ensure we stay strictly inside the brain after dilation
    anomaly = anomaly & brain_mask

    # ── Connected-component labeling → pick largest region ────────────────
    labeled, n = sp_label(anomaly)
    if n == 0:
        return None, None

    sizes = ndimage.sum(anomaly, labeled, range(1, n + 1))
    best  = int(np.argmax(sizes)) + 1
    tumor_mask = labeled == best

    # ── Bounding box with small padding ───────────────────────────────────
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, None

    pad = max(6, int(min(gray_norm.shape) * 0.015))
    H, W = gray_norm.shape
    bbox = (
        max(0,   cols[0]  - pad),   # x_min
        max(0,   rows[0]  - pad),   # y_min
        min(W-1, cols[-1] + pad),   # x_max
        min(H-1, rows[-1] + pad),   # y_max
    )

    return bbox, tumor_mask


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Visualization
# ═════════════════════════════════════════════════════════════════════════════
def draw_highlight(
    original_img: Image.Image,
    bbox: tuple,
    tumor_mask: np.ndarray,
) -> Image.Image:
    """
    Renders two layers on top of the original MRI:
      1. Semi-transparent red/orange segmentation overlay (shows exact shape).
      2. Crisp red bounding box + yellow crosshair for localization.
    """
    rgb = np.array(original_img.convert("RGB"), dtype=np.float32)

    # Layer 1 — coloured mask overlay (alpha blend)
    overlay = rgb.copy()
    overlay[tumor_mask] = [255, 55, 55]          # vivid red on detected pixels
    alpha = 0.42                                  # 42 % overlay, 58 % original
    blended = (alpha * overlay + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)

    result = Image.fromarray(blended)
    draw   = ImageDraw.Draw(result)

    # Layer 2 — bounding box
    draw.rectangle(bbox, outline=(255, 50, 50), width=3)

    # Layer 3 — crosshair at centroid
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(10, int(min(original_img.size) * 0.025))
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 230, 0), width=2)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 230, 0), width=2)

    # Layer 4 — label badge
    badge_h = 18
    bx0, by0 = bbox[0], max(0, bbox[1] - badge_h)
    draw.rectangle([bx0, by0, bx0 + 120, by0 + badge_h], fill=(255, 50, 50))
    draw.text((bx0 + 4, by0 + 2), "TUMOR REGION", fill=(255, 255, 255))

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Confidence estimation
# ═════════════════════════════════════════════════════════════════════════════
def estimate_confidence(
    gray_norm: np.ndarray,
    tumor_mask: np.ndarray,
    brain_mask: np.ndarray,
) -> float:
    """
    Measures how distinct the candidate region is from surrounding tissue.

    contrast_ratio = (tumor_mean - tissue_mean) / tissue_std

    Mapped to [0.50, 0.99] so UI always shows a sensible range.
    A high-contrast, compact region → high confidence.
    """
    tissue_px = gray_norm[brain_mask & ~tumor_mask]
    tumor_px  = gray_norm[tumor_mask]

    if tissue_px.size == 0 or tumor_px.size == 0:
        return 0.55

    contrast = (tumor_px.mean() - tissue_px.mean()) / (tissue_px.std() + 1e-8)
    # Sigmoid-like mapping: contrast of ~3σ → ~90 %; ~5σ → ~95 %
    confidence = min(0.99, max(0.50, 0.50 + contrast * 0.12))
    return float(confidence)


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Statistical anomaly detection · Morphological segmentation · MRI analysis</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    z_thresh = st.slider(
        "Sensitivity (Z-threshold)",
        min_value=1.5, max_value=4.5, value=2.5, step=0.1,
        help=(
            "Controls how many standard deviations above the mean brain "
            "intensity a pixel must be to be considered anomalous.\n\n"
            "Lower → more sensitive (flags subtler regions).\n"
            "Higher → stricter (only flags strong hyperintensity)."
        ),
    )

    show_debug = st.checkbox("🔬 Show intermediate steps", value=False,
                             help="Display brain mask and anomaly map alongside results.")

    st.divider()

    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Pipeline:**
1. **Normalize** — MRI intensities → [0, 1]
2. **Denoise** — Gaussian blur (σ=1.0)
3. **Brain mask** — threshold + fill holes + skull erosion
4. **Anomaly map** — pixels > mean + Z × std within brain
5. **Morphological cleanup** — erode → dilate → fill
6. **Largest component** — picks the most significant lesion
7. **Overlay** — coloured mask + bounding box on original
        """)

    st.divider()
    st.caption("VIBE6 INNOVATHON 2026 Submission")


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"],
    help="Supports T1, T1-CE, T2, and FLAIR axial slices.",
)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        run = st.button("🔍 Run Detection")

    if run:
        with st.spinner("Running detection pipeline…"):

            # ── Full pipeline ─────────────────────────────────────────────
            gray_norm  = preprocess_mri(raw_img)
            brain_mask = extract_brain_mask(gray_norm)
            bbox, tumor_mask = detect_tumor_region(
                gray_norm, brain_mask, z_threshold=z_thresh
            )

        # ── Results ───────────────────────────────────────────────────────
        if bbox is not None and tumor_mask is not None:
            result_img = draw_highlight(raw_img, bbox, tumor_mask)
            confidence = estimate_confidence(gray_norm, tumor_mask, brain_mask)
            area_pct   = 100.0 * tumor_mask.sum() / (brain_mask.sum() + 1e-8)

            st.success("✅ Tumor region detected and localized.")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence",        f"{confidence * 100:.1f} %")
            m2.metric("Label",             "Tumor Region")
            m3.metric("Tumor / Brain",     f"{area_pct:.1f} %")
            m4.metric("Z-threshold used",  f"{z_thresh:.1f} σ")

            # Main output
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-header">Original MRI</p>',
                            unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            with c2:
                st.markdown('<p class="section-header">Tumor Highlighted</p>',
                            unsafe_allow_html=True)
                st.image(result_img, use_container_width=True)

            # Debug / intermediate steps (optional)
            if show_debug:
                st.markdown("---")
                st.markdown("#### 🔬 Intermediate Pipeline Steps")

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.markdown('<p class="debug-label">① Normalised grayscale</p>',
                                unsafe_allow_html=True)
                    st.image((gray_norm * 255).astype(np.uint8),
                             use_container_width=True)

                with d2:
                    st.markdown('<p class="debug-label">② Brain mask (skull stripped)</p>',
                                unsafe_allow_html=True)
                    st.image((brain_mask.astype(np.uint8) * 255),
                             use_container_width=True)

                with d3:
                    st.markdown('<p class="debug-label">③ Tumour mask (anomaly region)</p>',
                                unsafe_allow_html=True)
                    tumor_vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                    tumor_vis[tumor_mask] = [255, 60, 60]
                    st.image(tumor_vis, use_container_width=True)

        else:
            st.warning(
                "⚠️ No significant tumor region detected at the current sensitivity. "
                "Try **lowering the Z-threshold** slider in the sidebar to increase sensitivity."
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-header">Uploaded MRI</p>',
                            unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
            if show_debug:
                with c2:
                    st.markdown('<p class="section-header">Brain mask</p>',
                                unsafe_allow_html=True)
                    st.image((brain_mask.astype(np.uint8) * 255),
                             use_container_width=True)
