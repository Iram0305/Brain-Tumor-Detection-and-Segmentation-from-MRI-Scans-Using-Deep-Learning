"""
NeuroScan API — Powered by Hugging Face InferenceClient
VIBE6 INNOVATHON 2026
"""
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st
from huggingface_hub import InferenceClient

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan · AI Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #C8D6E8;
}
.stApp { background-color: #080C14; }
section[data-testid="stSidebar"] { background-color: #0B1120; border-right: 1px solid #1A2540; }
section[data-testid="stSidebar"] * { color: #C8D6E8 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Hero Header ── */
.ns-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 2rem 0 0.5rem;
    border-bottom: 1px solid #1A2540;
    margin-bottom: 2rem;
}
.ns-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(110deg, #00D2FF 0%, #3A7BFF 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.ns-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3A7BFF;
    background: rgba(58, 123, 255, 0.1);
    border: 1px solid rgba(58, 123, 255, 0.3);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    align-self: center;
    margin-top: 0.3rem;
}
.ns-tagline {
    font-size: 0.85rem;
    color: #5A7095;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-top: 0.25rem;
    font-family: 'DM Mono', monospace;
}

/* ── Section Labels ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3A7BFF;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    display: block;
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1A2540 0%, transparent 100%);
}

/* ── Upload Zone ── */
.stFileUploader > div {
    background: #0D1526 !important;
    border: 1px dashed #1A2540 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}
.stFileUploader > div:hover { border-color: #3A7BFF !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #4A6080 !important; }

/* ── Scan Button ── */
.stButton > button {
    background: linear-gradient(110deg, #0061FF, #3A7BFF) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s ease, transform 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(0, 97, 255, 0.25) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #0D1526 !important;
    border: 1px solid #1A2540 !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #4A6080 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00D2FF !important;
}

/* ── Image Panels ── */
.img-panel {
    background: #0D1526;
    border: 1px solid #1A2540;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 0.5rem;
}
.img-caption {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3A7BFF;
    margin-bottom: 0.75rem;
}

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; }

/* ── Sidebar internals ── */
.sb-item {
    background: #0F1928;
    border: 1px solid #1A2540;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4A6080;
    word-break: break-all;
}
.sb-item span { color: #00D2FF; display: block; margin-bottom: 0.2rem; font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; }
.sb-rule { border: none; border-top: 1px solid #1A2540; margin: 1rem 0; }
.sb-footer { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #2A3C58; text-align: center; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Status dot ── */
.status-live { display:inline-flex; align-items:center; gap:0.4rem; }
.dot { width:6px; height:6px; border-radius:50%; background:#00FF87; box-shadow:0 0 8px #00FF87; display:inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3A7BFF !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_TOKEN  = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_API_TOKEN_HERE")
MODEL_ID  = "facebook/detr-resnet-50-panoptic"
# Fraction of total image pixels — segments covering more than this are background
BG_AREA_THRESHOLD = 0.40


# ── Core helpers ──────────────────────────────────────────────────────────────

def query_hf(file_path: str):
    """Call the HF InferenceClient and return the raw result list."""
    client = InferenceClient(token=HF_TOKEN)
    try:
        return client.image_segmentation(file_path, model=MODEL_ID)
    except Exception as exc:
        return {"error": str(exc)}


def pick_best_segment(results: list, img_w: int, img_h: int):
    """
    Return the best non-background segment.

    Strategy:
      1. Convert each mask to grayscale and compute its pixel-coverage ratio.
      2. Discard any segment whose mask covers ≥ BG_AREA_THRESHOLD of the image
         (those are almost always the scene background / surrounding tissue).
      3. Among the survivors, return the one with the highest confidence score.
         If all segments are large (whole-brain scans), fall back to the
         smallest-area high-confidence segment.
    """
    total_pixels = img_w * img_h
    candidates = []

    for seg in results:
        mask_pil = seg.get("mask")
        if mask_pil is None:
            continue

        # Resize mask to image dimensions and binarise
        mask_np = np.array(
            mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST).convert("L")
        )
        # Use Otsu-style mid-point threshold (128) — bright pixels = segment present
        binary = mask_np > 128
        coverage = binary.sum() / total_pixels

        if coverage < BG_AREA_THRESHOLD:
            candidates.append({
                "seg":      seg,
                "binary":   binary,
                "coverage": coverage,
                "score":    seg.get("score", 0.0),
                "label":    seg.get("label", "Region"),
            })

    if not candidates:
        # Fallback: pick the segment with smallest coverage (most localised)
        fallback = sorted(
            [
                {
                    "seg":      s,
                    "binary":   np.array(
                        s["mask"].resize((img_w, img_h), Image.Resampling.NEAREST).convert("L")
                    ) > 128,
                    "coverage": (
                        np.array(
                            s["mask"].resize((img_w, img_h), Image.Resampling.NEAREST).convert("L")
                        ) > 128
                    ).sum() / total_pixels,
                    "score":    s.get("score", 0.0),
                    "label":    s.get("label", "Region"),
                }
                for s in results if s.get("mask") is not None
            ],
            key=lambda x: x["coverage"],
        )
        return fallback[0] if fallback else None

    # Highest-confidence localised segment
    return max(candidates, key=lambda x: x["score"])


def draw_tight_bbox(original_img: Image.Image, binary_mask: np.ndarray):
    """
    Draw a tight bounding box around the True pixels in binary_mask.
    Returns the annotated image.
    """
    result_img = original_img.copy()

    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        return result_img  # nothing to draw

    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())

    draw = ImageDraw.Draw(result_img)

    # Outer glow  (semi-transparent wide rect)
    for offset, alpha in [(6, 30), (4, 60), (2, 120)]:
        overlay = Image.new("RGBA", result_img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rectangle(
            [min_x - offset, min_y - offset, max_x + offset, max_y + offset],
            outline=(0, 210, 255, alpha),
            width=1,
        )
        result_img = Image.alpha_composite(result_img.convert("RGBA"), overlay).convert("RGB")

    # Solid primary box
    draw = ImageDraw.Draw(result_img)
    draw.rectangle(
        [min_x, min_y, max_x, max_y],
        outline=(0, 210, 255),
        width=3,
    )

    # Corner accent marks
    corner_len = max(10, (max_x - min_x) // 8)
    tick_w = 5
    corners = [
        ((min_x, min_y), (min_x + corner_len, min_y), (min_x, min_y + corner_len)),
        ((max_x, min_y), (max_x - corner_len, min_y), (max_x, min_y + corner_len)),
        ((min_x, max_y), (min_x + corner_len, max_y), (min_x, max_y - corner_len)),
        ((max_x, max_y), (max_x - corner_len, max_y), (max_x, max_y - corner_len)),
    ]
    for pivot, h_end, v_end in corners:
        draw.line([pivot, h_end], fill=(0, 255, 135), width=tick_w)
        draw.line([pivot, v_end], fill=(0, 255, 135), width=tick_w)

    return result_img


def make_overlay(original_img: Image.Image, binary_mask: np.ndarray, alpha: int = 80):
    """Render a translucent cyan mask overlay on the original image."""
    base = original_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    mask_rgba[binary_mask] = [0, 210, 255, alpha]
    overlay = Image.fromarray(mask_rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
             background:linear-gradient(110deg,#00D2FF,#3A7BFF);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            NeuroScan
        </div>
        <div style='font-family:"DM Mono",monospace;font-size:0.6rem;color:#2A3C58;
             letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;'>
            v2.0 · Diagnostic Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Endpoint</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sb-item'>
        <span>Model ID</span>{MODEL_ID}
    </div>
    <div class='sb-item'>
        <span>Runtime</span>huggingface_hub · InferenceClient
    </div>
    <div class='sb-item'>
        <span>Task</span>Image Segmentation → Anomaly Localisation
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    for step in ["Upload MRI slice", "Panoptic segmentation", "Background filter", "Tight bounding box"]:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem;
             font-family:"DM Mono",monospace;font-size:.68rem;color:#4A6080;'>
            <div style='width:5px;height:5px;border-radius:50%;background:#3A7BFF;flex-shrink:0'></div>
            {step}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sb-footer">VIBE6 INNOVATHON 2026</div>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
    <div>
        <div class="ns-logo">NeuroScan <span style="font-size:1.4rem">API</span></div>
        <div class="ns-tagline">// AI-Powered Anomaly Localisation · Hugging Face InferenceClient</div>
    </div>
    <div style="margin-left:auto">
        <div class="status-live">
            <span class="dot"></span>
            <span style="font-family:'DM Mono',monospace;font-size:.65rem;color:#00FF87;letter-spacing:.1em;">LIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Upload zone ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Input · MRI Slice</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop an MRI image here or click to browse",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if not uploaded_file:
    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#2A3C58;
         font-family:"DM Mono",monospace;font-size:.75rem;letter-spacing:.08em;'>
        ↑ Upload a .png / .jpg MRI slice to begin analysis
    </div>""", unsafe_allow_html=True)

# ── Main analysis ─────────────────────────────────────────────────────────────
if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = raw_img.size

    # Compact preview + trigger
    prev_col, btn_col = st.columns([3, 1], gap="large")
    with prev_col:
        st.markdown('<div class="img-panel"><div class="img-caption">Preview</div>', unsafe_allow_html=True)
        st.image(raw_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col:
        st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-family:"DM Mono",monospace;font-size:.65rem;color:#4A6080;margin-bottom:.5rem;'>
            Resolution<br>
            <span style='color:#C8D6E8;font-size:.85rem;font-family:"Syne",sans-serif;font-weight:700;'>
                {img_w} × {img_h}
            </span>
        </div>""", unsafe_allow_html=True)
        run = st.button("🔬  Run NeuroScan")

    if run:
        if HF_TOKEN == "YOUR_HUGGINGFACE_API_TOKEN_HERE":
            st.error("⚠️ **No API Token** — Set `HF_TOKEN` in Streamlit Secrets or your environment.")
            st.stop()

        with st.spinner("Segmenting via Hugging Face SDK …"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                raw_img.save(tmp, format="JPEG")
                tmp_path = tmp.name
            try:
                result = query_hf(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # ── Error handling ────────────────────────────────────────────────────
        if isinstance(result, dict) and "error" in result:
            st.error(f"API Error: `{result['error']}`")
            st.stop()

        if not isinstance(result, list) or len(result) == 0:
            st.info("No segments returned by the model.")
            st.stop()

        # ── Segment selection (BUG FIX) ───────────────────────────────────────
        best = pick_best_segment(result, img_w, img_h)
        if best is None:
            st.warning("Could not isolate a localised anomaly region.")
            st.stop()

        binary_mask  = best["binary"]
        confidence   = best["score"]
        label        = best["label"].title()
        coverage_pct = best["coverage"] * 100

        annotated_img = draw_tight_bbox(raw_img, binary_mask)
        overlay_img   = make_overlay(raw_img, binary_mask)

        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Scan Results</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence Score", f"{confidence * 100:.2f}%")
        m2.metric("Detected Label",   label)
        m3.metric("Region Coverage",  f"{coverage_pct:.1f}%")
        m4.metric("Segments Found",   str(len(result)))

        # ── Image grid ───────────────────────────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Visual Analysis</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="img-panel"><div class="img-caption">Original MRI</div>', unsafe_allow_html=True)
            st.image(raw_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="img-panel"><div class="img-caption">Bounding Box</div>', unsafe_allow_html=True)
            st.image(annotated_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="img-panel"><div class="img-caption">Mask Overlay</div>', unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Segment table ─────────────────────────────────────────────────────
        with st.expander("📋  All Detected Segments"):
            rows = []
            for idx, seg in enumerate(result):
                m = seg.get("mask")
                if m:
                    np_m  = np.array(m.resize((img_w, img_h), Image.Resampling.NEAREST).convert("L")) > 128
                    cov   = np_m.sum() / (img_w * img_h) * 100
                else:
                    cov = 0.0
                rows.append({
                    "Rank":       idx + 1,
                    "Label":      seg.get("label", "—").title(),
                    "Confidence": f"{seg.get('score', 0)*100:.2f}%",
                    "Coverage":   f"{cov:.1f}%",
                    "Selected":   "✅" if seg is best["seg"] else "",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
