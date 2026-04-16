import base64
import os
import streamlit as st
from groq import Groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan API", page_icon="🧠", layout="centered")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.hero-title { font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #00E5FF, #7B61FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stButton > button { background: linear-gradient(135deg, #00E5FF, #7B61FF); border: none; border-radius: 8px; color: #000; font-weight: 700; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Groq API Setup ───────────────────────────────────────────────────────────
# Best practice: Store this in Streamlit secrets (.streamlit/secrets.toml)
# st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE") 

def get_image_base64(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def analyze_mri_with_groq(base64_image):
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = """
    You are an AI medical assistant specializing in radiology. 
    Analyze this brain MRI scan. 
    1. Do you detect any signs of a tumor or severe anomaly? (Answer clearly YES or NO).
    2. Provide a brief, 2-3 sentence explanation of what you observe.
    Disclaimer: Always state that you are an AI and this is not a medical diagnosis.
    """
    
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview", # Groq's fast vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content

# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API (Groq Vision)</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice (.png or .jpg) for instant LLM-based analysis.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_container_width=True)
    
    if st.button("🔍 Run Groq Analysis"):
        with st.spinner("Analyzing via Groq LPU..."):
            try:
                base64_img = get_image_base64(uploaded_file)
                result = analyze_mri_with_groq(base64_img)
                
                st.markdown("### 📋 Analysis Results")
                st.info(result)
                
            except Exception as e:
                st.error(f"API Error: {e}")

st.divider()
st.caption("⚕️ **Medical Disclaimer:** This tool is for educational prototyping only. Not a medical device.")
