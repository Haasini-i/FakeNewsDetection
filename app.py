import streamlit as st
import joblib
import re
import pytesseract
from PIL import Image

# ---- Page settings (must be the first Streamlit command) ----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- Styling ----
st.markdown("""
<style>
/* Hide default Streamlit extras */
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}

/* Background: deep green-black with soft green and silver light */
.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(80, 200, 140, 0.18), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(190, 200, 210, 0.12), transparent 60%),
        radial-gradient(900px 600px at 50% 110%, rgba(80, 200, 140, 0.10), transparent 60%),
        linear-gradient(160deg, #0a0f0d 0%, #0f1a16 50%, #0b1210 100%);
    color: #e8ecea;
}

.block-container {padding-top: 2.5rem; padding-bottom: 6rem; max-width: 760px;}

/* Title with a soft silver-green glow */
h1 {
    text-align: center;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    color: #f2f5f3 !important;
    text-shadow: 0 0 18px rgba(120, 220, 170, 0.45), 0 0 40px rgba(200, 210, 220, 0.15);
}
[data-testid="stCaptionContainer"] {text-align: center; color: #9aa9a2 !important;}

/* Text box with a glow when focused */
textarea {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(200, 210, 220, 0.25) !important;
    border-radius: 12px !important;
    color: #e8ecea !important;
}
textarea:focus {
    border-color: rgba(120, 220, 170, 0.8) !important;
    box-shadow: 0 0 0 1px rgba(120, 220, 170, 0.5), 0 0 22px rgba(120, 220, 170, 0.25) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2e9e6b, #52c98f);
    color: #07110c;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1rem;
    box-shadow: 0 0 20px rgba(82, 201, 143, 0.35);
    transition: all 0.25s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(82, 201, 143, 0.6);
    transform: translateY(-1px);
    color: #07110c;
}

/* Result card and filling bar */
.result-card {
    margin-top: 1.5rem;
    padding: 1.4rem 1.5rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(200, 210, 220, 0.18);
    backdrop-filter: blur(6px);
}
.result-top {display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.9rem;}
.result-label {font-size: 1.35rem; font-weight: 600;}
.result-pct {font-size: 1.1rem; color: #c9d2cd;}
.bar-track {
    width: 100%; height: 16px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.08); overflow: hidden;
}
.bar-fill {height: 100%; border-radius: 999px; animation: fillbar 1.1s ease-out;}
@keyframes fillbar {from {width: 0;}}

.real .result-label {color: #7be0a8;}
.real .bar-fill {background: linear-gradient(90deg, #2e9e6b, #7be0a8); box-shadow: 0 0 16px rgba(123, 224, 168, 0.7);}
.fake .result-label {color: #ff8a8a;}
.fake .bar-fill {background: linear-gradient(90deg, #b33a3a, #ff8a8a); box-shadow: 0 0 16px rgba(255, 138, 138, 0.6);}
.unsure .result-label {color: #f2c14e;}
.unsure .bar-fill {background: linear-gradient(90deg, #b8860b, #f2c14e); box-shadow: 0 0 16px rgba(242, 193, 78, 0.6);}

/* Footer note at the bottom of the screen */
.footer-note {
    position: fixed; left: 0; right: 0; bottom: 0;
    text-align: center; padding: 0.8rem 1rem;
    font-size: 0.8rem; color: #8f9c95;
    background: linear-gradient(to top, rgba(10, 15, 13, 0.95), rgba(10, 15, 13, 0));
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# ---- Load model once ----
@st.cache_resource
def load_files():
    return joblib.load("best_model.joblib"), joblib.load("tfidf.joblib")

model, tfidf = load_files()

def clean(text):
    text = text.lower()
    text = re.sub(r"^.{0,60}\(reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.replace("reuters", "")
    return text

def predict(text):
    nums = tfidf.transform([clean(text)])
    label = model.predict(nums)[0]
    confidence = model.predict_proba(nums)[0][label]
    return label, confidence

def show_result(text):
    if len(text.split()) < 15:
        st.warning("Please provide at least 15 words.")
        return
    label, confidence = predict(text)
    pct = round(confidence * 100)

    if confidence < 0.60:
        css, title = "unsure", "🤔 Uncertain"
    elif label == 1:
        css, title = "real", "✅ Looks REAL"
    else:
        css, title = "fake", "🚩 Looks FAKE"

    st.markdown(f"""
    <div class="result-card {css}">
        <div class="result-top">
            <span class="result-label">{title}</span>
            <span class="result-pct">{pct}% confident</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ---- Main page ----
st.title("📰 Fake News Detector")
st.caption("Paste text, upload an image, or take a photo of a news article.")

mode = st.radio("Input type", ["Text", "Upload image", "Take photo"], horizontal=True)

if mode == "Text":
    text = st.text_area("Paste a news article", height=220)
    if st.button("Check", use_container_width=True):
        show_result(text)

elif mode == "Upload image":
    uploaded = st.file_uploader("Upload an image of a news article", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        extracted = pytesseract.image_to_string(image)
        with st.expander("Text read from the image"):
            st.write(extracted)
        show_result(extracted)

elif mode == "Take photo":
    photo = st.camera_input("Take a photo of a news article")
    if photo:
        extracted = pytesseract.image_to_string(Image.open(photo))
        with st.expander("Text read from the photo"):
            st.write(extracted)
        show_result(extracted)

# ---- Footer note ----
st.markdown(
    '<div class="footer-note">It judges writing style, not facts. '
    'Always verify important news from trusted sources.</div>',
    unsafe_allow_html=True,
)