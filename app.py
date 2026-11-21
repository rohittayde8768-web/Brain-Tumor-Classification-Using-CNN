import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------------------------------------
# Brain Tumor Classification - CNN Streamlit Application
# Model: brain_tumor_model.keras (224 x 224 x 3, 4 outputs)
# IMPORTANT: Set CLASS_NAMES in the exact order used during
# model training (e.g. train_generator.class_indices).
# ---------------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

THEMES = {
    "Midnight": {
        "bg": "#0b1020", "panel": "#141c30", "text": "#f4f7ff",
        "muted": "#aab6d3", "accent": "#7c9cff", "border": "#293653"
    },
    "Light": {
        "bg": "#f3f6fb", "panel": "#ffffff", "text": "#172033",
        "muted": "#64748b", "accent": "#315eea", "border": "#dce3ef"
    },
    "Forest": {
        "bg": "#0c1b17", "panel": "#142a22", "text": "#effcf4",
        "muted": "#a9c8b5", "accent": "#53d69a", "border": "#29483a"
    },
    "Purple Glow": {
        "bg": "#160f24", "panel": "#241735", "text": "#fbf5ff",
        "muted": "#c4afd9", "accent": "#c084fc", "border": "#49315f"
    }
}

with st.sidebar:
    st.title("⚙️ Settings")
    theme_name = st.selectbox("Choose app theme", list(THEMES.keys()))
    st.caption("Upload an MRI image to run the CNN model.")
    st.divider()
    st.markdown("**Model information**")
    st.write("Input size: 224 × 224")
    st.write("Output classes: 4")
    st.caption("Prediction is an AI model output, not a medical diagnosis.")

c = THEMES[theme_name]
st.markdown(f"""
<style>
.stApp {{ background: {c['bg']}; color: {c['text']}; }}
[data-testid="stSidebar"] {{ background: {c['panel']}; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
h1, h2, h3, p, label, span, div {{ color: inherit; }}
.hero {{
    background: linear-gradient(120deg, {c['panel']}, {c['bg']});
    border: 1px solid {c['border']}; border-radius: 20px;
    padding: 26px 30px; margin-bottom: 22px;
}}
.hero-title {{ font-size: 2.15rem; font-weight: 800; margin: 0; }}
.hero-sub {{ color: {c['muted']}; margin-top: 8px; font-size: 1rem; }}
.card {{
    background: {c['panel']}; border: 1px solid {c['border']};
    border-radius: 16px; padding: 20px; margin-bottom: 14px;
}}
.result {{
    background: {c['panel']}; border: 1px solid {c['accent']};
    border-radius: 16px; padding: 22px;
}}
.stButton > button {{
    background: {c['accent']}; color: white; border: 0;
    border-radius: 10px; font-weight: 700; padding: 0.6rem 1.2rem;
}}
.stProgress > div > div > div > div {{ background: {c['accent']}; }}
small, .muted {{ color: {c['muted']}; }}
</style>
""", unsafe_allow_html=True)

# Keep this order identical to the class order used when training.
# If you used ImageDataGenerator.flow_from_directory, check:
# print(train_generator.class_indices)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras", compile=False)

st.markdown("""
<div class="hero">
  <div class="hero-title">🧠 Brain Tumor Classification</div>
  <div class="hero-sub">CNN-powered MRI image analysis · Upload, predict, and review model confidence</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📤 Upload MRI image")
    uploaded_file = st.file_uploader(
        "Choose an MRI image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI image", use_container_width=True)
    else:
        st.markdown(
            '<div class="card">Upload a JPG, JPEG, or PNG brain MRI image to begin.</div>',
            unsafe_allow_html=True
        )

with right:
    st.markdown("### 🔍 Prediction")
    if uploaded_file:
        if st.button("Analyze MRI", use_container_width=True):
            try:
                with st.spinner("Loading CNN model and analyzing image..."):
                    model = load_model()
                    if model.input_shape[1:4] != (224, 224, 3):
                        st.warning(
                            f"Model input shape is {model.input_shape}. "
                            "This app currently preprocesses images to 224 × 224 × 3."
                        )

                    resized = image.resize((224, 224))
                    img_array = np.asarray(resized, dtype=np.float32)
                    # This matches models trained on pixel values in [0, 1].
                    # If your training pipeline did NOT divide by 255,
                    # remove the division below.
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array, verbose=0)[0]
                    if len(predictions) != len(CLASS_NAMES):
                        st.error(
                            f"Model returned {len(predictions)} outputs, but "
                            f"{len(CLASS_NAMES)} class labels are configured."
                        )
                    else:
                        best_idx = int(np.argmax(predictions))
                        best_class = CLASS_NAMES[best_idx]
                        confidence = float(predictions[best_idx]) * 100

                        st.markdown(
                            f'<div class="result"><div class="muted">Predicted class</div>'
                            f'<h2 style="margin:6px 0;color:{c["accent"]};">{best_class}</h2>'
                            f'<h1 style="margin:0;">{confidence:.2f}%</h1>'
                            f'<div class="muted">Model confidence</div></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown("#### Class probabilities")
                        for name, score in sorted(
                            zip(CLASS_NAMES, predictions),
                            key=lambda item: item[1], reverse=True
                        ):
                            st.write(f"**{name}** — {float(score) * 100:.2f}%")
                            st.progress(float(np.clip(score, 0, 1)))

                        st.info(
                            "This result is for educational/informational use only. "
                            "It cannot confirm or rule out a brain tumor. Please consult "
                            "a qualified radiologist or medical professional."
                        )
            except FileNotFoundError:
                st.error(
                    "Model file not found. Keep brain_tumor_model.keras "
                    "in the same folder as app.py."
                )
            except Exception as exc:
                st.error(f"Could not analyze this image: {exc}")
    else:
        st.markdown(
            '<div class="card">Your prediction and class probabilities will appear here.</div>',
            unsafe_allow_html=True
        )

st.divider()
st.markdown(
    f'<p style="text-align:center;color:{c["muted"]};">'
    'CNN Brain Tumor Classification · Built with TensorFlow and Streamlit</p>',
    unsafe_allow_html=True
)
