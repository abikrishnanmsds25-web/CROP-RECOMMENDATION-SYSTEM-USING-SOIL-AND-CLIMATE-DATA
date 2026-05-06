"""
Crop Recommendation System — Streamlit App
==========================================
Loads artifacts/models.pkl and serves predictions via a polished UI.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense — Crop Recommendation AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root theme */
:root {
    --green-dark:  #1a3d2b;
    --green-mid:   #2e6b47;
    --green-light: #4caf50;
    --amber:       #f5a623;
    --cream:       #fdf6ec;
    --card-bg:     #ffffff;
    --text-main:   #1c2b1e;
    --text-muted:  #5a7260;
    --border:      #d4e6d0;
    --shadow:      0 4px 24px rgba(30,80,45,0.10);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--text-main);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #1a3d2b 0%, #2e6b47 60%, #4caf50 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(26,61,43,0.18);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "🌿";
    font-size: 8rem;
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    opacity: 0.12;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.75);
    font-weight: 300;
    margin: 0;
}

/* Cards */
.info-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid var(--green-light);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    text-align: center;
    box-shadow: 0 6px 32px rgba(76,175,80,0.15);
}
.result-crop {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: var(--green-dark);
    margin: 0.4rem 0;
    text-transform: capitalize;
}
.result-conf {
    font-size: 1rem;
    color: var(--text-muted);
    margin: 0;
}
.conf-pill {
    display: inline-block;
    background: var(--green-light);
    color: #fff;
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-weight: 600;
    font-size: 0.95rem;
    margin-left: 0.4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
}
[data-testid="stSidebar"] * {
    color: #e0f0e6 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider > label {
    color: #c8e6c9 !important;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2e6b47, #4caf50);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.7rem 2.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px rgba(46,107,71,0.25);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46,107,71,0.35);
}

/* Metric tiles */
.metric-tile {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow);
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--green-mid);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.2rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Selectbox label */
.model-select label { color: var(--text-muted) !important; font-size: 0.85rem; }

/* Prob bar */
.prob-row {
    display: flex; align-items: center; margin-bottom: 0.45rem; gap: 0.6rem;
}
.prob-name { min-width: 110px; font-size: 0.82rem; text-transform: capitalize; }
.prob-bar-bg {
    flex: 1; background: #e8f5e9; border-radius: 20px; height: 10px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 20px;
    background: linear-gradient(90deg, #4caf50, #81c784);
    transition: width 0.6s ease;
}
.prob-pct { min-width: 44px; font-size: 0.8rem; color: var(--text-muted); text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    model_path = "artifacts/models.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

bundle = load_models()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🌾 CropSense</p>
    <p class="hero-sub">AI-powered crop recommendation from soil & climate parameters</p>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("⚠️ `artifacts/models.pkl` not found. Run the training script first.")
    st.stop()

# Unpack bundle
rf_model    = bundle["random_forest"]
knn_model   = bundle["knn"]
dt_model    = bundle["decision_tree"]
scaler      = bundle["scaler"]
le          = bundle["label_encoder"]
feat_names  = bundle["feature_names"]
class_names = bundle["class_names"]
summary_df  = bundle.get("results_summary", None)

model_map = {
    "🌲 Random Forest (Best)" : rf_model,
    "📐 Decision Tree"        : dt_model,
    "🔵 K-Nearest Neighbors"  : knn_model,
}

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("Adjust soil & climate values below.")
    st.markdown("---")

    # Soil nutrients
    st.markdown("### 🧪 Soil Nutrients")
    N  = st.slider("Nitrogen (N)  mg/kg",    0,   140,  50)
    P  = st.slider("Phosphorus (P) mg/kg",   5,   145,  50)
    K  = st.slider("Potassium (K)  mg/kg",   5,   205,  50)
    pH = st.slider("pH",                      3.5,  9.5, 6.5, step=0.1)

    st.markdown("### 🌤️ Climate")
    temp     = st.slider("Temperature (°C)",   8.0,  44.0, 25.0, step=0.5)
    humidity = st.slider("Humidity (%)",        14.0, 100.0, 70.0, step=0.5)
    rainfall = st.slider("Rainfall (mm)",        20.0, 300.0, 100.0, step=1.0)

    st.markdown("---")
    st.markdown("### 🤖 Model")
    selected_model_name = st.selectbox("Choose model", list(model_map.keys()))
    selected_model = model_map[selected_model_name]

    st.markdown("---")
    predict_btn = st.button("🌱 Recommend Crop")


# ── Feature engineering helper (mirrors training pipeline) ───────────────────
def build_features(N, P, K, temp, humidity, rainfall, pH, feat_names):
    """Build the same feature vector used during training."""
    base = {
        "N": N, "P": P, "K": K,
        "temperature": temp, "humidity": humidity,
        "rainfall": rainfall, "ph": pH,
    }
    # Common engineered features — add only those present in feat_names
    extras = {
        "NPK_ratio"           : N / (P + K + 1),
        "NP_ratio"            : N / (P + 1),
        "NK_ratio"            : N / (K + 1),
        "PK_ratio"            : P / (K + 1),
        "NPK_sum"             : N + P + K,
        "NPK_product"         : N * P * K,
        "temp_humidity"       : temp * humidity,
        "temp_rainfall"       : temp * rainfall,
        "humidity_rainfall"   : humidity * rainfall,
        "ph_temp"             : pH * temp,
        "ph_squared"          : pH ** 2,
        "temp_squared"        : temp ** 2,
        "rainfall_squared"    : rainfall ** 2,
        "humidity_squared"    : humidity ** 2,
        "N_squared"           : N ** 2,
        "log_rainfall"        : np.log1p(rainfall),
        "log_N"               : np.log1p(N),
        "log_P"               : np.log1p(P),
        "log_K"               : np.log1p(K),
        "sqrt_rainfall"       : np.sqrt(rainfall),
        "sqrt_humidity"       : np.sqrt(humidity),
        "ph_neutral_dist"     : abs(pH - 7.0),
        "optimal_temp_dist"   : abs(temp - 25.0),
        "temp_humidity_index" : temp - 0.55 * (1 - humidity / 100) * (temp - 14.5),
    }
    row = {**base, **extras}
    vec = np.array([row.get(f, 0.0) for f in feat_names]).reshape(1, -1)
    return vec


# ── Main area layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    # Current input summary
    st.markdown("### 📋 Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, unit in zip(
        [c1, c2, c3, c4],
        ["Nitrogen", "Phosphorus", "Potassium", "pH"],
        [N, P, K, pH], ["mg/kg", "mg/kg", "mg/kg", ""]
    ):
        col.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}<br><span style="font-size:0.65rem">{unit}</span></div>
        </div>""", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    for col, label, value, unit in zip(
        [c5, c6, c7],
        ["Temperature", "Humidity", "Rainfall"],
        [temp, humidity, rainfall], ["°C", "%", "mm"]
    ):
        col.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}<br><span style="font-size:0.65rem">{unit}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Model performance table
    if summary_df is not None:
        st.markdown("### 📊 Model Performance")
        st.dataframe(
            summary_df.set_index("Model").style
            .background_gradient(cmap="Greens", subset=["Test Acc", "F1 Macro", "ROC-AUC", "CV Mean"])
            .format(precision=4),
            use_container_width=True,
        )

with col_right:
    st.markdown("### 🌱 Prediction")

    if predict_btn:
        raw_vec = build_features(N, P, K, temp, humidity, rainfall, pH, feat_names)

        # Scale
        try:
            vec = scaler.transform(raw_vec)
        except Exception:
            vec = raw_vec  # fallback if scaler mismatch

        # Predict
        pred_idx    = selected_model.predict(vec)[0]
        pred_proba  = selected_model.predict_proba(vec)[0]
        pred_label  = le.inverse_transform([pred_idx])[0] if hasattr(le, "inverse_transform") else class_names[pred_idx]
        confidence  = pred_proba[pred_idx] * 100

        # Crop emoji map
        CROP_EMOJI = {
            "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
            "pigeonpeas": "🫛", "mothbeans": "🌿", "mungbean": "🌿",
            "blackgram": "⚫", "lentil": "🟤", "pomegranate": "🍎",
            "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉",
            "muskmelon": "🍈", "apple": "🍎", "orange": "🍊", "papaya": "🍈",
            "coconut": "🥥", "cotton": "🪶", "jute": "🌿", "coffee": "☕",
            "wheat": "🌾", "tobacco": "🍃", "sugarcane": "🎋",
        }
        emoji = CROP_EMOJI.get(pred_label.lower(), "🌿")

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:4rem; line-height:1; margin-bottom:0.3rem">{emoji}</div>
            <div class="result-crop">{pred_label.title()}</div>
            <p class="result-conf">
                Confidence: <span class="conf-pill">{confidence:.1f}%</span>
            </p>
            <p style="font-size:0.78rem; color:#5a7260; margin-top:0.6rem">
                Model: {selected_model_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Top-5 probabilities bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Top crop probabilities**")

        top5_idx  = np.argsort(pred_proba)[::-1][:8]
        top5_bars = ""
        for i in top5_idx:
            c_name = le.inverse_transform([i])[0] if hasattr(le, "inverse_transform") else class_names[i]
            pct    = pred_proba[i] * 100
            width  = pct
            top5_bars += f"""
            <div class="prob-row">
                <span class="prob-name">{c_name.title()}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{width:.1f}%"></div>
                </div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>"""

        st.markdown(top5_bars, unsafe_allow_html=True)

        # Radar-style soil quality hints
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**🧪 Soil Health Hints**")

        hints = []
        if N < 20:   hints.append(("⚠️ Low Nitrogen",    "Consider nitrogen-rich fertilizer or leguminous crops."))
        if P < 15:   hints.append(("⚠️ Low Phosphorus",  "Add phosphate fertilizer to improve root development."))
        if K < 20:   hints.append(("⚠️ Low Potassium",   "Potassium deficiency may reduce crop yield."))
        if pH < 5.5: hints.append(("⚠️ Acidic Soil",     "Apply lime to raise pH closer to neutral."))
        if pH > 8.0: hints.append(("⚠️ Alkaline Soil",   "Soil may need sulfur or organic matter to lower pH."))
        if humidity < 30: hints.append(("💧 Low Humidity", "Irrigation may be necessary."))
        if rainfall < 40: hints.append(("🌧️ Low Rainfall",  "Drip irrigation recommended."))
        if not hints:
            st.success("✅ Soil and climate parameters look well-balanced!")
        else:
            for title, msg in hints:
                st.warning(f"**{title}** — {msg}")

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:3.5rem">🌿</div>
            <p style="font-size:1.05rem; color:#5a7260; margin-top:1rem">
                Adjust the soil & climate parameters in the sidebar,
                then click <strong>Recommend Crop</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#5a7260; font-size:0.8rem">
    CropSense · Powered by scikit-learn · Built with Streamlit
</p>
""", unsafe_allow_html=True)
