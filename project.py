import streamlit as st
import pickle
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Iris Flower Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected CSS for custom styling, fonts, and dark mode aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Core App Styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Poppins', sans-serif !important;
    background: radial-gradient(circle at 50% 50%, rgb(18, 16, 32) 0%, rgb(10, 10, 15) 100%) !important;
    color: #f1f3f9 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 26px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(168, 85, 247, 0.3);
}

/* Gradient Titles */
.gradient-title {
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Metric Display Cards */
.metric-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Custom Table Design for Sidebar */
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    color: #e2e8f0;
    margin-top: 15px;
}
.metric-table th {
    text-align: left;
    padding: 8px 4px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    color: #c084fc;
    font-weight: 600;
}
.metric-table td {
    padding: 8px 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.status-good {
    color: #4ade80;
    font-weight: 600;
    background: rgba(74, 222, 128, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
}
.status-overfit {
    color: #fb923c;
    font-weight: 600;
    background: rgba(251, 146, 60, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
}
.active-row {
    background-color: rgba(168, 85, 247, 0.15) !important;
    border-left: 3px solid #a855f7;
}

/* Styled progress bars for confidence display */
.prob-container {
    margin-bottom: 16px;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.prob-name {
    font-weight: 500;
    color: #cbd5e1;
}
.prob-value {
    font-weight: 600;
    color: #a855f7;
}
.prob-bar-bg {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    height: 12px;
    width: 100%;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
}

/* Prediction Output Box */
.prediction-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
    border-radius: 20px;
    border: 1px solid rgba(168, 85, 247, 0.25);
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(168, 85, 247, 0.15);
    backdrop-filter: blur(16px);
}
.pred-species-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #ffffff;
    letter-spacing: 0.5px;
}
.pred-species-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 500px;
    margin: 15px auto;
}

/* Sidebar Custom Details */
[data-testid="stSidebar"] {
    background-color: rgb(13, 12, 22) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Streamlit button overrides */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
}
div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Hide standard Streamlit elements for clean look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Image paths
images = {
    0: 'Iris Flowers/Irissetosa1.jpg',
    1: 'Iris Flowers/Iris_versicolor__flo_npyvSQOSVQ8O.jpeg',
    2: 'Iris Flowers/iris_virginica_virginica_lg.jpg'
}

# Auto-train fallback if models file is missing
if not os.path.exists("models.pkl"):
    st.info("Retraining models for 80/20 train/test split. Please wait...")
    import train
    train.train_and_evaluate()
    st.rerun()

# Load models and metrics
with open("models.pkl", "rb") as f:
    data = pickle.load(f)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#a855f7; margin-top:0;'>⚙️ ML Core</h2>",
                    unsafe_allow_html=True)

active_model_name = st.sidebar.selectbox(
    "Select ML Predictor",
    options=list(data['models'].keys()),
    index=list(data['models'].keys()).index(data['best_model_name'])
)

model   = data['models'][active_model_name]
metrics = data['metrics'][active_model_name]

# Active model stats
st.sidebar.markdown("<h4 style='color:#cbd5e1; margin-top:15px;'>Active Model Performance</h4>",
                    unsafe_allow_html=True)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.markdown(f"""
<div class="metric-box">
    <div class="metric-value">{metrics['test_acc']*100:.1f}%</div>
    <div class="metric-label">Test Acc</div>
</div>""", unsafe_allow_html=True)
with col_s2:
    st.markdown(f"""
<div class="metric-box">
    <div class="metric-value">{metrics['test_err']*100:.1f}%</div>
    <div class="metric-label">Test Error</div>
</div>""", unsafe_allow_html=True)

status_cls = "status-good" if "Good" in metrics['status'] else "status-overfit"
st.sidebar.markdown(f"""
<div style="margin-top:12px; text-align:center;">
    <span style="font-size:0.8rem; color:#94a3b8;">Fit Diagnose: </span>
    <span class="{status_cls}">{metrics['status']}</span>
</div>""", unsafe_allow_html=True)

# Algorithm comparison table
st.sidebar.markdown("<h4 style='color:#cbd5e1; margin-top:25px;'>Algorithm Comparison</h4>",
                    unsafe_allow_html=True)

rows = ""
for name, m in data['metrics'].items():
    row_cls = "active-row" if name == active_model_name else ""
    s_cls   = "status-good" if "Good" in m['status'] else "status-overfit"
    rows += f"""
<tr class="{row_cls}">
    <td style="font-weight:{'600' if name==active_model_name else '400'};">{name}</td>
    <td>{m['test_acc']*100:.1f}%</td>
    <td><span class="{s_cls}">{m['status'].split()[0]}</span></td>
</tr>"""

st.sidebar.markdown(f"""
<table class="metric-table">
  <thead><tr><th>Algorithm</th><th>Test Acc</th><th>Fit</th></tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

# ─── MAIN CONTENT ────────────────────────────────────────────────────────────
st.markdown(
    "<h1 class='gradient-title' style='font-size:2.8rem;'>🌸 Iris Classifier Dashboard</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8; font-size:1.1rem; margin-top:-10px; margin-bottom:25px;'>"
    "Final Year Project | Enhanced Multi-Classifier Interactive Deployment</p>",
    unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.1], gap="large")

# ── Left column – inputs ──────────────────────────────────────────────────────
with col_left:
    st.markdown("""
<div class="glass-card">
    <h3 style="color:#ffffff; margin-top:0; margin-bottom:20px; font-size:1.25rem;">
        📐 Flower Dimensions (cm)
    </h3>
    <p style="color:#64748b; font-size:0.85rem; margin-top:-15px; margin-bottom:20px;">
        Use the sliders below to adjust the sepal and petal measurements.
    </p>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1,
                                 help="Length of the flower sepal in cm")
        sepal_width  = st.slider("Sepal Width",  2.0, 4.5, 3.0, 0.1,
                                 help="Width of the flower sepal in cm")
    with c2:
        petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8, 0.1,
                                 help="Length of the flower petal in cm")
        petal_width  = st.slider("Petal Width",  0.1, 2.5, 1.2, 0.1,
                                 help="Width of the flower petal in cm")

    st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("CLASSIFY FLOWER")

# ── Right column – prediction output ─────────────────────────────────────────
with col_right:
    if predict_clicked:
        input_data  = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction  = model.predict(input_data)[0]
        class_name  = data['target_names'][prediction]

        descriptions = {
            0: ("Known for its distinctively small petals and large, prominent sepals. "
                "It is highly resilient and typically thrives in damp, swampy habitats."),
            1: ("Often called the 'Blue Flag', this species features elegant purplish-blue "
                "petals with vibrant yellow and white markings at the base of its sepals."),
            2: ("The 'Virginia Iris' is characterised by its tall, sturdy stem, large bright "
                "violet-blue petals, and rich dark green leaf blades.")
        }

        # Probability bars HTML (built in Python, emitted in one block)
        prob_html = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            prob_html += "<h4 style='color:#ffffff; text-align:left; margin-top:25px; margin-bottom:15px; font-size:1.05rem;'>⚡ Classifier Probabilities</h4>"
            for i, p in enumerate(probs):
                sp_name = data['target_names'][i].title()
                pct     = p * 100
                prob_html += f"""
<div class="prob-container">
  <div class="prob-label-row">
    <span class="prob-name">Iris {sp_name}</span>
    <span class="prob-value">{pct:.1f}%</span>
  </div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{pct}%;"></div>
  </div>
</div>"""

        st.markdown(f"""
<div class="prediction-box">
    <span style="font-size:0.8rem; text-transform:uppercase; letter-spacing:2px;
                 color:#c084fc; font-weight:600;">Prediction Result</span>
    <div class="pred-species-title">Iris {class_name.title()}</div>
    <p class="pred-species-desc">{descriptions[prediction]}</p>
    {prob_html}
</div>""", unsafe_allow_html=True)

        # Flower image rendered by Streamlit (reliable, no base64 needed here)
        img_path = images.get(prediction)
        if img_path and os.path.exists(img_path):
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.image(img_path, caption=f"Iris {class_name.title()}", width='stretch')

    else:
        st.markdown(f"""
<div style="border:1px dashed rgba(255,255,255,0.1); border-radius:20px;
            padding:60px 30px; text-align:center;
            background:rgba(255,255,255,0.01);">
    <div style="font-size:4rem; margin-bottom:20px; opacity:0.6;">🌸</div>
    <h3 style="color:#ffffff; margin-top:0;">Ready for Classification</h3>
    <p style="color:#64748b; font-size:0.95rem; max-width:320px; margin:0 auto 25px;">
        Adjust the slider dimensions on the left panel and click
        <b>Classify Flower</b> to predict its species.
    </p>
    <div style="padding:10px 15px; background:rgba(168,85,247,0.1);
                border-radius:8px; display:inline-block;
                border:1px solid rgba(168,85,247,0.15);">
        <span style="font-size:0.85rem; color:#c084fc;">
            Active Predictor: <b>{active_model_name}</b>
        </span>
    </div>
</div>""", unsafe_allow_html=True)
