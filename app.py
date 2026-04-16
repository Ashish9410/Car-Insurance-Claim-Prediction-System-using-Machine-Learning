# ============================================================
# 🛡️ AutoShield AI — Premium Car Insurance Intelligence
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import time
import datetime

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AutoShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #080b12;
    --bg2:       #0d1120;
    --card:      rgba(255,255,255,0.035);
    --border:    rgba(255,255,255,0.07);
    --border-h:  rgba(99,179,255,0.35);
    --accent:    #3b82f6;
    --accent2:   #60a5fa;
    --green:     #22c55e;
    --red:       #ef4444;
    --yellow:    #eab308;
    --text:      #e2e8f0;
    --muted:     #475569;
    --subtle:    #1e2a3a;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 10% 0%,   rgba(59,130,246,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 90% 100%,  rgba(249,115,22,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 50% 50%,   rgba(16,185,129,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1380px !important; position: relative; z-index: 1; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--subtle); border-radius: 3px; }

/* NAV */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.nav-logo { font-size: 1.25rem; font-weight: 800; color: var(--text); display: flex; align-items: center; gap: 0.5rem; letter-spacing: -0.02em; }
.nav-logo span { color: var(--accent2); }
.nav-pill { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); color: var(--accent2); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 50px; }
.nav-time { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--muted); }

/* HERO */
.hero { text-align: center; padding: 2rem 0 3rem; }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); color: var(--accent2); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.35rem 1rem; border-radius: 50px; margin-bottom: 1.3rem; }
.hero-eyebrow::before { content: '●'; color: var(--green); font-size: 0.5rem; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.hero h1 { font-size: clamp(2.6rem, 4.5vw, 4rem); font-weight: 800; line-height: 1.08; letter-spacing: -0.03em; margin-bottom: 1rem; background: linear-gradient(135deg, #ffffff 0%, #93c5fd 40%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero p { font-size: 1rem; color: var(--muted); max-width: 480px; margin: 0 auto; line-height: 1.75; font-weight: 400; }

/* KPI */
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border-radius: 16px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 2.5rem; }
.kpi-cell { background: var(--bg2); padding: 1.1rem 1.4rem; display: flex; flex-direction: column; gap: 0.2rem; transition: background 0.2s; }
.kpi-cell:hover { background: rgba(59,130,246,0.06); }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 500; color: var(--accent2); }
.kpi-lbl { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* SECTION LABEL */
.sec-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* CARD */
.card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1.6rem; margin-bottom: 1.2rem; transition: border-color 0.25s; position: relative; overflow: hidden; }
.card:hover { border-color: var(--border-h); }
.card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(99,179,255,0.3), transparent); opacity: 0; transition: opacity 0.3s; }
.card:hover::before { opacity: 1; }
.card-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.4rem; }
.card-icon { width: 32px; height: 32px; background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.2); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; flex-shrink: 0; }
.card-title { font-size: 0.9rem; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
.card-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }

/* FORM */
[data-testid="stWidgetLabel"] p, label { font-family: 'Outfit', sans-serif !important; font-size: 0.76rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stSlider"] > div > div > div { background: rgba(59,130,246,0.15) !important; height: 4px !important; border-radius: 2px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; box-shadow: 0 0 12px rgba(59,130,246,0.6) !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }
[data-testid="stNumberInput"] input { background: rgba(255,255,255,0.04) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stFileUploader"] { background: rgba(59,130,246,0.03) !important; border: 1.5px dashed rgba(59,130,246,0.25) !important; border-radius: 14px !important; }
[data-testid="stFileUploader"]:hover { border-color: rgba(59,130,246,0.5) !important; }

/* BUTTON */
.stButton > button { background: var(--accent) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.8rem 2rem !important; font-family: 'Outfit', sans-serif !important; font-size: 0.95rem !important; font-weight: 700 !important; width: 100% !important; box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(59,130,246,0.45) !important; }

/* METRIC */
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 1.1rem 1.3rem !important; }
[data-testid="stMetricLabel"] p { font-family: 'Outfit', sans-serif !important; font-size: 0.68rem !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.7rem !important; font-weight: 500 !important; color: var(--text) !important; }

/* PROGRESS */
[data-testid="stProgress"] > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 50px !important; height: 6px !important; }
[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; border-radius: 50px !important; box-shadow: 0 0 10px rgba(59,130,246,0.4) !important; }

/* TABS */
[data-testid="stTabs"] [role="tablist"] { background: var(--bg2) !important; border-radius: 10px !important; padding: 0.25rem !important; gap: 0.2rem !important; border: 1px solid var(--border) !important; }
[data-testid="stTabs"] [role="tab"] { font-family: 'Outfit', sans-serif !important; font-size: 0.82rem !important; font-weight: 600 !important; color: var(--muted) !important; border-radius: 8px !important; padding: 0.4rem 1rem !important; border: none !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: rgba(59,130,246,0.15) !important; color: var(--accent2) !important; }

[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid var(--border) !important; }
[data-testid="stAlert"] { border-radius: 12px !important; font-family: 'Outfit', sans-serif !important; font-size: 0.85rem !important; border: none !important; }

/* CUSTOM */
.score-ring-wrap { display: flex; flex-direction: column; align-items: center; padding: 1.5rem 0; }
.score-num { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; font-weight: 500; line-height: 1; margin-bottom: 0.3rem; }
.score-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }

.verdict-box { border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center; margin: 1rem 0; display: flex; align-items: center; justify-content: center; gap: 0.6rem; }
.verdict-approve { background: rgba(34,197,94,0.07); border: 1px solid rgba(34,197,94,0.25); color: #4ade80; }
.verdict-review  { background: rgba(234,179,8,0.07);  border: 1px solid rgba(234,179,8,0.25);  color: #facc15; }
.verdict-deny    { background: rgba(239,68,68,0.07);  border: 1px solid rgba(239,68,68,0.25);  color: #f87171; }
.verdict-text { font-size: 1rem; font-weight: 700; letter-spacing: -0.01em; }
.verdict-sub  { font-size: 0.72rem; font-weight: 400; opacity: 0.7; margin-top: 0.15rem; }

.factor-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.65rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.factor-row:last-child { border-bottom: none; }
.factor-icon { font-size: 1rem; width: 24px; text-align: center; flex-shrink: 0; }
.factor-body { flex: 1; }
.factor-name { font-size: 0.78rem; font-weight: 600; color: #94a3b8; }
.factor-desc { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }
.factor-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; padding: 0.2rem 0.55rem; border-radius: 6px; font-weight: 500; flex-shrink: 0; }
.badge-low  { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.badge-med  { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.2); }
.badge-high { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.badge-ok   { background: rgba(59,130,246,0.12); color: var(--accent2); border: 1px solid rgba(59,130,246,0.2); }

.history-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 0.9rem; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 0.5rem; transition: background 0.2s; }
.history-row:hover { background: rgba(59,130,246,0.05); }
.history-score { font-family:'JetBrains Mono',monospace; font-size:0.9rem; font-weight:500; min-width:50px; }
.history-meta  { flex:1; font-size:0.75rem; color:var(--muted); }
.history-verdict { font-size:0.7rem; font-weight:600; }

.tip-box { background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.18); border-radius: 12px; padding: 1rem 1.2rem; margin-top: 0.8rem; }
.tip-title { font-size:0.72rem; font-weight:700; color:var(--accent2); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
.tip-text  { font-size:0.8rem; color:#64748b; line-height:1.6; }

.placeholder-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 4rem 2rem; background: var(--card); border: 1.5px dashed var(--border); border-radius: 20px; min-height: 340px; }
.placeholder-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.2; }
.placeholder-title { font-size: 1rem; font-weight: 700; color: #1e2a3a; margin-bottom: 0.4rem; }
.placeholder-sub { font-size: 0.78rem; color: #1e2a3a; max-width: 220px; line-height: 1.6; }

.mini-bar-wrap { margin: 0.4rem 0 0.8rem; }
.mini-bar-label { display:flex; justify-content:space-between; font-size:0.7rem; color:var(--muted); margin-bottom:0.3rem; }
.mini-bar-track { background:rgba(255,255,255,0.06); border-radius:50px; height:5px; overflow:hidden; }
.mini-bar-fill  { height:100%; border-radius:50px; }
.fill-blue   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.fill-green  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.fill-orange { background: linear-gradient(90deg, #f97316, #fb923c); }
.fill-red    { background: linear-gradient(90deg, #ef4444, #f87171); }

.divider { height:1px; background:var(--border); margin:1.2rem 0; }

.footer { text-align: center; padding: 3rem 0 1rem; border-top: 1px solid var(--border); margin-top: 3rem; color: var(--muted); font-size: 0.72rem; letter-spacing: 0.06em; display: flex; align-items: center; justify-content: center; gap: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────
@st.cache_resource
def load_numeric_model():
    return joblib.load("car_claim_numeric_model.pkl")

@st.cache_resource
def load_image_model():
    device = torch.device("cpu")
    m = torchvision.models.alexnet(weights="IMAGENET1K_V1")
    m.classifier[6] = nn.Linear(4096, 2)
    m.load_state_dict(torch.load("best_image_model.pth", map_location=device))
    m.eval()
    return m

@st.cache_data
def load_train_data():
    df = pd.read_csv("train.csv")
    for col in ["max_power", "gross_weight"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
        )
        df[col].fillna(df[col].mean(), inplace=True)
    return df

numeric_model = load_numeric_model()
train_df      = load_train_data()
template      = train_df.drop(columns=["is_claim"]).iloc[0:1].copy()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
CLASSES = ["Normal", "Suspicious"]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def compute_risk_factors(age, car_age, fuel, airbags, ncap, transmission):
    factors = []
    if age < 25:
        factors.append(("👤", "Driver Age", f"{age} yrs — Young driver, elevated risk", "high", 0.75))
    elif age > 65:
        factors.append(("👤", "Driver Age", f"{age} yrs — Senior driver, moderate risk", "med", 0.55))
    else:
        factors.append(("👤", "Driver Age", f"{age} yrs — Prime age, low risk", "low", 0.2))

    if car_age > 12:
        factors.append(("🚗", "Vehicle Age", f"{car_age} yrs — High wear, elevated risk", "high", 0.8))
    elif car_age > 6:
        factors.append(("🚗", "Vehicle Age", f"{car_age} yrs — Mid-life vehicle", "med", 0.5))
    else:
        factors.append(("🚗", "Vehicle Age", f"{car_age} yrs — Newer vehicle, lower risk", "low", 0.2))

    if airbags >= 6:
        factors.append(("🛡️", "Safety Package", f"{airbags} airbags — Excellent protection", "low", 0.15))
    elif airbags >= 2:
        factors.append(("🛡️", "Safety Package", f"{airbags} airbags — Standard protection", "ok", 0.35))
    else:
        factors.append(("🛡️", "Safety Package", f"{airbags} airbags — Minimal protection", "high", 0.7))

    if ncap >= 4:
        factors.append(("⭐", "Safety Rating", f"{ncap}/5 NCAP — Top rated", "low", 0.1))
    elif ncap >= 2:
        factors.append(("⭐", "Safety Rating", f"{ncap}/5 NCAP — Average", "med", 0.45))
    else:
        factors.append(("⭐", "Safety Rating", f"{ncap}/5 NCAP — Poorly rated", "high", 0.75))

    fuel_map2 = {"Petrol": ("ok", 0.35), "Diesel": ("med", 0.5), "CNG": ("low", 0.2)}
    fr, fv = fuel_map2.get(fuel, ("ok", 0.35))
    factors.append(("⛽", "Fuel Type", f"{fuel} — {'Standard risk' if fr=='ok' else 'Slightly higher risk' if fr=='med' else 'Lower risk'}", fr, fv))

    if transmission == "Manual":
        factors.append(("⚙️", "Transmission", "Manual — Higher driver-error risk", "med", 0.5))
    else:
        factors.append(("⚙️", "Transmission", "Automatic — Lower driver-error risk", "low", 0.25))
    return factors

def badge_class(level):
    return {"low":"badge-low","med":"badge-med","high":"badge-high","ok":"badge-ok"}.get(level,"badge-ok")

def badge_label(level):
    return {"low":"LOW","med":"MED","high":"HIGH","ok":"OK"}.get(level,"OK")

def fill_class(val):
    if val < 0.3: return "fill-green"
    if val < 0.6: return "fill-orange"
    return "fill-red"


# ─────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────
now = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">🛡️ Auto<span>Shield</span> AI</div>
    <div class="nav-pill">v2.0 · Live</div>
    <div class="nav-time">{now}</div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Insurance Intelligence</div>
    <h1>Smart Claim Analysis<br>in Seconds</h1>
    <p>Enter policy details below. Our dual AI engine combines structured risk modeling with computer vision to evaluate every claim.</p>
</div>
""", unsafe_allow_html=True)

# KPI STRIP
st.markdown("""
<div class="kpi-row">
    <div class="kpi-cell"><div class="kpi-val">2</div><div class="kpi-lbl">AI Models Active</div></div>
    <div class="kpi-cell"><div class="kpi-val">43</div><div class="kpi-lbl">Risk Features</div></div>
    <div class="kpi-cell"><div class="kpi-val">XGBoost</div><div class="kpi-lbl">Numeric Engine</div></div>
    <div class="kpi-cell"><div class="kpi-val">AlexNet</div><div class="kpi-lbl">Vision Engine</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1], gap="large")

# ════════════════════════════════
# LEFT — INPUT
# ════════════════════════════════
with left_col:

    st.markdown('<div class="sec-label">Policy Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-head"><div class="card-icon">👤</div><div><div class="card-title">Policyholder Profile</div><div class="card-sub">Basic demographic and vehicle details</div></div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Driver Age", 18, 80, 32)
    with c2:
        car_age = st.slider("Car Age (yrs)", 0, 20, 4)

    c3, c4 = st.columns(2)
    with c3:
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    with c4:
        airbags = st.slider("Airbags", 0, 10, 4)

    c5, c6 = st.columns(2)
    with c5:
        ncap = st.slider("NCAP Safety Rating", 0, 5, 3)
    with c6:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Claim Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-head"><div class="card-icon">📋</div><div><div class="card-title">Incident Details</div><div class="card-sub">Tell us about the claim being filed</div></div></div>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        claim_type = st.selectbox("Claim Type", ["Accident", "Theft", "Natural Damage", "Fire", "Vandalism"])
    with c8:
        claim_amount = st.number_input("Claimed Amount (₹)", min_value=1000, max_value=2000000, value=50000, step=5000)

    c9, c10 = st.columns(2)
    with c9:
        policy_tenure = st.slider("Policy Tenure (yrs)", 1, 10, 3)
    with c10:
        prev_claims = st.selectbox("Previous Claims", ["None", "1 claim", "2 claims", "3+ claims"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Vehicle Image (Optional)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-head"><div class="card-icon">📷</div><div><div class="card-title">Damage Photo Analysis</div><div class="card-sub">Vision AI checks for fraud patterns</div></div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        st.image(Image.open(uploaded_file).convert("RGB"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run AI Claim Analysis", use_container_width=True)


# ════════════════════════════════
# RIGHT — RESULTS
# ════════════════════════════════
with right_col:

    if not predict_btn:
        st.markdown("""
        <div class="placeholder-wrap">
            <div class="placeholder-icon">🛡️</div>
            <div class="placeholder-title">No Analysis Yet</div>
            <div class="placeholder-sub">Complete the form and click "Run AI Claim Analysis" to see results.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running dual-engine AI analysis..."):
            time.sleep(0.4)

            # FEATURE PREP
            input_data = template.copy()
            fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            input_data["age_of_policyholder"] = age
            input_data["age_of_car"]          = car_age
            input_data["fuel_type"]           = fuel_map[fuel]
            input_data["airbags"]             = airbags
            input_data["policy_tenure"]       = policy_tenure
            if "ncap_rating" in input_data.columns:
                input_data["ncap_rating"] = ncap
            if "transmission_type" in input_data.columns:
                input_data["transmission_type"] = 0 if transmission == "Manual" else 1
            input_data["max_power"]           = train_df["max_power"].mean() + age
            input_data["gross_weight"]        = train_df["gross_weight"].mean() + car_age * 10
            input_data["engine_per_weight"]   = input_data["max_power"] / (input_data["gross_weight"] + 1)
            input_data["age_ratio"]           = car_age / (age + 1)
            input_data.fillna(0, inplace=True)

            # NUMERIC PREDICTION
            claim_prob = round(float(numeric_model.predict_proba(input_data)[:, 1][0]), 4)

            # PENALTIES
            amount_penalty = 0.08 if claim_amount > 500000 else 0.04 if claim_amount > 200000 else 0
            prev_map = {"None": 0, "1 claim": 0.04, "2 claims": 0.08, "3+ claims": 0.14}
            adjusted_prob = min(claim_prob + amount_penalty + prev_map.get(prev_claims, 0), 0.99)

            # IMAGE PREDICTION
            confidence = 0.0
            image_pred = "Not Provided"
            if uploaded_file:
                image_model = load_image_model()
                img = Image.open(uploaded_file).convert("RGB")
                img_t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out   = image_model(img_t)
                    probs = torch.softmax(out, dim=1)
                    pred  = torch.argmax(probs, 1).item()
                    confidence = round(float(probs[0][pred].item()), 4)
                image_pred = "Uncertain" if confidence < 0.6 else CLASSES[pred]

            # FINAL SCORE
            if not uploaded_file or confidence < 0.6:
                approval_score = 1 - adjusted_prob
            else:
                fraud_penalty  = 0.2 if image_pred == "Suspicious" else 0
                approval_score = max((1 - adjusted_prob) * 0.75 + confidence * 0.25 - fraud_penalty, 0)
            approval_score = round(approval_score, 4)

            # VERDICT
            if approval_score > 0.68:
                verdict_cls, verdict_icon, verdict_text, verdict_sub = (
                    "verdict-approve", "✅", "Auto Approve",
                    "Low risk profile — recommended for fast-track approval"
                )
            elif approval_score > 0.40:
                verdict_cls, verdict_icon, verdict_text, verdict_sub = (
                    "verdict-review", "⚠️", "Manual Review Required",
                    "Moderate risk — assign to underwriter for verification"
                )
            else:
                verdict_cls, verdict_icon, verdict_text, verdict_sub = (
                    "verdict-deny", "❌", "Flag for Investigation",
                    "High risk signals detected — do not auto-approve"
                )

            # SAVE HISTORY
            st.session_state.history.insert(0, {
                "time": datetime.datetime.now().strftime("%H:%M"),
                "score": approval_score,
                "verdict": verdict_text,
                "verdict_icon": verdict_icon,
                "age": age,
                "car_age": car_age,
                "amount": claim_amount,
                "claim_type": claim_type,
            })
            st.session_state.history = st.session_state.history[:6]

        # RESULTS TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Risk Factors", "📷 Vision AI", "🕒 History"])

        # ── OVERVIEW ──
        with tab1:
            st.markdown(f"""
            <div class="score-ring-wrap">
                <div class="score-num" style="color:{'#4ade80' if approval_score>0.68 else '#facc15' if approval_score>0.40 else '#f87171'}">
                    {approval_score*100:.1f}%
                </div>
                <div class="score-label">Approval Probability</div>
            </div>
            <div class="verdict-box {verdict_cls}">
                <div>
                    <div class="verdict-text">{verdict_icon} {verdict_text}</div>
                    <div class="verdict-sub">{verdict_sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Base Risk",     f"{claim_prob*100:.1f}%")
            with m2: st.metric("Adjusted Risk", f"{adjusted_prob*100:.1f}%")
            with m3: st.metric("Approval",      f"{approval_score*100:.1f}%")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label" style="margin-top:0">Score Breakdown</div>', unsafe_allow_html=True)

            for label, val, fc in [
                ("Model Risk Score",  claim_prob,    fill_class(claim_prob)),
                ("Adjusted Risk",     adjusted_prob, fill_class(adjusted_prob)),
                ("Approval Score",    approval_score,"fill-blue"),
            ]:
                pct = int(val * 100)
                st.markdown(f"""
                <div class="mini-bar-wrap">
                    <div class="mini-bar-label"><span>{label}</span><span>{pct}%</span></div>
                    <div class="mini-bar-track"><div class="mini-bar-fill {fc}" style="width:{pct}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

            # Claim-amount band
            band = "🟢 Low" if claim_amount < 100000 else "🟡 Medium" if claim_amount < 500000 else "🔴 High"
            st.markdown(f"""
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.6rem;">
                <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:50px;padding:0.25rem 0.8rem;font-size:0.72rem;color:#94a3b8;">
                    {claim_type} claim
                </span>
                <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:50px;padding:0.25rem 0.8rem;font-size:0.72rem;color:#94a3b8;">
                    ₹{claim_amount:,} — {band}
                </span>
                <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:50px;padding:0.25rem 0.8rem;font-size:0.72rem;color:#94a3b8;">
                    {prev_claims} prior
                </span>
            </div>
            """, unsafe_allow_html=True)

            # AI tip
            tip = ""
            if prev_claims != "None":
                tip = f"⚡ Prior claims ({prev_claims}) added a risk penalty. Clean history significantly improves approval odds."
            elif claim_amount > 300000:
                tip = f"💡 High claim amount (₹{claim_amount:,}) triggers additional scrutiny. Thorough documentation is essential."
            elif age < 25:
                tip = "📌 Young drivers statistically have higher claim rates. Additional verification may be needed."
            elif approval_score > 0.68:
                tip = "✅ Strong profile. This claim qualifies for fast-track processing with minimal documentation needed."
            elif image_pred == "Suspicious":
                tip = "🚨 Vision AI flagged the damage photo. A field inspection is strongly recommended before approval."

            if tip:
                st.markdown(f'<div class="tip-box"><div class="tip-title">AI Insight</div><div class="tip-text">{tip}</div></div>', unsafe_allow_html=True)

        # ── RISK FACTORS ──
        with tab2:
            st.markdown('<div class="sec-label">Individual Risk Factor Analysis</div>', unsafe_allow_html=True)
            factors = compute_risk_factors(age, car_age, fuel, airbags, ncap, transmission)
            html_rows = ""
            for icon, name, desc, level, val in factors:
                pct = int(val * 100)
                html_rows += f"""
                <div class="factor-row">
                    <span class="factor-icon">{icon}</span>
                    <div class="factor-body">
                        <div class="factor-name">{name}</div>
                        <div class="factor-desc">{desc}</div>
                        <div class="mini-bar-wrap" style="margin:0.3rem 0 0">
                            <div class="mini-bar-track"><div class="mini-bar-fill {fill_class(val)}" style="width:{pct}%"></div></div>
                        </div>
                    </div>
                    <span class="factor-badge {badge_class(level)}">{badge_label(level)}</span>
                </div>"""
            st.markdown(html_rows, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Claim-type specific advice
            claim_notes = {
                "Accident":       ("med", "Requires FIR report + repair estimate + photos"),
                "Theft":          ("high","Requires FIR, RC book, all keys, NOC from RTO"),
                "Natural Damage": ("low", "Cross-check with local weather authority records"),
                "Fire":           ("high","Requires fire brigade report + forensic assessment"),
                "Vandalism":      ("med", "CCTV footage + police complaint strongly recommended"),
            }
            ct_level, ct_note = claim_notes.get(claim_type, ("ok", "Standard documentation required"))
            st.markdown(f"""
            <div class="factor-row" style="background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:10px;padding:0.8rem 1rem;">
                <span class="factor-icon">📋</span>
                <div class="factor-body">
                    <div class="factor-name">Claim Type — {claim_type}</div>
                    <div class="factor-desc">{ct_note}</div>
                </div>
                <span class="factor-badge {badge_class(ct_level)}">{badge_label(ct_level)}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── VISION AI ──
        with tab3:
            if not uploaded_file:
                st.markdown("""
                <div style="text-align:center;padding:3rem 1rem;color:#1e2a3a;">
                    <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.15">📷</div>
                    <div style="font-size:0.85rem;color:#334155;">No image uploaded.<br>Add a damage photo to activate Vision AI fraud detection.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                col_a, col_b = st.columns(2)
                with col_a: st.metric("Classification", image_pred)
                with col_b: st.metric("Confidence",     f"{confidence*100:.1f}%")

                conf_pct = int(confidence * 100)
                conf_fill = "fill-green" if confidence > 0.75 else "fill-orange" if confidence > 0.5 else "fill-red"
                st.markdown(f"""
                <div class="mini-bar-wrap" style="margin:0.5rem 0 1rem">
                    <div class="mini-bar-label"><span>Model Confidence</span><span>{conf_pct}%</span></div>
                    <div class="mini-bar-track"><div class="mini-bar-fill {conf_fill}" style="width:{conf_pct}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

                if image_pred == "Suspicious":
                    st.error("🚨 **Suspicious pattern detected.** Visual anomalies suggest possible staged or fabricated damage. Field inspection recommended.")
                elif image_pred == "Normal":
                    st.success("✅ **Genuine damage detected.** No visual fraud indicators found in the uploaded photo.")
                else:
                    st.warning("⚠️ **Low confidence.** Image quality insufficient for reliable analysis. Request a clearer photo.")

                st.image(Image.open(uploaded_file).convert("RGB"), use_container_width=True)

                st.markdown("""
                <div class="tip-box">
                    <div class="tip-title">How Vision AI Works</div>
                    <div class="tip-text">
                        The AlexNet-based model was trained to distinguish genuine collision damage from staged or fabricated damage patterns. For best results, upload clear, well-lit photos taken from multiple angles.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── HISTORY ──
        with tab4:
            if not st.session_state.history:
                st.markdown('<div style="text-align:center;color:#1e2a3a;padding:2rem;font-size:0.85rem;">No history yet in this session.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sec-label">Recent Analyses — This Session</div>', unsafe_allow_html=True)
                for h in st.session_state.history:
                    sc = h["score"]
                    color = "#4ade80" if sc > 0.68 else "#facc15" if sc > 0.40 else "#f87171"
                    st.markdown(f"""
                    <div class="history-row">
                        <div class="history-score" style="color:{color}">{sc*100:.0f}%</div>
                        <div class="history-meta">
                            Age {h['age']} · Car {h['car_age']}yr · ₹{h['amount']:,} · {h.get('claim_type','—')}<br>
                            <span style="opacity:0.5">{h['time']}</span>
                        </div>
                        <div class="history-verdict" style="color:{color}">{h['verdict_icon']} {h['verdict']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("🗑️ Clear History"):
                    st.session_state.history = []
                    st.rerun()


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>🛡️ AutoShield AI v2.0</span>
    <span>·</span>
    <span>XGBoost + AlexNet Dual Engine</span>
    <span>·</span>
    <span>AI estimates — not a substitute for professional underwriting</span>
</div>
""", unsafe_allow_html=True)
