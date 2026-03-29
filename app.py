"""
CareSync — Healthcare Diagnosis & Recommendation System
app.py  |  Main entry point

Run:  streamlit run app.py
Note: Run `python train_models.py` once before launching to generate the ML model.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── New modules (auth + database) ──────────────────────────────────────────
from database import init_db, save_prediction, save_metric
from auth import is_logged_in, get_current_user, render_auth_gate, render_sidebar_user
from history import render_history_page
from profile import render_profile_page

# ── Existing utilities (unchanged) ─────────────────────────────────────────
from utils.preprocessing import load_preprocessing_artifacts
from utils.prediction import DiseasePredictor
from utils.medicine_recommender import MedicineRecommender
from utils.diet_planner import DietPlanner
from utils.routine_generator import RoutineGenerator
from utils.analytics import (
    calculate_bmi, get_bmi_category, create_bmi_chart, create_temperature_chart,
    create_symptom_frequency_chart, create_disease_risk_chart,
    create_health_metrics_dashboard, create_trend_analysis,
)


def ensure_list(items):
    if items is None: return []
    if isinstance(items, str): return [items]
    try: return list(items)
    except TypeError: return [items]


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareSync — Healthcare Diagnosis & Recommendation System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise database on startup ─────────────────────────────────────────
init_db()

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --navy:#0A1628;--blue:#4A9EFF;--blue-lt:#60AEFF;--teal:#2DD4BF;
  --amber:#FBBF24;--red:#F87171;
  --bg-base:#141414;--bg-card:#1C1C1C;--bg-raised:#242424;
  --border:rgba(255,255,255,.08);--border-md:rgba(255,255,255,.13);
  --text-1:#F0F0F0;--text-2:#A0A0A0;--text-3:#606060;
  --font-display:'DM Serif Display',Georgia,serif;
  --font-body:'DM Sans',system-ui,sans-serif;
  --radius:12px;--radius-lg:20px;
  --shadow-sm:0 1px 4px rgba(0,0,0,.4);--shadow-lg:0 12px 40px rgba(0,0,0,.6);
}
html,body,[class*="css"]{font-family:var(--font-body) !important;color:var(--text-1);}
.stApp{background:var(--bg-base) !important;}
.block-container{padding-top:2rem !important;}
p,li,span,div,label{color:var(--text-1);}
h1,h2,h3,h4,h5,h6{color:var(--text-1) !important;}
[data-testid="stSidebar"]{background:#0E0E0E !important;border-right:1px solid var(--border) !important;}
[data-testid="stSidebar"] *{color:rgba(255,255,255,.85) !important;}
.stButton>button{background:linear-gradient(135deg,#1E4FBF,#2563EB) !important;color:white !important;
  border:none !important;border-radius:10px !important;font-weight:600 !important;
  padding:.7rem 1.5rem !important;box-shadow:0 2px 12px rgba(37,99,235,.35) !important;}
.stButton>button:hover{opacity:.88 !important;transform:translateY(-1px) !important;}
.stDownloadButton>button{background:transparent !important;color:var(--blue) !important;
  border:1px solid rgba(74,158,255,.35) !important;border-radius:10px !important;font-weight:600 !important;}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;background:var(--bg-card);
  border:1px solid var(--border);border-radius:10px;padding:.3rem;}
.stTabs [data-baseweb="tab"]{border-radius:8px !important;font-size:.84rem !important;
  font-weight:500 !important;color:var(--text-2) !important;}
.stTabs [aria-selected="true"]{background:var(--bg-raised) !important;color:var(--text-1) !important;}
[data-testid="metric-container"]{background:var(--bg-card) !important;
  border:1px solid var(--border) !important;border-radius:var(--radius) !important;padding:1rem 1.25rem !important;}
.mediscan-header{background:linear-gradient(135deg,var(--navy) 0%,#0D2952 50%,var(--blue) 100%);
  border-radius:var(--radius-lg);padding:2.5rem 3rem;margin-bottom:2rem;box-shadow:var(--shadow-lg);
  position:relative;overflow:hidden;}
.mediscan-header h1{font-family:var(--font-display) !important;font-size:2.6rem !important;
  font-weight:400 !important;color:white !important;margin:0 0 .4rem !important;}
.mediscan-header .tagline{color:rgba(255,255,255,.65) !important;font-size:.95rem;
  letter-spacing:.04em;text-transform:uppercase;}
.mediscan-header .badge{display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
  border-radius:100px;padding:4px 14px;font-size:.75rem;color:rgba(255,255,255,.9) !important;
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:1rem;}
.section-title{font-family:var(--font-display) !important;font-size:1.7rem !important;
  color:var(--text-1) !important;font-weight:400 !important;margin:0 0 .2rem !important;}
.section-subtitle{color:var(--text-3) !important;font-size:.85rem;letter-spacing:.03em;
  text-transform:uppercase;margin-bottom:1.6rem !important;}
.med-item{display:flex;align-items:center;gap:.6rem;padding:.6rem .75rem;
  background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;
  margin-bottom:.4rem;font-size:.875rem;color:var(--text-1);}
.med-item .dot-blue{width:7px;height:7px;background:var(--blue);border-radius:50%;flex-shrink:0;}
.med-item .dot-teal{width:7px;height:7px;background:var(--teal);border-radius:50%;flex-shrink:0;}
.med-item .dot-amber{width:7px;height:7px;background:var(--amber);border-radius:50%;flex-shrink:0;}
.chip-wrap{display:flex;flex-wrap:wrap;gap:.5rem;margin:.75rem 0;}
.chip{background:rgba(74,158,255,.1);border:1px solid rgba(74,158,255,.25);color:var(--blue);
  border-radius:100px;padding:4px 14px;font-size:.8rem;font-weight:500;}
.result-banner{background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
  border:1px solid rgba(74,158,255,.2);border-radius:var(--radius-lg);padding:2rem;
  color:white;margin:1.5rem 0;box-shadow:0 8px 32px rgba(0,0,0,.5);}
.result-banner .disease-name{font-family:var(--font-display) !important;font-size:2rem !important;
  color:white !important;margin:0 0 .25rem !important;}
.result-banner .conf-label{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;
  color:rgba(255,255,255,.45);margin-bottom:.25rem;}
.result-banner .conf-value{font-size:1.4rem;font-weight:600;color:#4ADE80;}
.disclaimer{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);
  border-left:4px solid var(--amber);border-radius:var(--radius);padding:1rem 1.25rem;
  font-size:.82rem;color:#D4A72C;line-height:1.6;margin-bottom:1.5rem;}
.info-panel{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:2rem;box-shadow:var(--shadow-sm);}
.sec-divider{border:none;border-top:1px solid var(--border);margin:2rem 0;}
.risk-low{background:rgba(45,212,100,.1);color:#4ADE80;border:1px solid rgba(74,222,128,.25);
  padding:4px 14px;border-radius:100px;font-size:.8rem;font-weight:600;display:inline-block;}
.risk-medium{background:rgba(251,191,36,.1);color:#FBBF24;border:1px solid rgba(251,191,36,.25);
  padding:4px 14px;border-radius:100px;font-size:.8rem;font-weight:600;display:inline-block;}
.risk-high{background:rgba(248,113,113,.1);color:#F87171;border:1px solid rgba(248,113,113,.25);
  padding:4px 14px;border-radius:100px;font-size:.8rem;font-weight:600;display:inline-block;}
.status-ok{background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.2);
  color:#4ADE80 !important;border-radius:8px;padding:.5rem .75rem;font-size:.78rem;
  margin:.5rem 0;text-align:center;}
.status-err{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);
  color:#F87171 !important;border-radius:8px;padding:.5rem .75rem;font-size:.78rem;
  margin:.5rem 0;text-align:center;}
.meal-card{background:var(--bg-raised);border:1px solid var(--border);
  border-radius:var(--radius);padding:1rem 1.25rem;height:100%;}
.meal-card .meal-title{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--text-3);font-weight:600;margin-bottom:.4rem;}
.meal-card .meal-content{font-size:.875rem;color:var(--text-1);line-height:1.5;}
.meal-card .snack-note{font-size:.78rem;color:var(--teal);margin-top:.5rem;}
.day-label{display:inline-flex;align-items:center;gap:.4rem;background:var(--bg-raised);
  border:1px solid var(--border-md);color:var(--text-1);border-radius:100px;
  padding:3px 14px;font-size:.75rem;font-weight:600;letter-spacing:.05em;
  text-transform:uppercase;margin-bottom:1rem;}
.report-header{background:linear-gradient(135deg,#111827 0%,#1E3A5F 100%);
  border:1px solid rgba(74,158,255,.15);border-radius:var(--radius-lg);
  padding:2.5rem;color:white;margin-bottom:2rem;position:relative;overflow:hidden;
  box-shadow:0 8px 32px rgba(0,0,0,.4);}
.report-header .rh-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;
  color:rgba(255,255,255,.4);margin-bottom:.2rem;}
.report-header .rh-name{font-family:var(--font-display) !important;font-size:2rem !important;
  color:white !important;margin:0 !important;}
.report-header .rh-meta{font-size:.85rem;color:rgba(255,255,255,.5);margin-top:.5rem;}
.sidebar-logo{text-align:center;padding:1.5rem 1rem 1rem;}
.sidebar-logo .logo-icon{font-size:2.2rem;display:block;margin-bottom:.3rem;}
.sidebar-logo .logo-name{font-family:var(--font-display) !important;font-size:1.3rem !important;color:white !important;}
.sidebar-logo .logo-ver{font-size:.7rem;color:rgba(255,255,255,.35) !important;
  letter-spacing:.08em;text-transform:uppercase;}
.feat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);
  padding:2rem;position:relative;overflow:hidden;transition:transform .25s,box-shadow .25s;}
.feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  border-radius:var(--radius-lg) var(--radius-lg) 0 0;}
.feat-card.blue::before{background:linear-gradient(90deg,#4A9EFF,#60AEFF);}
.feat-card.teal::before{background:linear-gradient(90deg,#2DD4BF,#5EEAD4);}
.feat-card.amber::before{background:linear-gradient(90deg,#FBBF24,#FCD34D);}
.feat-card.red::before{background:linear-gradient(90deg,#F87171,#FCA5A5);}
.feat-card:hover{transform:translateY(-4px);box-shadow:var(--shadow-lg);}
.feat-card .icon-wrap{width:48px;height:48px;border-radius:12px;display:flex;
  align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:1rem;}
.feat-card.blue .icon-wrap{background:rgba(74,158,255,.12);}
.feat-card.teal .icon-wrap{background:rgba(45,212,191,.12);}
.feat-card.amber .icon-wrap{background:rgba(251,191,36,.12);}
.feat-card.red .icon-wrap{background:rgba(248,113,113,.12);}
.feat-card h3{font-size:1rem !important;font-weight:600 !important;color:var(--text-1) !important;margin:0 0 .5rem !important;}
.feat-card p{font-size:.875rem !important;color:var(--text-2) !important;line-height:1.6 !important;margin:0 !important;}
.spacer-sm{margin-top:.75rem;}.spacer-md{margin-top:1.5rem;}
.text-muted{color:var(--text-2) !important;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


# ── Auth Gate — stop here if not logged in ─────────────────────────────────
if not is_logged_in():
    render_auth_gate()
    st.stop()


# ── Session state defaults ─────────────────────────────────────────────────
_defaults = {
    "predictor": None, "medicine_recommender": None,
    "diet_planner": None, "routine_generator": None,
    "health_history": [], "symptom_history": [],
    "patient_data": {
        "name": "", "age": None, "gender": "", "weight": None,
        "height": None, "temperature": None, "symptoms": [],
        "disease": "", "predictions": [],
    },
    "medicine_result": None, "diet_plan_result": None, "routine_result": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Load pre-trained models ────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"
MODELS_DIR = "models"

@st.cache_resource
def load_pretrained_models():
    predictor = DiseasePredictor()
    if not os.path.exists(MODEL_PATH):
        return None, MedicineRecommender(), DietPlanner(), RoutineGenerator()
    try:
        predictor.load_model(MODEL_PATH)
        le, symptom_list = load_preprocessing_artifacts(MODELS_DIR)
        predictor.label_encoder = le
        predictor.symptom_list  = symptom_list
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        predictor = None
    return predictor, MedicineRecommender(), DietPlanner(), RoutineGenerator()


predictor, medicine_recommender, diet_planner, routine_generator = load_pretrained_models()
st.session_state.predictor            = predictor
st.session_state.medicine_recommender = medicine_recommender
st.session_state.diet_planner         = diet_planner
st.session_state.routine_generator    = routine_generator
current_user = get_current_user()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">⚕️</span>
        <div class="logo-name">CareSync</div>
        <div class="logo-ver">Healthcare Diagnosis System v1.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    render_sidebar_user()

    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "🔬  Symptom Analysis",
        "💊  Medications",
        "🥗  Nutrition Plan",
        "🗓️  Daily Routine",
        "📊  Health Analytics",
        "📋  Clinical Report",
        "🕑  History",
        "👤  Profile",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    if predictor:
        st.markdown('<div class="status-ok">● AI Model Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">● Model Offline — run train_models.py</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:.72rem;color:rgba(255,255,255,.3);padding:.5rem;line-height:1.6;margin-top:.5rem;'>
    For educational purposes only. Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)


# ── Page routing ───────────────────────────────────────────────────────────

if page == "🕑  History":
    render_history_page()
    st.stop()

elif page == "👤  Profile":
    render_profile_page()
    st.stop()


# ===========================================================================
# PAGE: Overview
# ===========================================================================
elif page == "🏠  Overview":
    user_name = (current_user.get("full_name") or current_user["username"]).split()[0] if current_user else "there"

    st.markdown(f"""
    <div class="mediscan-header">
        <div class="badge"><span style="width:6px;height:6px;background:#4CAF50;border-radius:50%;display:inline-block;box-shadow:0 0 0 3px rgba(76,175,80,.3);"></span>AI-Powered · Clinical Grade</div>
        <h1>Welcome back, {user_name} 👋</h1>
        <div class="tagline">Intelligent symptom analysis &amp; personalised health recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, color, icon, title, desc in [
        (col1, "blue",  "🔬", "Symptom Analysis",  "AI-powered disease predictions with confidence scoring."),
        (col2, "teal",  "💊", "Medication Guide",   "Evidence-based medicine recommendations and dosage guidance."),
        (col3, "amber", "🥗", "Nutrition Planning", "Personalised 7-day meal plans for your condition."),
        (col4, "red",   "📊", "Health Analytics",   "Track BMI, temperature trends, and risk scores over time."),
    ]:
        with col:
            st.markdown(f'<div class="feat-card {color}"><div class="icon-wrap">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        steps = [
            ("Navigate to <strong>Symptom Analysis</strong>",        "Select all your current symptoms from the comprehensive list."),
            ("Review your <strong>Diagnosis</strong>",                "Ranked probable conditions with confidence scores."),
            ("Explore <strong>Medications</strong>",                  "OTC, prescriptions, natural remedies, and dosage guidance."),
            ("Follow your <strong>Nutrition &amp; Routine Plan</strong>", "Tailored 7-day diet and a structured daily schedule."),
            ("Track in <strong>Health Analytics</strong>",            "Monitor trends and download your clinical report."),
        ]
        steps_html = "".join(
            f'''<div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:.75rem;">
                <div style="width:28px;height:28px;min-width:28px;background:#4A9EFF;color:#0A0A0A;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:700;">{i}</div>
                <div style="font-size:.9rem;color:#A0A0A0;line-height:1.5;padding-top:4px;">{t}<br>
                    <span style="font-size:.82rem;">{b}</span></div>
            </div>'''
            for i, (t, b) in enumerate(steps, 1)
        )
        st.markdown(f'''
        <div class="info-panel">
            <h3 style="font-size:1.1rem;font-weight:600;color:#F0F0F0;margin:0 0 1rem;">🚀 Getting Started</h3>
            {steps_html}
        </div>
        ''', unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer</strong><br>
            For informational purposes only. Does not constitute medical advice,
            diagnosis, or treatment. Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: Symptom Analysis
# ===========================================================================
elif page == "🔬  Symptom Analysis":
    st.markdown('<p class="section-subtitle">Step 1 of 4</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Symptom Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ Select all symptoms you are currently experiencing. More symptoms = higher accuracy. This is <strong>not</strong> a medical diagnosis.</div>', unsafe_allow_html=True)

    if predictor is None:
        st.error("⚠️ Pre-trained model not found. Please run `python train_models.py` and restart.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### 👤 Patient Profile")
            patient_name_input = st.text_input("Full Name", value=st.session_state.patient_data.get("name", ""), placeholder="e.g. Rahul Sharma")
            c1i, c2i = st.columns(2)
            with c1i:
                age       = st.number_input("Age",         min_value=1,   max_value=120,   value=int(st.session_state.patient_data.get("age") or 30))
                weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=float(st.session_state.patient_data.get("weight") or 70.0))
            with c2i:
                gender    = st.selectbox("Gender", ["Male", "Female", "Other"])
                height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(st.session_state.patient_data.get("height") or 170.0))
            temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

        with col2:
            st.markdown("#### 🤒 Symptom Selection")
            full_symptom_list = predictor.symptom_list or [
                "fever","cough","headache","fatigue","nausea","vomiting","diarrhea",
                "abdominal pain","chest pain","shortness of breath","dizziness",
                "joint pain","muscle pain","sore throat","runny nose",
            ]
            symptom_list_ui   = full_symptom_list[:100]
            selected_symptoms = st.multiselect("Search & select symptoms", options=symptom_list_ui, default=[],
                                               help=f"{len(symptom_list_ui)} symptoms available")
            if selected_symptoms:
                chips = "".join(f'<span class="chip">{s}</span>' for s in selected_symptoms)
                st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                st.caption(f"✓ {len(selected_symptoms)} symptom(s) selected")

        submitted = st.form_submit_button("🔬 Run Analysis", use_container_width=True)

    if submitted:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("Analysing symptom profile …"):
                symptom_vector = np.zeros(len(full_symptom_list))
                for s in selected_symptoms:
                    if s in full_symptom_list:
                        symptom_vector[full_symptom_list.index(s)] = 1
                predictions = predictor.predict(symptom_vector, top_k=3)
                bmi         = calculate_bmi(weight_kg, height_cm / 100)
                risk_score  = predictor.calculate_risk_score(symptom_vector, age, bmi, temperature)

            top        = predictions[0]
            risk_val   = risk_score["total_risk"]
            risk_label = risk_score["risk_level"]
            risk_class = "risk-high" if risk_val > .7 else ("risk-medium" if risk_val > .4 else "risk-low")

            st.markdown(f"""
            <div class="result-banner">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                    <div>
                        <div class="conf-label">Primary Diagnosis</div>
                        <div class="disease-name">{top['disease']}</div>
                        <div style="margin-top:.5rem;"><span class="{risk_class}">Risk: {risk_label}</span></div>
                    </div>
                    <div style="text-align:right;">
                        <div class="conf-label">Model Confidence</div>
                        <div class="conf-value">{top['probability_percent']:.1f}%</div>
                        <div style="font-size:.78rem;color:rgba(255,255,255,.45);margin-top:.3rem;">Based on {len(selected_symptoms)} symptom(s)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Base Risk",       f"{risk_score['base_risk']*100:.1f}%")
            c2.metric("Additional Risk", f"{risk_score['additional_risk']*100:.1f}%")
            c3.metric("Overall Risk",    f"{risk_val*100:.1f}%", delta=risk_label)

            if len(predictions) > 1:
                st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
                st.markdown("#### 🔄 Differential Diagnoses")
                for pred in predictions[1:]:
                    pct = pred["probability_percent"]
                    bw  = int(pct * 3)
                    st.markdown(f"""
                    <div class="med-item">
                        <div class="dot-blue"></div>
                        <div style="flex:1;">{pred['disease']}</div>
                        <div style="display:flex;align-items:center;gap:.75rem;">
                            <div style="width:{bw}px;height:5px;background:var(--blue-lt);border-radius:3px;min-width:4px;"></div>
                            <div style="font-size:.8rem;font-weight:600;color:var(--text-2);width:45px;text-align:right;">{pct:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Save to database ───────────────────────────────────────
            if current_user:
                save_prediction(
                    user_id     = current_user["id"],
                    disease     = top["disease"],
                    confidence  = top["probability_percent"],
                    risk_score  = risk_val,
                    risk_level  = risk_label,
                    symptoms    = selected_symptoms,
                    bmi         = bmi,
                    temperature = temperature,
                )
                save_metric(
                    user_id       = current_user["id"],
                    bmi           = bmi,
                    temperature   = temperature,
                    symptom_count = len(selected_symptoms),
                    risk_score    = risk_val,
                )

            # ── Save to session state ──────────────────────────────────
            st.session_state.last_prediction = {
                "disease": top["disease"], "confidence": top["probability_percent"],
                "symptoms": selected_symptoms, "name": patient_name_input.strip(),
                "age": age, "gender": gender, "weight": weight_kg,
                "height": height_cm, "bmi": bmi, "temperature": temperature,
                "risk_score": risk_val, "timestamp": datetime.now(),
                "all_predictions": predictions,
            }
            st.session_state.patient_data.update({
                "name": patient_name_input.strip(), "age": age, "gender": gender,
                "weight": weight_kg, "height": height_cm, "temperature": temperature,
                "symptoms": selected_symptoms, "disease": top["disease"], "predictions": predictions,
            })
            st.session_state.health_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "bmi": bmi, "temperature": temperature, "symptom_count": len(selected_symptoms),
            })
            for s in selected_symptoms:
                st.session_state.symptom_history.append({"date": datetime.now().strftime("%Y-%m-%d"), "symptom": s})

            st.success("✅ Analysis complete. Navigate to **Medications**, **Nutrition Plan**, or **Daily Routine**.")


# ===========================================================================
# PAGE: Medications
# ===========================================================================
elif page == "💊  Medications":
    st.markdown('<p class="section-subtitle">Step 2 of 4</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Medication Recommendations</h2>', unsafe_allow_html=True)

    patient_name = st.session_state.patient_data.get("name")
    if patient_name:
        st.markdown(f'<p class="text-muted">Patient: <strong>{patient_name}</strong></p>', unsafe_allow_html=True)

    default_disease = st.session_state.get("last_prediction", {}).get("disease", "")
    if default_disease:
        st.info(f"🔬 Condition from analysis: **{default_disease}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        disease_input = st.text_input("Condition / Disease", value=default_disease, placeholder="e.g. Common Cold")
    with col2:
        age_input = st.number_input("Patient Age", min_value=1, max_value=120,
                                    value=st.session_state.get("last_prediction", {}).get("age", 30))

    symptoms_for_meds = st.session_state.get("last_prediction", {}).get("symptoms", [])
    result_generated  = False

    if st.button("💊 Generate Recommendations", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a condition name.")
        else:
            recs = medicine_recommender.recommend_medicines(disease_input, age_input, symptoms=symptoms_for_meds)
            st.session_state.medicine_result = {
                "disease": disease_input, "age": age_input,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "recommendations": recs,
            }
            result_generated = True

    stored = st.session_state.get("medicine_result")
    if stored:
        recs = stored["recommendations"]
        if not result_generated:
            st.caption(f"Showing saved recommendations for **{stored['disease']}** · {stored['generated_at']}")

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 💊 Over-the-Counter")
            for m in recs.get("otc", []):
                st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{m}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("##### 📋 Prescription")
            for m in recs.get("prescription", []):
                st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{m}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown("##### 🌿 Natural Remedies")
            for m in recs.get("natural", []):
                st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{m}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        tabs = st.tabs(["📝 Dosage", "⚠️ Age Considerations", "🚨 Contraindications", "🕒 Timing", "🏥 See a Doctor"])
        with tabs[0]: st.info(recs.get("dosage_detail", recs.get("dosage", "Consult your physician.")))
        with tabs[1]: st.warning(recs.get("age_considerations", "Consult a healthcare provider."))
        with tabs[2]: st.error(recs.get("contraindications_detail", recs.get("contraindications", "Consult a doctor.")))
        with tabs[3]: st.info(recs.get("medication_timing_detail", recs.get("dosage", "Follow doctor's instructions.")))
        with tabs[4]:
            risk_level = "high" if st.session_state.get("last_prediction", {}).get("risk_score", 0) > 0.7 else "moderate"
            when_doc   = medicine_recommender.get_when_to_see_doctor(stored["disease"], risk_level)
            st.markdown("**General Guidelines**")
            for g in when_doc["general"]:
                st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{g}</div>', unsafe_allow_html=True)
            if when_doc["disease_specific"]:
                st.markdown("**Condition-Specific**")
                for g in when_doc["disease_specific"]:
                    st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{g}</div>', unsafe_allow_html=True)

        if recs.get("symptom_specific_guidance"):
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
            st.markdown("##### 🤒 Symptom-Specific Notes")
            for note in recs["symptom_specific_guidance"]:
                st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{note}</div>', unsafe_allow_html=True)

        if recs.get("standardized_categories"):
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
            st.markdown("##### 🗂️ Standardised Treatment Lines")
            sc = st.columns(3)
            cat = recs["standardized_categories"]
            with sc[0]:
                st.markdown("**First-line**")
                for m in cat.get("first_line", []): st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{m}</div>', unsafe_allow_html=True)
            with sc[1]:
                st.markdown("**Second-line**")
                for m in cat.get("second_line", []): st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{m}</div>', unsafe_allow_html=True)
            with sc[2]:
                st.markdown("**Adjunct**")
                for m in cat.get("adjunct", []): st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{m}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-panel" style="text-align:center;padding:3rem;"><div style="font-size:2.5rem;margin-bottom:1rem;">💊</div><div style="font-size:1rem;color:var(--text-2);">Enter a condition and click <strong>Generate Recommendations</strong>.</div></div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: Nutrition Plan
# ===========================================================================
elif page == "🥗  Nutrition Plan":
    st.markdown('<p class="section-subtitle">Step 3 of 4</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Personalised Nutrition Plan</h2>', unsafe_allow_html=True)

    default_disease = st.session_state.get("last_prediction", {}).get("disease", "")
    if default_disease:
        st.info(f"🔬 Condition from analysis: **{default_disease}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        disease_input = st.text_input("Condition / Disease", value=default_disease, placeholder="e.g. Diabetes")
    with col2:
        days = st.number_input("Plan Duration (days)", min_value=1, max_value=7, value=7)

    symptoms_for_diet = st.session_state.get("last_prediction", {}).get("symptoms", [])
    diet_generated    = False

    if st.button("🥗 Generate Nutrition Plan", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a condition name.")
        else:
            meal_plan = diet_planner.generate_meal_plan(disease_input, days, symptoms=symptoms_for_diet)
            st.session_state.diet_plan_result = {
                "disease": disease_input, "days": days,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "plan": meal_plan,
            }
            diet_generated = True

    stored_diet = st.session_state.get("diet_plan_result")
    if stored_diet:
        meal_plan = stored_diet["plan"]
        if not diet_generated:
            st.caption(f"Showing saved plan for **{stored_diet['disease']}** · {stored_diet['generated_at']}")

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ✅ Recommended Foods")
            for food in meal_plan["foods_to_eat"]:
                st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{food}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("##### ❌ Foods to Avoid")
            for food in meal_plan["foods_to_avoid"]:
                st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{food}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 📅 Weekly Meal Schedule")
        for day_plan in meal_plan["daily_plans"]:
            st.markdown(f'<div class="day-label">📅 Day {day_plan["day"]}</div>', unsafe_allow_html=True)
            cA, cB, cC = st.columns(3)
            with cA:
                st.markdown(f"""<div class="meal-card"><div class="meal-title">🌅 Breakfast</div>
                <div class="meal-content">{day_plan["breakfast"]}</div>
                {"<div class='snack-note'>☕ " + day_plan.get("morning_snack","") + "</div>" if day_plan.get("morning_snack") else ""}
                </div>""", unsafe_allow_html=True)
            with cB:
                st.markdown(f"""<div class="meal-card"><div class="meal-title">☀️ Lunch</div>
                <div class="meal-content">{day_plan["lunch"]}</div>
                {"<div class='snack-note'>🍎 " + day_plan.get("afternoon_snack","") + "</div>" if day_plan.get("afternoon_snack") else ""}
                </div>""", unsafe_allow_html=True)
            with cC:
                st.markdown(f"""<div class="meal-card"><div class="meal-title">🌙 Dinner</div>
                <div class="meal-content">{day_plan["dinner"]}</div>
                {"<div class='snack-note'>🌙 " + day_plan.get("evening_snack","") + "</div>" if day_plan.get("evening_snack") else ""}
                </div>""", unsafe_allow_html=True)
            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        ct, ce = st.columns(2)
        with ct:
            st.markdown("##### 💡 Nutrition Tips")
            for tip in meal_plan["general_tips"]:
                st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{tip}</div>', unsafe_allow_html=True)
        with ce:
            exercise_suggestions = diet_planner.get_exercise_suggestions(stored_diet["disease"], symptoms_for_diet)
            if exercise_suggestions:
                st.markdown("##### 🏃 Complementary Exercise")
                for s in exercise_suggestions:
                    st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{s}</div>', unsafe_allow_html=True)

        if meal_plan.get("alternative_meals"):
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
            st.markdown("##### 🍽️ Variety Options")
            ac = st.columns(3)
            alt = meal_plan["alternative_meals"]
            with ac[0]:
                st.markdown("**Breakfast Alternatives**")
                for item in alt.get("breakfast", []):
                    st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{item}</div>', unsafe_allow_html=True)
            with ac[1]:
                st.markdown("**Lunch & Dinner Alternatives**")
                for item in alt.get("lunch", [])[:3] + alt.get("dinner", [])[:3]:
                    st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{item}</div>', unsafe_allow_html=True)
            with ac[2]:
                st.markdown("**Healthy Snacks**")
                for item in alt.get("snacks", []):
                    st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-panel" style="text-align:center;padding:3rem;"><div style="font-size:2.5rem;margin-bottom:1rem;">🥗</div><div style="font-size:1rem;color:var(--text-2);">Enter a condition and click <strong>Generate Nutrition Plan</strong>.</div></div>', unsafe_allow_html=True)

    # Recipe suggestions
    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
    st.markdown("##### 👨‍🍳 Quick Recipe Ideas")
    recipe_disease = disease_input if 'disease_input' in dir() else (stored_diet["disease"] if stored_diet else "")
    rc = st.columns(3)
    with rc[0]:
        st.markdown("**Breakfast**")
        for r in ensure_list(diet_planner.get_simple_recipe("breakfast", recipe_disease)):
            st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{r}</div>', unsafe_allow_html=True)
    with rc[1]:
        st.markdown("**Lunch**")
        for r in ensure_list(diet_planner.get_simple_recipe("lunch", recipe_disease)):
            st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{r}</div>', unsafe_allow_html=True)
    with rc[2]:
        st.markdown("**Dinner**")
        for r in ensure_list(diet_planner.get_simple_recipe("dinner", recipe_disease)):
            st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{r}</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: Daily Routine
# ===========================================================================
elif page == "🗓️  Daily Routine":
    st.markdown('<p class="section-subtitle">Step 4 of 4</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Daily Routine Planner</h2>', unsafe_allow_html=True)

    lp = st.session_state.get("last_prediction", {})
    default_disease = lp.get("disease", "")
    default_age     = lp.get("age", 30)
    default_bmi     = lp.get("bmi", 22.0)
    default_temp    = lp.get("temperature", 37.0)
    if default_disease:
        st.info(f"🔬 Condition from analysis: **{default_disease}**")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        disease_input     = st.text_input("Condition / Disease", value=default_disease)
        age_input         = st.number_input("Age", min_value=1, max_value=120, value=int(default_age))
        bmi_input         = st.number_input("BMI", min_value=10.0, max_value=50.0, value=float(default_bmi))
    with col2:
        temperature_input = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=float(default_temp))
        wake_preference   = st.text_input("Preferred Wake-up Time", value="7:00 AM", placeholder="e.g. 6:30 AM")

    routine_generated = False
    if st.button("🗓️ Build Daily Routine", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a condition name.")
        else:
            routine = routine_generator.generate_routine(disease_input, age_input, bmi_input, temperature_input, wake_preference)
            st.session_state.routine_result = {
                "disease": disease_input,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "routine": routine,
            }
            routine_generated = True

    stored_routine = st.session_state.get("routine_result")
    if stored_routine:
        routine = stored_routine["routine"]
        if not routine_generated:
            st.caption(f"Showing saved routine for **{stored_routine['disease']}** · {stored_routine['generated_at']}")

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("⏰ Wake-up",    routine["wake_up"])
        c2.metric("💤 Bedtime",    routine["sleep"])
        c3.metric("💧 Daily Water", routine["hydration"])

        st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("##### 🏃 Exercise Prescription")
            st.info(routine["exercise"])
        with c5:
            st.markdown("##### 💊 Medication Schedule")
            st.warning(routine["medication_timing"])

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 😴 Rest Periods")
        rp = routine["rest_periods"]
        rc_cols = st.columns(len(rp) or 1)
        for i, r in enumerate(rp):
            with rc_cols[i % len(rc_cols)]:
                st.markdown(f'<div class="med-item"><div class="dot-teal"></div>{r}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 📋 Hour-by-Hour Schedule")
        st.dataframe(pd.DataFrame(routine["detailed_schedule"]), use_container_width=True, hide_index=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 💡 Routine Tips")
        tc = st.columns(2)
        for i, tip in enumerate(routine["tips"]):
            with tc[i % 2]:
                st.markdown(f'<div class="med-item"><div class="dot-blue"></div>{tip}</div>', unsafe_allow_html=True)

        if routine.get("variation_suggestions"):
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
            st.markdown("##### 🔄 Variation Ideas")
            for s in routine["variation_suggestions"]:
                st.markdown(f'<div class="med-item"><div class="dot-amber"></div>{s}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-panel" style="text-align:center;padding:3rem;"><div style="font-size:2.5rem;margin-bottom:1rem;">🗓️</div><div style="font-size:1rem;color:var(--text-2);">Fill in the details and click <strong>Build Daily Routine</strong>.</div></div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: Health Analytics
# ===========================================================================
elif page == "📊  Health Analytics":
    st.markdown('<h2 class="section-title">Health Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Visual overview of your health metrics from this session</p>', unsafe_allow_html=True)

    # ── Colour legend ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.5rem;
                background:#1C1C1C;border:1px solid rgba(255,255,255,.08);
                border-radius:12px;padding:1rem 1.5rem;align-items:center;">
        <span style="font-size:.78rem;font-weight:600;color:#606060;text-transform:uppercase;
                     letter-spacing:.06em;margin-right:.5rem;">Colour Guide:</span>
        <span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:#4ADE80;">
            <span style="width:12px;height:12px;background:#4ADE80;border-radius:3px;display:inline-block;"></span>Good / Normal
        </span>
        <span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:#FBBF24;">
            <span style="width:12px;height:12px;background:#FBBF24;border-radius:3px;display:inline-block;"></span>Caution / Monitor
        </span>
        <span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:#F87171;">
            <span style="width:12px;height:12px;background:#F87171;border-radius:3px;display:inline-block;"></span>Attention Needed
        </span>
        <span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:#60AEFF;">
            <span style="width:12px;height:12px;background:#60AEFF;border-radius:3px;display:inline-block;"></span>Low / Underweight
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── BMI Calculator ─────────────────────────────────────────────────────
    st.markdown("##### 📏 BMI Calculator")
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    with c1: weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, key="bmi_w")
    with c2: height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, key="bmi_h")
    with c3:
        bmi_val      = calculate_bmi(weight, height / 100)
        bmi_category = get_bmi_category(bmi_val)
        st.metric("Your BMI", f"{bmi_val:.1f}" if bmi_val else "—", bmi_category)
    with c4:
        st.markdown("""
        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
                    border-radius:8px;padding:.6rem 1rem;font-size:.8rem;line-height:1.9;margin-top:.5rem;">
            <span style="color:#60AEFF;">● &lt; 18.5</span>&nbsp; Underweight &nbsp;&nbsp;
            <span style="color:#4ADE80;">● 18.5 – 24.9</span>&nbsp; Normal<br>
            <span style="color:#FBBF24;">● 25 – 29.9</span>&nbsp; Overweight &nbsp;&nbsp;
            <span style="color:#F87171;">● ≥ 30</span>&nbsp; Obese
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)

    history         = st.session_state.health_history
    symptom_history = st.session_state.symptom_history
    last_pred       = st.session_state.get("last_prediction")

    if not history and not symptom_history and not last_pred:
        st.markdown("""
        <div class="info-panel" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:.75rem;">📊</div>
            <div style="font-size:1rem;color:var(--text-2);">
                No data yet. Complete a <strong>Symptom Analysis</strong> first
                and your health metrics will appear here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        analytics_tabs, tab_renderers, tab_descs = [], [], []

        if history:
            latest   = history[-1]
            cur_bmi  = latest.get("bmi", bmi_val)
            cur_temp = latest.get("temperature", 37.0)
            sym_cnt  = latest.get("symptom_count", 0)
            risk     = last_pred.get("risk_score", 0.5) if last_pred else 0.5
            dash_fig = create_health_metrics_dashboard(cur_bmi, cur_temp, sym_cnt, risk)
            analytics_tabs.append("📊 Snapshot")
            tab_descs.append("Your 4 key health indicators from the latest session — BMI, temperature, symptom count, and overall risk.")
            tab_renderers.append(lambda f=dash_fig: st.plotly_chart(f, use_container_width=True))

            if len(history) > 1:
                for label, desc, fn in [
                    ("📈 Trends",       "BMI and temperature plotted together so you can see how both change over time.",      lambda h=history: create_trend_analysis(h)),
                    ("📉 BMI History",  "Your BMI across sessions. The coloured bands show which weight category you fall in.", lambda h=history: create_bmi_chart(h)),
                    ("🌡️ Temp History", "Body temperature across sessions. The green zone is the normal healthy range.",        lambda h=history: create_temperature_chart(h)),
                ]:
                    fig = fn()
                    if fig:
                        analytics_tabs.append(label)
                        tab_descs.append(desc)
                        tab_renderers.append(lambda f=fig: st.plotly_chart(f, use_container_width=True))

        if symptom_history:
            sf = create_symptom_frequency_chart(symptom_history)
            if sf:
                analytics_tabs.append("🤒 Symptoms")
                tab_descs.append("Which symptoms you reported most. Taller bars = reported more often. The dotted line is the average frequency.")
                tab_renderers.append(lambda f=sf: st.plotly_chart(f, use_container_width=True))

        if last_pred:
            rf = create_disease_risk_chart([{"disease": last_pred["disease"], "risk_score": last_pred["risk_score"]}])
            if rf:
                analytics_tabs.append("🎯 Risk Profile")
                tab_descs.append("Your overall risk score for the predicted condition. Green = low, amber = moderate, red = high risk.")
                tab_renderers.append(lambda f=rf: st.plotly_chart(f, use_container_width=True))

        if analytics_tabs:
            tabs = st.tabs(analytics_tabs)
            for idx, (render, desc) in enumerate(zip(tab_renderers, tab_descs)):
                with tabs[idx]:
                    st.caption(desc)
                    render()


# ===========================================================================
# PAGE: Clinical Report
# ===========================================================================
elif page == "📋  Clinical Report":
    st.markdown('<h2 class="section-title">Clinical Report</h2>', unsafe_allow_html=True)

    if "last_prediction" not in st.session_state or not st.session_state.last_prediction:
        st.markdown('<div class="info-panel" style="text-align:center;padding:3rem;"><div style="font-size:2.5rem;margin-bottom:.75rem;">📋</div><div style="font-size:1rem;color:var(--text-2);">No prediction data. Complete a <strong>Symptom Analysis</strong> first.</div></div>', unsafe_allow_html=True)
    else:
        pred         = st.session_state.last_prediction
        patient_name = pred.get("name") or (current_user.get("full_name") or current_user["username"] if current_user else "Patient")
        ts           = pred["timestamp"]
        ts_str       = ts.strftime("%d %B %Y, %H:%M") if isinstance(ts, datetime) else str(ts)

        st.markdown(f"""
        <div class="report-header">
            <div class="rh-label">Clinical Assessment Report</div>
            <div class="rh-name">{patient_name}</div>
            <div class="rh-meta">Generated on {ts_str} &nbsp;·&nbsp; CareSync v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Age",         str(pred.get("age", "—")))
        c2.metric("Gender",      pred.get("gender", "—"))
        c3.metric("BMI",         f"{pred['bmi']:.1f}" if pred.get("bmi") else "—")
        c4.metric("Temperature", f"{pred['temperature']:.1f} °C" if pred.get("temperature") else "—")

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        risk_val   = pred.get("risk_score", 0)
        risk_class = "risk-high" if risk_val > .7 else ("risk-medium" if risk_val > .4 else "risk-low")
        risk_text  = "High Risk" if risk_val > .7 else ("Moderate Risk" if risk_val > .4 else "Low Risk")

        col_d, col_r = st.columns([2, 1])
        with col_d:
            st.markdown("##### 🔬 Primary Diagnosis")
            st.markdown(f"""
            <div class="result-banner" style="padding:1.5rem;">
                <div class="disease-name" style="font-size:1.6rem;">{pred['disease']}</div>
                <div style="display:flex;gap:1rem;margin-top:.5rem;flex-wrap:wrap;">
                    <div><span class="conf-label">Confidence</span><br><span class="conf-value" style="font-size:1.1rem;">{pred['confidence']:.1f}%</span></div>
                    <div><span class="conf-label">Risk Level</span><br><span class="{risk_class}">{risk_text}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown("##### 📊 Risk Assessment")
            if risk_val > 0.7:   st.error("⚠️ **High Risk** — Please consult a healthcare professional immediately.")
            elif risk_val > 0.4: st.warning("⚠️ **Moderate Risk** — Monitor symptoms closely.")
            else:                st.success("✅ **Low Risk** — Continue monitoring your symptoms.")

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 🤒 Reported Symptoms")
        syms = pred.get("symptoms", [])
        if syms:
            chips = "".join(f'<span class="chip">{s}</span>' for s in syms)
            st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)

        all_preds = pred.get("all_predictions", [])
        if len(all_preds) > 1:
            st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
            st.markdown("##### 🔄 Differential Diagnoses")
            for ap in all_preds[1:]:
                pct = ap["probability_percent"]
                st.markdown(f'<div class="med-item"><div class="dot-blue"></div><div style="flex:1;">{ap["disease"]}</div><div style="font-size:.8rem;font-weight:600;color:var(--text-2);">{pct:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("##### 💊 Recommended Actions")
        for icon, title, body in [
            ("💊", "Follow medicine recommendations",   "Review the Medications page for OTC, prescription, and natural remedies."),
            ("🥗", "Adhere to your nutrition plan",     "Follow the personalised 7-day meal plan on the Nutrition Plan page."),
            ("🗓️", "Implement your daily routine",      "Structured rest, hydration, and medication timing improve recovery."),
            ("📊", "Monitor your symptoms",              "Log symptoms regularly and track trends in Health Analytics."),
            ("🏥", "Consult a healthcare professional", "If symptoms persist or worsen, seek medical attention immediately."),
        ]:
            st.markdown(f'<div class="med-item" style="padding:.75rem 1rem;border-radius:10px;margin-bottom:.5rem;"><div style="font-size:1.1rem;">{icon}</div><div><strong>{title}</strong><br><span style="font-size:.82rem;color:var(--text-2);">{body}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        report_text = f"""MEDISCAN AI — CLINICAL ASSESSMENT REPORT
Generated: {ts_str}
{"="*54}
PATIENT:    {patient_name}
AGE:        {pred.get('age','—')}
GENDER:     {pred.get('gender','—')}
BMI:        {pred.get('bmi','—')}
TEMP:       {pred.get('temperature','—')} °C

DIAGNOSIS:  {pred['disease']}
CONFIDENCE: {pred['confidence']:.1f}%
RISK:       {risk_val*100:.1f}% ({risk_text})

SYMPTOMS:   {', '.join(pred.get('symptoms', [])) or 'None recorded'}

{"="*54}
DISCLAIMER: For informational purposes only.
Always consult a qualified healthcare provider.
{"="*54}
CareSync Healthcare Diagnosis & Recommendation System v1.0
"""
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"mediscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.08);margin:2rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 2rem;color:#404040;font-size:.8rem;line-height:1.8;">
    ⚠️ <strong>Medical Disclaimer</strong> — CareSync is for informational and educational purposes only.
    Always consult a qualified healthcare professional.<br>
    CareSync Healthcare Diagnosis &amp; Recommendation System &nbsp;·&nbsp; v1.0 &nbsp;·&nbsp; Built with ❤️ for better health awareness
</div>
""", unsafe_allow_html=True)
