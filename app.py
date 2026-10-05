"""
Titanic Survival Predictor — Streamlit App
Ocean-themed UI with a trained Logistic Regression model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- OCEAN THEME CSS ----------------
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(180deg, #0a1f44 0%, #0f2e5e 50%, #061530 100%);
    }

    /* Title */
    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        text-align: center;
        font-size: 2.6rem !important;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Ship decoration */
    .ship {
        text-align: center;
        font-size: 4rem;
        opacity: 0.6;
        margin: 0.5rem 0;
        animation: float 5s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(-2deg); }
        50%      { transform: translateY(-12px) rotate(2deg); }
    }

    /* Card containers */
    div[data-testid="stForm"] {
        background: rgba(15, 40, 80, 0.55);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 24px;
        backdrop-filter: blur(16px);
    }

    /* Input labels */
    label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Input fields */
    input, select, .stSelectbox, .stNumberInput {
        background: rgba(10, 31, 68, 0.7) !important;
        border-radius: 10px !important;
        color: #e6f0ff !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        padding: 14px;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.4);
    }

    /* Result card */
    .result-card {
        background: rgba(15, 40, 80, 0.55);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        backdrop-filter: blur(16px);
        margin-top: 1rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-title.survived { color: #10b981; }
    .result-title.deceased { color: #ef4444; }

    .result-prob {
        font-size: 3.5rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    """Load the trained model. Trains fresh if model.pkl missing."""
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    elif os.path.exists("Backend/model.pkl"):
        with open("Backend/model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        # Train on the fly if no pkl exists
        from sklearn.linear_model import LogisticRegression
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy()
        y = df['Survived']
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model


model = load_model()


# ---------------- HEADER ----------------
st.markdown('<div class="ship">🚢</div>', unsafe_allow_html=True)
st.markdown("# Titanic Survival Predictor")
st.markdown(
    '<p class="subtitle">Enter passenger details to estimate survival probability — '
    'powered by a trained Logistic Regression model.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1], gap="large")

# --- LEFT: Form ---
with col1:
    st.markdown("### 🎫 Passenger Details")

    with st.form("predictor_form"):
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1st Class (Upper)",
                2: "2nd Class (Middle)",
                3: "3rd Class (Lower)"
            }[x],
        )

        sex = st.selectbox("Gender", options=["female", "male"])

        age = st.number_input(
            "Age", min_value=0, max_value=100, value=29, step=1,
        )

        col_sib, col_par = st.columns(2)
        with col_sib:
            sibsp = st.number_input("Siblings / Spouses", min_value=0,
                                    max_value=10, value=0, step=1)
        with col_par:
            parch = st.number_input("Parents / Children", min_value=0,
                                    max_value=10, value=0, step=1)

        submitted = st.form_submit_button("🔮  Predict Survival")

# --- RIGHT: Result ---
with col2:
    st.markdown("### 📊 Prediction Result")

    if not submitted:
        st.markdown("""
        <div class="result-card">
            <div style="font-size: 4rem; opacity: 0.6;">🧭</div>
            <h3 style="color: #94a3b8;">Awaiting Input</h3>
            <p style="color: #64748b;">
                Fill in the passenger details and click
                <strong>Predict Survival</strong> to see the result.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build features in correct order
        sex_val = 1 if sex == "male" else 0
        features = np.array([[pclass, sex_val, age, sibsp, parch]])

        # Predict
        prob = model.predict_proba(features)[0][1]
        pred = int(model.predict(features)[0])
        prob_pct = round(float(prob) * 100, 2)

        # Result card
        if pred == 1:
            title = "Survived 🛟"
            title_class = "survived"
        else:
            title = "Deceased 🌊"
            title_class = "deceased"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title {title_class}">{title}</div>
            <p style="color: #94a3b8; text-transform: uppercase;
                      letter-spacing: 1.5px; font-size: 0.8rem;">
                Survival Probability
            </p>
            <div class="result-prob">{prob_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress gauge
        st.progress(prob)

        # Meta info
        st.markdown("---")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown("**Model**")
            st.caption("Logistic Regression")
            st.markdown("**Accuracy**")
            st.caption("~81%")
        with meta_col2:
            st.markdown("**Features**")
            st.caption("5 (Pclass, Sex, Age, SibSp, Parch)")
            st.markdown("**Dataset**")
            st.caption("Titanic (891 rows)")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    © 2026 Krithi K M · Built with Streamlit + Scikit-learn
</div>
""", unsafe_allow_html=True)