import streamlit as st
import joblib
import section_overview
import section_database
import section_models
import section_predict

# ---------- Custom CSS for Expander Heading Colors ----------
st.markdown("""
<style>
/* ---- Main app background and text ---- */
html, body, .block-container {
    background: #e6eaf3 !important;
    color: #18181b !important;
}

/* ---- Headings ---- */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #18181b !important;
}

/* ---- Input labels ---- */
label, .stNumberInput label, .stSelectbox label, 
div[data-testid="stNumberInputLabel"], div[data-testid="stSelectboxLabel"] {
    color: #18181b !important;
    font-size: 1.11rem !important;    
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}

/* ---- Result cards ---- */
.result-card-row {
    display: flex;
    gap: 1.6rem;
    margin-top: 1.8rem;
    margin-bottom: 1.8rem;
    justify-content: center;
}
.result-card {
    background: #fff;
    border-radius: 17px;
    padding: 1.25rem 1.3rem 1.05rem 1.3rem;
    box-shadow: 0 4px 24px 0 rgba(70, 109, 236, 0.10), 0 2px 4px 0 rgba(0,0,0,0.04);
    min-width: 180px;
    max-width: 230px;
    text-align: center;
    border: 2.5px solid #2563eb;
    transition: box-shadow 0.2s, transform 0.2s;
    color: #18181b !important;
}
.result-card:hover {
    box-shadow: 0 6px 36px 0 rgba(70,109,236,0.16);
    transform: translateY(-2px) scale(1.03);
}
.result-label {
    color: #2563eb;
    font-size: 1.22rem;
    font-weight: 800;
    margin-bottom: 1.15rem;
    letter-spacing: 0.7px;
}
.result-value {
    color: #18181b;
    font-size: 2.3rem;
    font-weight: 900;
    margin-bottom: 0.17rem;
    letter-spacing: -1px;
}
.unit-label {
    color: #6d768a;
    font-size: 1.1rem;
    margin-top: 0.3rem;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ---- Button style ---- */
.stButton>button {
    font-size: 1.12rem !important;
    color: #fff !important;
    background: linear-gradient(90deg, #2563eb 0%, #4080fa 100%) !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.65rem 2.0rem !important;
    font-weight: 800 !important;
    transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
    margin-top: 0.5rem !important;
    margin-bottom: 0.4rem !important;
    box-shadow: 0 2px 16px 0 rgba(70,109,236,0.09);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
    box-shadow: 0 6px 20px 0 rgba(70,109,236,0.17);
    transform: translateY(-1.5px) scale(1.03);
}

/* ---- Expander headings: bigger, bold, more contrast ---- */
div[data-testid="stExpander"] summary span {
    font-size: 1.5rem !important;     
    font-weight: 800 !important;
    color: #101125 !important;
    letter-spacing: 0.4px;
}
div[data-testid="stExpander"]:nth-of-type(1) > details > summary {
    background: linear-gradient(90deg, #4f88fd 0%, #d0e7ff 100%) !important;
    color: #08134b !important;
    border-bottom: 3px solid #2563eb;
    box-shadow: 0 4px 20px 0 #c2d2ff;
}
div[data-testid="stExpander"]:nth-of-type(2) > details > summary {
    background: linear-gradient(90deg, #21bc96 0%, #e0f4eb 100%) !important;
    color: #083726 !important;
    border-bottom: 3px solid #11ad75;
    box-shadow: 0 4px 20px 0 #d2ffe8;
}
div[data-testid="stExpander"]:nth-of-type(3) > details > summary {
    background: linear-gradient(90deg, #ffe066 0%, #fff6c7 100%) !important;
    color: #866000 !important;
    border-bottom: 3px solid #fdc600;
    box-shadow: 0 4px 20px 0 #fff9c2;
}
div[data-testid="stExpander"]:nth-of-type(4) > details > summary {
    background: linear-gradient(90deg, #a390fd 0%, #f4edff 100%) !important;
    color: #1a1165 !important;
    border-bottom: 3px solid #6366f1;
    box-shadow: 0 4px 20px 0 #e0d7ff;
}
div[data-testid="stExpander"] {
    border-radius: 17px !important;
    overflow: hidden !important;
    margin-bottom: 1.3rem;
    box-shadow: 0 2px 14px 0 rgba(31,38,135,0.07);
    border: 2px solid #e2e6f0 !important;
}

/* ---- Mobile responsive ---- */
@media (max-width: 900px) {
    .result-card-row {
        flex-direction: column;
        gap: 1.1rem;
    }
    .result-card {
        min-width: unset;
        max-width: 97vw;
    }
}
</style>
""", unsafe_allow_html=True)


# ----------------- Main Content -----------------

st.markdown(
    """
    <div style="font-size:2.2rem; font-weight:800; color:#18181b; margin-top:1em; margin-bottom:0.25em; letter-spacing:-1px;">
        SFRC Strength Prediction App
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:1.15rem; color:#18181b; margin-bottom:1.0em;">
        <em>A novel hybrid machine learning framework for predicting the mechanical properties of steel fiber-reinforced concrete</em>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model chỉ 1 lần ở đây:
model_CS = joblib.load("CatBoost_optimized_CS.pkl")
model_ST = joblib.load("CatBoost_optimized_ST.pkl")
model_FC = joblib.load("CatBoost_optimized_FC.pkl")


with st.expander("1. Overview / Introduction"):
    section_overview.show()

with st.expander("2. Database"):
    section_database.show()

with st.expander("3. Machine Learning Models"):
    section_models.show()

with st.expander("4. Prediction Tool"):
    section_predict.show(model_CS, model_ST, model_FC)

st.markdown("---")
st.markdown("""
<div style="
    background: #d8dee8;
    border-left: 5px solid #d8dee8;
    border-radius: 8px;
    padding: 0.95rem 1.3rem;
    font-size: 1.09rem;
    color: #18181b;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    font-weight: 600;">
    📧 Contact: <a href="mailto:huyenle3089@gmail.com" style="color:#18181b; text-decoration:underline; font-weight:600;">huyenle3089@gmail.com</a>
</div>
""", unsafe_allow_html=True)

