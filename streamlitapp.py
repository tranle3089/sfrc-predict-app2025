import streamlit as st
import joblib
import section_overview
import section_database
import section_models
import section_predict

# ---------- Custom CSS for Expander Heading Colors ----------
st.markdown("""
<style>
html, body, .block-container {
    background: #d8dee8 !important;
    color: #18181b !important;
}
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #18181b !important;
}
/* ---- Label ---- */
label, .stNumberInput label, .stSelectbox label, 
div[data-testid="stNumberInputLabel"], div[data-testid="stSelectboxLabel"] {
    color: #18181b !important;
    font-size: 1.06rem !important;    
    font-weight: 600 !important;
}
/* ---- Result card ---- */
.result-card-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    justify-content: center;
}
.result-card {
    background: #fff;
    border-radius: 17px;
    padding: 1.0rem 1.1rem 0.7rem 1.1rem;
    box-shadow: 0 2px 10px 0 rgba(70, 109, 236, 0.06), 0 1px 2px 0 rgba(0,0,0,0.02);
    min-width: 150px;
    max-width: 180px;
    text-align: center;
    border: 2px solid #2563eb;
    transition: box-shadow 0.2s, transform 0.2s;
    color: #18181b !important;
}
.result-card:hover {
    box-shadow: 0 6px 30px 0 rgba(70,109,236,0.13);
    transform: translateY(-2px) scale(1.04);
}
.result-label {
    color: #2563eb;
    font-size: 1.09rem;
    font-weight: 800;
    margin-bottom: 1.0rem;
    letter-spacing: 0.5px;
}
.result-value {
    color: #18181b;
    font-size: 2.05rem;
    font-weight: 900;
    margin-bottom: 0.2rem;
    letter-spacing: -2px;
}
.unit-label {
    color: #b3b5ba;
    font-size: 1.0rem;
    margin-top: 0.4rem;
    font-weight: 600;
    letter-spacing: 1px;
}
/* ---- Button ---- */
.stButton>button {
    font-size: 1.13rem !important; 
    color: #fff !important;
    background: #0e54cc !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.0rem !important;
    font-weight: 800 !important;
    transition: background 0.2s;
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}
.stButton>button:hover {
    background: #1d4ed8 !important;
}
/* -------- Expander section: chữ vừa phải, màu box đậm hơn -------- */
div[data-testid="stExpander"] summary span {
    font-size: 1.45rem !important;     
    font-weight: 800 !important;
    color: inherit !important;
    letter-spacing: 0.4px;
}
/* Section 1 */
div[data-testid="stExpander"]:nth-of-type(1) > details > summary {
    background: linear-gradient(90deg, #74aaff 0%, #cce3fd 100%) !important;
    color: #011c4f !important;
    border-bottom: 2.5px solid #2563eb;
    box-shadow: 0 2px 16px 0 #f5f1b8;
    font-size: 1.35rem !important;
}
/* Section 2 */
div[data-testid="stExpander"]:nth-of-type(2) > details > summary {
    background: linear-gradient(90deg, #71d8af 0%, #d8f6e6 100%) !important;
    color: #011c4f !important;
    border-bottom: 2.5px solid #107143;
    box-shadow: 0 2px 16px 0 #f5f1b8;
}
/* Section 3 */
div[data-testid="stExpander"]:nth-of-type(3) > details > summary {
    background: linear-gradient(90deg, #f7e997 0%, #f7e997 100%) !important;
    color: #011c4f !important;
    border-bottom: 2.5px solid #fb923c;
    box-shadow: 0 2px 16px 0 #f5f1b8;
}
/* Section 4 */
div[data-testid="stExpander"]:nth-of-type(4) > details > summary {
    background: linear-gradient(90deg, #b0aaff 0%, #ede9fe 100%) !important;
    color: #011c4f !important;
    border-bottom: 2.5px solid #6366f1;
    box-shadow: 0 2px 16px 0 #f5f1b8;
}
div[data-testid="stExpander"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px 0 rgba(31,38,135,0.06);
    border: 2px solid #e5e7eb !important;
}
@media (max-width: 900px) {
    .result-card-row {
        flex-direction: column;
        gap: 1.0rem;
    }
    .result-card {
        min-width: unset;
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

