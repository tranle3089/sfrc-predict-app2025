import streamlit as st
import joblib
import section_overview
import section_database
import section_models
import section_predict

# ---------- Custom CSS for Expander Heading Colors ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    background: #f5f7fa !important;
}
.block-container {
    background: #f5f7fa !important;
    box-shadow: 0 0 20px 0 rgba(45, 76, 151, 0.05), 0 1.5px 5px 0 rgba(160,180,215,0.03);
    border-radius: 18px !important;
    padding-top: 2.2rem !important;
    padding-bottom: 1.2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}

/* -------- Kết quả prediction card đẹp -------- */
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
}
.result-card:hover {
    box-shadow: 0 6px 30px 0 rgba(70,109,236,0.13);
    transform: translateY(-2px) scale(1.04);
}
.result-label {
    color: #2563eb;
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 1.0rem;
    letter-spacing: 0.5px;
}
.result-value {
    color: #18181b;
    font-size: 2.1rem;
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
@media (max-width: 900px) {
    .result-card-row {
        flex-direction: column;
        gap: 1.0rem;
    }
    .result-card {
        min-width: unset;
    }
}

/* -------- Heading expander cực to & nhiều màu -------- */
div[data-testid="stExpander"] summary span {
    font-size: 2.15rem !important;    
    font-weight: 900 !important;
    color: inherit !important;
}
div[data-testid="stExpander"]:nth-of-type(1) > details > summary {
    background: linear-gradient(90deg, #e3eaff 0%, #f6f8fe 100%) !important;
    color: #2563eb !important;
    border-bottom: 1.5px solid #e3eaff;
}
div[data-testid="stExpander"]:nth-of-type(2) > details > summary {
    background: linear-gradient(90deg, #e9fae9 0%, #f7fef6 100%) !important;
    color: #107143 !important;
    border-bottom: 1.5px solid #e9fae9;
}
div[data-testid="stExpander"]:nth-of-type(3) > details > summary {
    background: linear-gradient(90deg, #fff2e6 0%, #fdf7f2 100%) !important;
    color: #c26622 !important;
    border-bottom: 1.5px solid #fff2e6;
}
div[data-testid="stExpander"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-bottom: 1.1rem;
    box-shadow: 0 2px 8px 0 rgba(31,38,135,0.06);
    border: 1.2px solid #e5e7eb !important;
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
st.info("Contact: huyenle3089@gmail.com")
