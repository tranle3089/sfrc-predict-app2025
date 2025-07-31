import streamlit as st
import joblib
import section_overview
import section_database
import section_models
import section_predict

# ---------- Custom CSS for Expander Heading Colors ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 1.0rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    background: #f5f7fa;
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
    border: 1.5px solid #3b82f6;
    transition: box-shadow 0.2s, transform 0.2s;
}
.result-card:hover {
    box-shadow: 0 6px 30px 0 rgba(70,109,236,0.13);
    transform: translateY(-2px) scale(1.04);
}
.result-label {
    color: #2563eb;
    font-size: 1.0rem;
    font-weight: 800;
    margin-bottom: 1.0rem;
    letter-spacing: 0.5px;
}
.result-value {
    color: #18181b;
    font-size: 2.0rem;
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
    font-size: 2.35rem !important;    /* cực to, ~37px */
    font-weight: 900 !important;
    color: inherit !important;
}
div[data-testid="stExpander"]:nth-of-type(1) > details > summary {
    background: linear-gradient(90deg, #e3eaff 0%, #f6f8fe 100%) !important;
    color: #2563eb !important;
}
div[data-testid="stExpander"]:nth-of-type(2) > details > summary {
    background: linear-gradient(90deg, #e9fae9 0%, #f7fef6 100%) !important;
    color: #107143 !important;
}
div[data-testid="stExpander"]:nth-of-type(3) > details > summary {
    background: linear-gradient(90deg, #fff2e6 0%, #fdf7f2 100%) !important;
    color: #c26622 !important;
}
div[data-testid="stExpander"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------- Main Content -----------------

st.title("SFRC Strength Prediction App")
st.write("""
#### *A novel hybrid machine learning framework for predicting the mechanical properties of steel fiber-reinforced concrete*
""")
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
