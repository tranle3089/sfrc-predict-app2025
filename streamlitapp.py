import streamlit as st
import joblib
import section_overview
import section_database
import section_models
import section_predict

# ---------- Custom CSS for Expander Heading Colors ----------

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
