import streamlit as st
import numpy as np
import time
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 0.1rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        .result-card, .result-label, .result-value, .unit-label,
        .result-card-row, .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: ##101112 !important;
        }
        html, body, [class*="css"]  {
            font-size: 20px !important;
        }
        h1 {
            font-size: 34px !important;
            font-weight: 800 !important;
            margin-top: 0.0rem !important;
            margin-bottom: 0.0rem !important;
        }
        h2, h3, h4 {
            font-size: 24px !important;
            font-weight: 800 !important;
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            border-left: 4px solid #2563eb;
            padding-left: 2.1rem;
            background: linear-gradient(90deg, #eff6ff 50%, transparent 80%);
        }
        label, .stNumberInput label, .stSelectbox label {
            font-size: 20px !important;
            font-weight: 500;
            color: #18181b!important;
            margin-bottom: 0.1rem !important;
        }
        .stNumberInput, .stSelectbox {
            margin-bottom: 0.1rem !important;
        }
        .stButton>button {
            font-size: 22px !important;
            color: #fff !important;
            background: #021e5c !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.0rem !important;
            font-weight: 800 !important;
            transition: background 0.3s;
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        .stButton>button:hover {
            background: #1d4ed8 !important;
        }
        .stMarkdown p {
            font-size: 20px !important;
            margin-bottom: 0.1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
def show(model_CS, model_ST, model_FC):
    st.markdown("---")
    st.markdown("### Concrete Matrix")
    col1, col2, col3 = st.columns(3)
    with col1:
        W = st.number_input("Water (kg)", min_value=123.0, max_value=398.0, value=181.3)
        C = st.number_input("Cement (kg)", min_value=258.0, max_value=651.0, value=396.8)
    with col2:
        S = st.number_input("Sand (kg)", min_value=408.0, max_value=1225.0, value=800.4)
        CA = st.number_input("Coarse Aggregate (kg)", min_value=356.0, max_value=1594.0, value=1066.0)
    with col3:
        SP = st.number_input("Admixture (kg)", min_value=0.0, max_value=18.4, value=3.3)
        smax = st.number_input("Max Aggregate Size (mm)", min_value=10.0, max_value=40.0, value=19.2)

    st.markdown("### Steel Fiber Components")
    col4, col5 = st.columns(2)
    with col4:
        Vf = st.number_input("Fiber Volume Content (%)", min_value=0.0, max_value=2.0, value=1.0)
        df = st.number_input("Fiber Diameter (mm)", min_value=0.1, max_value=1.2, value=0.8)
    with col5:
        Lf = st.number_input("Fiber Length (mm)", min_value=25.0, max_value=62.0, value=44.1)
        pf = st.selectbox(
            "Fiber Shape",
            options=[1.0, 0.75, 0.5],
            format_func=lambda x: {
                1.0: "Hooked",
                0.75: "Crimped / Straight",
                0.5: "Smooth / Mill-cut"
            }.get(x)
        )

    st.markdown("---")
    input_data = np.array([[W, C, S, CA, smax, SP, pf, Vf, df, Lf]])

    if st.button("🔍 Predict"):
        # BỎ spinner và time.sleep hoàn toàn
        cs_pred = model_CS.predict(input_data)[0]
        st_pred = model_ST.predict(input_data)[0]
        fc_pred = model_FC.predict(input_data)[0]

        st.markdown("""
        <div style="background-color:#e6f9eb; border-radius: 13px; padding: 1.0rem 1.0rem; margin-bottom:1rem; color:#20723a; font-weight: 700; font-size: 1.1rem; border: 1.0px solid #74c99e;">
            ✅ Results
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-card-row" style="flex-direction:column;gap:1.0rem;">
                <div class="result-card" style="width:100%;max-width:unset;margin:0;">
                    <div class="result-label">Compressive Strength (CS)</div>
                    <div class="result-value">{cs_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
                <div class="result-card" style="width:100%;max-width:unset;margin:0;">
                    <div class="result-label">Tensile Strength (ST)</div>
                    <div class="result-value">{st_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
                <div class="result-card" style="width:100%;max-width:unset;margin:0;">
                    <div class="result-label">Flexural Strength (FC)</div>
                    <div class="result-value">{fc_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


