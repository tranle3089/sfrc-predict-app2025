import streamlit as st
import numpy as np

st.markdown("""
    <style>
        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 1.0rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
            background: #f5f7fa;
        }
        html, body, [class*="css"]  {
            font-size: 20px !important;
            color: #191a23 !important;
        }
        h1 {
            font-size: 34px !important;
            font-weight: 800 !important;
            margin-top: 0.0rem !important;
            margin-bottom: 0.5rem !important;
            color: #22223b !important;
        }
        h2, h3, h4 {
            font-size: 24px !important;
            font-weight: 800 !important;
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            border-left: 4px solid #2563eb;
            padding-left: 1.1rem;
            color: #171821 !important;
            background: linear-gradient(90deg, #e9efff 50%, transparent 100%);
        }
        label, .stNumberInput label, .stSelectbox label {
            font-size: 20px !important;
            font-weight: 600;
            margin-bottom: 0.1rem !important;
            color: #2c2d35 !important;
        }
        .stNumberInput, .stSelectbox {
            margin-bottom: 0.4rem !important;
            background: #fff !important;
            border-radius: 11px !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 2px 12px 0 rgba(31,38,135,0.06);
        }
        .stButton>button {
            font-size: 22px !important;
            color: #fff !important;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            border: none !important;
            border-radius: 7px !important;
            padding: 0.6rem 2.1rem !important;
            font-weight: 800 !important;
            transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 2px 16px 0 rgba(31, 38, 135, 0.07);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
            box-shadow: 0 6px 24px 0 rgba(70,109,236,0.16);
            transform: translateY(-2px) scale(1.04);
        }
        .stMarkdown p {
            font-size: 20px !important;
            margin-bottom: 0.1rem !important;
            color: #68758f !important;
        }
        /* ----------- Kết quả card đẹp ----------- */
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

    if st.button("🔍 Predict All"):
        cs_pred = model_CS.predict(input_data)[0]
        st_pred = model_ST.predict(input_data)[0]
        fc_pred = model_FC.predict(input_data)[0]
    
        st.markdown("""
        <div style="background-color:#e6f9eb; border-radius: 13px; padding: 1.0rem 1.5rem; margin-bottom:1rem; color:#20723a; font-weight: 700; font-size: 1.1rem; border: 1.5px solid #74c99e;">
            ✅ Prediction Results
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f'''
            <div class="result-card-row">
                <div class="result-card">
                    <div class="result-label">Compressive Strength (CS)</div>
                    <div class="result-value">{cs_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Tensile Strength (ST)</div>
                    <div class="result-value">{st_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Flexural Strength (FC)</div>
                    <div class="result-value">{fc_pred:.2f}</div>
                    <div class="unit-label">MPa</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
