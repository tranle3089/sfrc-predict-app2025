import streamlit as st
import numpy as np

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
