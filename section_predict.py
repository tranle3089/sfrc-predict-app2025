import streamlit as st
import numpy as np
import time

spinner_html = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:130px">
  <div class="lds-spinner">
    <div></div><div></div><div></div><div></div><div></div><div></div>
    <div></div><div></div><div></div><div></div><div></div><div></div>
  </div>
  <div style="margin-top:18px;font-weight:700;color:#c2410c;font-size:1.1rem;">Processing...</div>
</div>
<style>
.lds-spinner {
  color: official;
  display: inline-block;
  position: relative;
  width: 80px;
  height: 80px;
}
.lds-spinner div {
  transform-origin: 40px 40px;
  animation: lds-spinner 1.2s linear infinite;
  position: absolute;
}
.lds-spinner div:after {
  content: " ";
  display: block;
  position: absolute;
  top: 3px; left: 37px;
  width: 6px; height: 18px;
  border-radius: 20%;
  background: #c2410c;
}
.lds-spinner div:nth-child(1) { transform: rotate(0deg); animation-delay: -1.1s; }
.lds-spinner div:nth-child(2) { transform: rotate(30deg); animation-delay: -1s; }
.lds-spinner div:nth-child(3) { transform: rotate(60deg); animation-delay: -0.9s; }
.lds-spinner div:nth-child(4) { transform: rotate(90deg); animation-delay: -0.8s; }
.lds-spinner div:nth-child(5) { transform: rotate(120deg); animation-delay: -0.7s; }
.lds-spinner div:nth-child(6) { transform: rotate(150deg); animation-delay: -0.6s; }
.lds-spinner div:nth-child(7) { transform: rotate(180deg); animation-delay: -0.5s; }
.lds-spinner div:nth-child(8) { transform: rotate(210deg); animation-delay: -0.4s; }
.lds-spinner div:nth-child(9) { transform: rotate(240deg); animation-delay: -0.3s; }
.lds-spinner div:nth-child(10) { transform: rotate(270deg); animation-delay: -0.2s; }
.lds-spinner div:nth-child(11) { transform: rotate(300deg); animation-delay: -0.1s; }
.lds-spinner div:nth-child(12) { transform: rotate(330deg); animation-delay: 0s; }
@keyframes lds-spinner {
  0% { opacity: 1; }
  100% { opacity: 0; }
}
</style>
"""


def show(model_CS, model_ST, model_FC):
    # Không load lại model ở đây!
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
        loading_placeholder = st.empty()
        loading_placeholder.markdown(spinner_html, unsafe_allow_html=True)
        time.sleep(2)
        loading_placeholder.empty()

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

