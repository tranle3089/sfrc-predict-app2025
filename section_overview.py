import streamlit as st

def show():
    # Tiêu đề
    st.markdown("""
    <div style='text-align:center; margin-bottom:0em;'>
        <span style='font-size:1.30rem; font-weight:700; color:#18181b;'>Welcome to the <u>SFRC Strength Prediction App</u></span>
    </div>
    """, unsafe_allow_html=True)

    # Đoạn mô tả framework
    st.markdown("""
    <div style='margin-top:0.5em; font-size:1.09rem; text-align:justify; color:#18181b;'>
    <b>Figure 1</b> presents the research framework for predicting the mechanical properties of SFRC, which is divided into three main stages: <b>data collection</b>, <b>data processing</b>, and <b>prediction models development</b>. The following sections provide a detailed description of the respective stages.
    </div>
    """, unsafe_allow_html=True)

    # Hình workflow
    st.image("Picture1.png",
             caption="Figure 1. Workflow of the proposed hybrid ML framework for SFRC prediction.",
             use_container_width=True)

    # Table of Contents
    st.markdown("""
    <div style="margin-top:1.5em;">
    <span style="font-size:1.14rem; font-weight:700; color:#18181b;">Table of Contents</span>
    </div>
    <div style="font-size:1.07rem; line-height:1.7; color:#18181b;">
    <ol style="list-style-type: decimal; margin-left: 0.7em;">
      <li><b>Introduction</b></li>
      <li><b>Materials and Methods</b>
        <ul style="list-style-type:none; margin-left: 0;">
          <li>2.1 Data Collection</li>
          <li>2.2 Data Preprocessing</li>
          <li>2.3 Machine Learning Model Selection
            <ul style="list-style-type:none;">
              <li>2.3.1 Regression Algorithms</li>
              <li>2.3.2 Tree-Based Algorithms</li>
              <li>2.3.3 Neural Network Algorithms</li>
            </ul>
          </li>
          <li>2.4 Optimization Using TPE Method
            <ul style="list-style-type:none;">
              <li>2.4.1 Bayesian Optimization</li>
              <li>2.4.2 Tree-Structured Parzen Estimator (TPE)</li>
            </ul>
          </li>
          <li>2.5 Predictive Model Development
            <ul style="list-style-type:none;">
              <li>2.5.1 Training and Hyperparameter Tuning</li>
              <li>2.5.2 Performance Evaluation Metrics</li>
            </ul>
          </li>
        </ul>
      </li>
      <li><b>Results and Discussion</b>
        <ul style="list-style-type:none;">
          <li>3.1 Optimized Hyperparameters</li>
          <li>3.2 Performance of the Hybrid Machine Learning Model</li>
          <li>3.3 SHAP (Shapley Additive Explanations) Analysis</li>
          <li>3.4 Partial Dependence Plot (PDP) Analysis</li>
        </ul>
      </li>
      <li><b>Conclusion</b></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
