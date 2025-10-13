import streamlit as st

def show():
    st.markdown("""
    <ul style="list-style-type:none; margin-left:0; padding-left:0;">
      <li><b>Materials and Methods</b>
        <ul style="list-style-type:none; margin-left:0; padding-left:15px;">
          <li>2.1 Data Collection</li>
          <li>2.2 Data Preprocessing</li>
          <li>2.3 Machine Learning Model Selection
            <ul style="list-style-type:none; margin-left:0; padding-left:25px;">
              <li>2.3.1 Regression Algorithms</li>
              <li>2.3.2 Tree-Based Algorithms</li>
              <li>2.3.3 Neural Network Algorithms</li>
            </ul>
          </li>
          <li>2.4 Optimization Using TPE Method
            <ul style="list-style-type:none; margin-left:0; padding-left:25px;">
              <li>2.4.1 Bayesian Optimization</li>
              <li>2.4.2 Tree-Structured Parzen Estimator (TPE)</li>
            </ul>
          </li>
          <li>2.5 Predictive Model Development
            <ul style="list-style-type:none; margin-left:0; padding-left:25px;">
              <li>2.5.1 Training and Hyperparameter Tuning</li>
              <li>2.5.2 Performance Evaluation Metrics</li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    """, unsafe_allow_html=True)
