import streamlit as st

def show():
    # Nội dung sơ đồ dạng chữ
    diagram = """
**Machine Learning Model Selection**

- Regression based algorithms
    - Adaboost
    - Catboost
    - XGboost

- Tree based algorithms
    - Extra tree
    - Random forest

- Neural based algorithm
    - Artificial Neural Network
"""

    # CSS định dạng hiển thị
    st.markdown("""
        <style>
        .diagram-box {
            font-family: 'Times New Roman', serif;
            font-size: 16px;
            color: black;
            background-color: white;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #cccccc;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        strong, b {
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị sơ đồ trong Streamlit
    st.markdown(f"<div class='diagram-box'>{diagram}</div>", unsafe_allow_html=True)
