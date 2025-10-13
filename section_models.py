import streamlit as st

def show():
    # Nội dung hiển thị dạng chữ
    diagram = """
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

    # CSS định dạng Times New Roman, tiêu đề lớn, nội dung nhỏ hơn
    st.markdown("""
        <style>
        .heading {
            font-family: 'Times New Roman', serif;
            font-size: 22px;
            font-weight: bold;
            color: black;
            background-color: white;
            margin-bottom: 15px;
        }
        .text-content {
            font-family: 'Times New Roman', serif;
            font-size: 16px;
            color: black;
            background-color: white;
            line-height: 1.6;
            padding-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị tiêu đề và nội dung
    st.markdown("<div class='heading'>Machine Learning Model Selection</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='text-content'>{diagram}</div>", unsafe_allow_html=True)
