import streamlit as st

def show():
    diagram = """
Machine Learning Model Selection

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

    st.markdown(f"```text\n{diagram}\n```")
