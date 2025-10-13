import streamlit as st

def show():
    st.markdown("""
        <style>
        .ml-title {
            font-family: 'Times New Roman', serif;
            font-size: 22px;
            font-weight: bold;
            color: #000;
            margin: 0 0 14px 0;
        }
        .ml-list {
            font-family: 'Times New Roman', serif;
            font-size: 16px;
            color: #000;
            line-height: 1.6;
            background: #fff;
            padding-left: 22px;  /* thụt lề */
        }
        .ml-list ul {
            margin: 0 0 14px 0;
            list-style-type: none;  /* loại bỏ dấu chấm đầu dòng */
        }
        .ml-list li { margin: 2px 0; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ml-title'>Machine Learning Model Selection</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="ml-list">
      <ul>
        <li>Regression based algorithms
          <ul>
            <li>Adaboost</li>
            <li>Catboost</li>
            <li>XGboost</li>
          </ul>
        </li>
      </ul>

      <ul>
        <li>Tree based algorithms
          <ul>
            <li>Extra tree</li>
            <li>Random forest</li>
          </ul>
        </li>
      </ul>

      <ul>
        <li>Neural based algorithm
          <ul>
            <li>Artificial Neural Network</li>
          </ul>
        </li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
