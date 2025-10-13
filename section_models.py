import streamlit as st

def show():
    st.markdown("""
        <style>
        /* ===== Tiêu đề ===== */
        .ml-title {
            font-family: 'Times New Roman', serif;
            font-size: 22px;
            font-weight: bold;
            color: #000;
            text-align: center;
            margin: 0 0 18px 0;
        }

        /* ===== Nội dung danh sách ===== */
        .ml-list {
            font-family: 'Times New Roman', serif;
            font-size: 18px;
            color: #000;
            line-height: 1.8;
            background: #fff;
            padding-left: 25px;
        }

        /* Bỏ dấu chấm đầu dòng */
        .ml-list ul {
            margin: 0 0 16px 0;
            list-style-type: none;
        }

        /* Khoảng cách từng dòng */
        .ml-list li {
            margin: 4px 0;
        }

        /* Thụt sâu cho mô hình con */
        .ml-list li ul {
            margin-left: 45px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị tiêu đề và danh sách
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
