import streamlit as st

def show():
    st.markdown("## 4. Database")

    st.markdown(
        """
        <div style='text-align:justify; font-size:1.07rem; color:#25344b; line-height:1.5;'>
        The database presented in Table 1 provides a comprehensive overview of the experimental datasets
        employed in this study. It encompasses key mixture proportions, steel fiber characteristics, and corresponding mechanical
        properties of SFRC specimens, which serve as the foundation for the machine learning model development.
        The detailed organization of the data ensures transparency and reproducibility, facilitating robust training, validation, and benchmarking of predictive algorithms.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Caption above the image
    st.markdown(
        "<div style='font-size:1.08rem; font-weight:700; color:#374151; margin-bottom:1rem;'>"
        "Table 1. Database used for SFRC machine learning modeling"
        "</div>",
        unsafe_allow_html=True
    )
    st.image("Picture2.png", use_container_width=True)
