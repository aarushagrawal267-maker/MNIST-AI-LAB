"""PURF — handwritten digit lab.

Launch with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import LAB_NAME
from src.ui import common, theme

st.set_page_config(
    page_title=f"{LAB_NAME} · Handwritten Digit Lab",
    page_icon="assets/icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()
theme.register_plotly_template()

P = "pages/"
pages = {
    "Overview": [
        st.Page(P + "0_Home.py", title="Home Dashboard", icon=":material/space_dashboard:", default=True),
    ],
    "Playground": [
        st.Page(P + "9_Draw_Your_Digit.py", title="Draw Your Digit", icon=":material/draw:"),
        st.Page(P + "10_Upload_Your_Digit.py", title="Upload Your Digit", icon=":material/upload_file:"),
        st.Page(P + "11_Batch_Prediction.py", title="Batch Prediction", icon=":material/stacks:"),
    ],
    "Data": [
        st.Page(P + "1_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/dataset:"),
        st.Page(P + "2_Image_Preprocessing.py", title="Image Preprocessing", icon=":material/tune:"),
        st.Page(P + "3_Image_Augmentation.py", title="Image Augmentation", icon=":material/rotate_right:"),
    ],
    "Modeling": [
        st.Page(P + "4_Model_Training.py", title="Model Training", icon=":material/model_training:"),
        st.Page(P + "5_Model_Comparison.py", title="Model Comparison", icon=":material/compare_arrows:"),
        st.Page(P + "6_Model_Evaluation.py", title="Model Evaluation", icon=":material/analytics:"),
    ],
    "Insight": [
        st.Page(P + "7_Error_Analysis.py", title="Error Analysis", icon=":material/bug_report:"),
        st.Page(P + "8_Explainable_AI.py", title="Explainable AI", icon=":material/visibility:"),
    ],
    "Lab": [
        st.Page(P + "12_About.py", title="About the Project", icon=":material/info:"),
    ],
}

st.logo("assets/logo.svg", size="large", icon_image="assets/icon.svg")
common.sidebar_status()
nav = st.navigation(pages, position="sidebar", expanded=True)
nav.run()
theme.footer()
