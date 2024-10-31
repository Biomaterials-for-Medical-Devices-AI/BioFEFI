import streamlit as st
from biofefi.components.images.logos import header_logo, sidebar_logo
from biofefi.components.navigation import navbar

st.set_page_config(
    page_title="BioFEFI",
    page_icon=sidebar_logo(),
)
header_logo()
sidebar_logo()
navbar()

st.write("# Welcome")
st.write(
    """
    **BioFEFI** stands for biological data feature importance fusion framework.

    Using BioFEFI, you can **rapidly** develop machine learning models of many kinds, and evaluate their performance
    down to a **feature-by-feature** level.
    
    You can create models to solve either **classification** problems (e.g. is this image a cat 🐱 or a dog 🐶?)
    or **regression** problems (e.g. what will be the price of gold 🏅 tomorrow 📈?).

    Your models can then be evaluated by general measures, such as **accuracy**, and by individual feature metrics,
    such as **SHAP**.
    """
)
col1, col2 = st.columns(2)
col1.page_link(
    "pages/new_experiment.py", label="New Experiment", use_container_width=True
)
col2.button("My experiments", use_container_width=True)
