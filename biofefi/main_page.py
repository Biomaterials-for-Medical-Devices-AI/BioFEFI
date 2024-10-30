import streamlit as st
from biofefi.components.images.logos import header_logo, sidebar_logo

st.set_page_config(
    page_title="BioFEFI",
    page_icon=sidebar_logo(),
)
header_logo()
sidebar_logo()

st.write("# Welcome")
st.write(
    """
    **BioFEFI** allows you to **rapidly** develop machine learning models of many kinds, and evaluate their performance
    down to a **feature-by-feature** level.
    
    You can create models to solve either **classification** problems (e.g. is this image a cat 🐱 or a dog 🐶?)
    or **regression** problems (e.g. what will be the price of gold 🏅 tomorrow 📈?).

    Your models can then be evaluated by general measures, such as **accuracy**, and by individual feature metrics,
    such as **SHAP**.
    """
)
col1, col2 = st.columns(2)
col1.button("New experiment", use_container_width=True, type="primary")
col2.button("My experiments", use_container_width=True)
