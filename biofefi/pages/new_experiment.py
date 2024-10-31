import streamlit as st
from biofefi.components.images.logos import sidebar_logo

st.set_page_config(
    page_title="New Experiment",
    page_icon=sidebar_logo(),
)

st.header("New experiment")
