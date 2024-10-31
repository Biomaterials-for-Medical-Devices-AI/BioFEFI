import streamlit as st
from biofefi.components.images.logos import sidebar_logo

st.set_page_config(
    page_title="BioFEFI",
    page_icon=sidebar_logo(),
)
