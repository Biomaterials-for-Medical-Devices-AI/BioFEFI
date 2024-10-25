import streamlit as st
from biofefi.options.enums import ConfigStateKeys


@st.experimental_fragment
def log_box():
    """Display a text area which shows that logs of the current pipeline run."""
    with st.expander("Pipeline report", expanded=True):
        st.text_area(
            "Logs",
            key=ConfigStateKeys.LogBox,
            height=200,
            disabled=True,
        )
