import streamlit as st


def header_logo():
    """Generate a centred header logo for the app."""
    _, centre, _ = st.columns(3)
    with centre:
        st.image("static/BioFEFI_Logo_Transparent_160x160.png")