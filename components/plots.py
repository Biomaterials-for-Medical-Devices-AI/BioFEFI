import streamlit as st
from pathlib import Path

from options.enums import PlotOptionKeys


def plot_box(plot_dir: Path, box_title: str):
    """Display the plots in the given directory in the UI.

    Args:
        plot_dir (Path): The directory containing the plots.
        box_title (str): The title of the plot box.
    """
    plots = list(plot_dir.iterdir())
    with st.expander(box_title, expanded=len(plots) > 0):
        for p in plots:
            st.image(str(p))


def plot_options_box():
    """Expander containing the options for making plots"""
    with st.expander("Plot options", expanded=False):
        st.number_input(
            "Angle to rotate X-axis labels",
            min_value=0,
            max_value=90,
            value=10,
            key=PlotOptionKeys.RotateXAxisLabels,
        )
        st.number_input(
            "Angle to rotate Y-axis labels",
            min_value=0,
            max_value=90,
            value=60,
            key=PlotOptionKeys.RotateYAxisLabels,
        )
        st.checkbox(
            "Save all plots",
            key=PlotOptionKeys.SavePlots,
        )
