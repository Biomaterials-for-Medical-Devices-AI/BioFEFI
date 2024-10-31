import os
import streamlit as st

from biofefi.components.images.logos import sidebar_logo
from biofefi.components.navigation import navbar
from biofefi.components.experiments import experiment_selector
from biofefi.options.enums import ViewExperimentKeys
from biofefi.options.file_paths import biofefi_experiments_base_dir


st.set_page_config(
    page_title="View Experiment",
    page_icon=sidebar_logo(),
)
navbar()

header = st.session_state.get(ViewExperimentKeys.ExperimentName)

st.header(header if header is not None else "View experiment")

# Get the base directory of all experiments
base_dir = biofefi_experiments_base_dir()
choices = os.listdir(base_dir)
# Filter out hidden files and directories
choices = filter(lambda x: not x.startswith("."), choices)
# Filter out files
choices = filter(lambda x: os.path.isdir(os.path.join(base_dir, x)), choices)
experiment_selector(choices)
