import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.plots import plot_box
from biofefi.options.enums import ConfigStateKeys
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    data_analysis_plots_dir,
    execution_options_path,
    plot_options_path,
)
from biofefi.services.configuration import load_execution_options, load_plot_options
from biofefi.services.experiments import get_experiments
from biofefi.utils.utils import create_directory

st.set_page_config(
    page_title="Data Visualisation",
    page_icon=sidebar_logo(),
)

st.header("Data Visualisation")
st.write(
    """
    Here you can visualise your data. This is useful for understanding the distribution of your data, 
    as well as the correlation between different features.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = biofefi_experiments_base_dir()

if experiment_name:
    st.session_state[ConfigStateKeys.ExperimentName] = experiment_name

    path_to_exec_opts = execution_options_path(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    path_to_plot_opts = plot_options_path(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    data_analysis_plot_dir = data_analysis_plots_dir(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    if not data_analysis_plot_dir.exists():
        create_directory(data_analysis_plot_dir)

    exec_opt = load_execution_options(path_to_exec_opts)
    plot_opt = load_plot_options(path_to_plot_opts)

    data = pd.read_csv(exec_opt.data_path)

    st.write("### Data")

    st.write(data)

    st.write("#### Data Description")

    st.write(data.describe())

    st.write("### Data Visualisation")

    st.write("#### Target Variable Distribution")

    if st.checkbox(
        "Create Target Variable Distribution Plot",
        key=ConfigStateKeys.TargetVarDistribution,
    ):
        show_kde = st.toggle("Show KDE", value=True, key=ConfigStateKeys.ShowKDE)
        n_bins = st.slider(
            "Number of Bins",
            min_value=5,
            max_value=50,
            value=10,
            key=ConfigStateKeys.NBins,
        )

    st.write("#### Correlation Heatmap")

    if st.checkbox(
        "Create Correlation Heatmap Plot", key=ConfigStateKeys.CorrelationHeatmap
    ):

        if st.toggle(
            "Select All Descriptors",
            value=False,
            key=ConfigStateKeys.SelectAllDescriptorsCorrelation,
        ):
            default_corr = list(data.columns[:-1])
        else:
            default_corr = None

        corr_descriptors = st.multiselect(
            "Select columns to include in the correlation heatmap",
            data.columns[:-1],
            default=default_corr,
            key=ConfigStateKeys.DescriptorCorrelation,
        )

        corr_data = data[corr_descriptors + [data.columns[-1]]]

        if len(corr_descriptors) < 1:
            st.warning(
                "Please select at least one descriptor to create the correlation heatmap."
            )
            st.stop()

    st.write("#### Pairplot")

    if st.checkbox("Create Pairplot", key=ConfigStateKeys.PairPlot):

        if st.toggle(
            "Select All Descriptors",
            value=False,
            key=ConfigStateKeys.SelectAllDescriptorsPairPlot,
        ):
            default_corr = list(data.columns[:-1])
        else:
            default_corr = None

        descriptors = st.multiselect(
            "Select columns to include in the pairplot",
            data.columns[:-1],
            default=default_corr,
            key=ConfigStateKeys.DescriptorPairPlot,
        )

        pairplot_data = data[descriptors + [data.columns[-1]]]

        if len(descriptors) < 1:
            st.warning(
                "Please select at least one descriptor to create the correlation plot."
            )
            st.stop()

    st.write("#### t-SNE Plot")

    if st.checkbox("Create t-SNE Plot", key=ConfigStateKeys.tSNEPlot):
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

        X = StandardScaler().fit_transform(X)

    if st.button("Save Plots", key=ConfigStateKeys.SavePlots):
        st.write("Saving plots...")
        if st.session_state[ConfigStateKeys.TargetVarDistribution]:
            displot = sns.displot(
                data=data, x=data.columns[-1], kde=show_kde, bins=n_bins
            )
            displot.set(title=f"{exec_opt.dependent_variable} Distribution")
            displot.savefig(
                data_analysis_plot_dir
                / f"{exec_opt.dependent_variable}_distribution.png"
            )
            plt.clf()
        if st.session_state[ConfigStateKeys.CorrelationHeatmap]:
            corr = corr_data.corr()
            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            heatmap = sns.heatmap(
                corr,
                mask=mask,
                cmap=cmap,
                vmax=0.3,
                center=0,
                square=True,
                linewidths=0.5,
                annot=True,
                cbar_kws={"shrink": 0.5},
            )
            f.savefig(data_analysis_plot_dir / "correlation_heatmap.png")
            plt.clf()
        if st.session_state[ConfigStateKeys.PairPlot]:
            pairplot = sns.pairplot(pairplot_data, corner=True)
            pairplot.savefig(data_analysis_plot_dir / "pairplot.png")
            plt.clf()
        if st.session_state[ConfigStateKeys.tSNEPlot]:
            tsne = TSNE(n_components=2, random_state=exec_opt.random_state)
            X_embedded = tsne.fit_transform(X)

            df = pd.DataFrame(X_embedded, columns=["x", "y"])
            df["target"] = y

            fig = plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x="x", y="y", hue="target", palette="viridis")
            plt.title("t-SNE Plot")
            plt.ylabel("t-SNE Component 2")
            plt.xlabel("t-SNE Component 1")
            fig.savefig(data_analysis_plot_dir / "tsne_plot.png")
            plt.clf()

        st.success("Plots created and saved successfully.")

        plot_box(data_analysis_plot_dir, "Data Visualisation Plots")
