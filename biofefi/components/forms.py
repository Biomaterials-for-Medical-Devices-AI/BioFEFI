import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from biofefi.options.choices.ui import SVM_KERNELS
from biofefi.options.enums import (
    ConfigStateKeys,
    ExecutionStateKeys,
    Normalisations,
    PlotOptionKeys,
    ProblemTypes,
)
from biofefi.options.file_paths import biofefi_experiments_base_dir, ml_model_dir
from biofefi.options.search_grids import (
    LINEAR_MODEL_GRID,
    RANDOM_FOREST_GRID,
    SVM_GRID,
    XGB_GRID,
)
from biofefi.services.ml_models import models_exist


def data_upload_form():
    """
    The main form for BioFEFI where the user supplies the data
    and says where they want their experiment to be saved.
    """
    st.header("Data Upload")
    save_dir = _save_directory_selector()
    # If a user has tried to enter a destination to save an experiment, show it
    # if it's valid, else show some red text showing the destination and saying
    # it's invalid.
    if not _directory_is_valid(save_dir) and st.session_state.get(
        ConfigStateKeys.ExperimentName
    ):
        st.markdown(f":red[Cannot use {save_dir}; it already exists.]")
    else:
        st.session_state[ConfigStateKeys.ExperimentName] = save_dir.name
    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    if not st.session_state.get(ConfigStateKeys.IsMachineLearning, False):
        st.file_uploader(
            "Upload machine leaerning models",
            type="pkl",
            accept_multiple_files=True,
            key=ConfigStateKeys.UploadedModels,
        )
    st.button("Run", key=ExecutionStateKeys.RunPipeline)


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = biofefi_experiments_base_dir()

    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

    col1.text(f"{root}{os.path.sep}", help="Your experiment will be saved here")
    sub_dir = col2.text_input("Name of the experiment", placeholder="e.g. MyExperiment")

    return root / sub_dir


def _directory_is_valid(directory: Path) -> bool:
    """Determine if the directory supplied by the user is valid. If it already exists,
    it is invalid.

    Args:
        directory (Path): The path to check.

    Returns:
        bool: `True` if the directory doesn't already exist, else `False`
    """
    return not directory.exists()


@st.experimental_fragment
def fi_options_form():
    global_methods = {}

    st.write("### Global Feature Importance Methods")
    st.write(
        "Select global methods to assess feature importance across the entire dataset. "
        "These methods help in understanding overall feature impact."
    )

    use_permutation = st.checkbox(
        "Permutation Importance",
        help="Evaluate feature importance by permuting feature values.",
    )

    global_methods["Permutation Importance"] = {
        "type": "global",
        "value": use_permutation,
    }

    use_shap = st.checkbox(
        "SHAP",
        help="Apply SHAP (SHapley Additive exPlanations) for global interpretability.",
    )
    global_methods["SHAP"] = {"type": "global", "value": use_shap}

    st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = global_methods

    st.write("### Ensemble Feature Importance Methods")
    st.write(
        "Ensemble methods combine results from multiple feature importance techniques, "
        "enhancing robustness. Choose how to aggregate feature importance insights."
    )

    # global methods need to be set to perform ensemble methods
    ensemble_is_disabled = not (use_permutation or use_shap)
    if ensemble_is_disabled:
        st.warning(
            "You must configure at least one global feature importance method to perform ensemble methods.",
            icon="⚠",
        )
    ensemble_methods = {}
    use_mean = st.checkbox(
        "Mean",
        help="Calculate the mean importance score across methods.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Mean"] = use_mean
    use_majority = st.checkbox(
        "Majority Vote",
        help="Use majority voting to identify important features.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Majority Vote"] = use_majority

    st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

    st.write("### Local Feature Importance Methods")
    st.write(
        "Select local methods to interpret individual predictions. "
        "These methods focus on explaining predictions at a finer granularity."
    )

    local_importance_methods = {}
    use_lime = st.checkbox(
        "LIME",
        help="Use LIME (Local Interpretable Model-Agnostic Explanations) for local interpretability.",
    )
    local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
    use_local_shap = st.checkbox(
        "Local SHAP",
        help="Use SHAP for local feature importance at the instance level.",
    )
    local_importance_methods["SHAP"] = {
        "type": "local",
        "value": use_local_shap,
    }

    st.session_state[ConfigStateKeys.LocalImportanceFeatures] = local_importance_methods

    st.write("### Additional Configuration Options")

    # Number of important features
    st.number_input(
        "Number of most important features to plot",
        min_value=1,
        value=10,
        help="Select how many top features to visualise based on their importance score.",
        key=ConfigStateKeys.NumberOfImportantFeatures,
    )

    # Scoring function for permutation importance
    if (
        st.session_state.get(ConfigStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Regression
    ):
        scoring_options = [
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
    elif (
        st.session_state.get(ConfigStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Classification
    ):
        scoring_options = ["accuracy", "f1"]
    else:
        scoring_options = []

    st.selectbox(
        "Scoring function for permutation importance",
        scoring_options,
        help="Choose a scoring function to evaluate the model during permutation importance.",
        key=ConfigStateKeys.ScoringFunction,
    )

    # Number of repetitions for permutation importance
    st.number_input(
        "Number of repetitions for permutation importance",
        min_value=1,
        value=5,
        help="Specify the number of times to shuffle each feature for importance estimation.",
        key=ConfigStateKeys.NumberOfRepetitions,
    )

    # Percentage of data to consider for SHAP
    st.slider(
        "Percentage of data to consider for SHAP",
        0,
        100,
        100,
        help="Set the percentage of data used to calculate SHAP values.",
        key=ConfigStateKeys.ShapDataPercentage,
    )

    # Fuzzy Options
    st.write("### Fuzzy Feature Importance Options")
    st.write(
        "Activate fuzzy methods to capture interactions between features in a fuzzy rule-based system. "
        "Define the number of features, clusters, and granular options for enhanced interpretability."
    )

    # both ensemble_methods and local_importance_methods
    fuzzy_is_disabled = (not (use_lime or use_local_shap)) or (
        not (use_mean or use_majority)
    )
    if fuzzy_is_disabled:
        st.warning(
            "You must configure both ensemble and local importance methods to use fuzzy feature selection.",
            icon="⚠",
        )
    fuzzy_feature_importance = st.checkbox(
        "Enable Fuzzy Feature Importance",
        help="Toggle fuzzy feature importance to analyze feature interactions.",
        key=ConfigStateKeys.FuzzyFeatureSelection,
        disabled=fuzzy_is_disabled,
    )

    if fuzzy_feature_importance:

        st.number_input(
            "Number of features for fuzzy interpretation",
            min_value=1,
            value=5,
            help="Set the number of features for fuzzy analysis.",
            key=ConfigStateKeys.NumberOfFuzzyFeatures,
        )

        st.checkbox(
            "Granular features",
            help="Divide features into granular categories for in-depth analysis.",
            key=ConfigStateKeys.GranularFeatures,
        )

        st.number_input(
            "Number of clusters for target variable",
            min_value=2,
            value=5,
            help="Set the number of clusters to categorise the target variable for fuzzy interpretation.",
            key=ConfigStateKeys.NumberOfClusters,
        )

        st.text_input(
            "Names of clusters (comma-separated)",
            help="Specify names for each cluster (e.g., Low, Medium, High).",
            key=ConfigStateKeys.ClusterNames,
            value=", ".join(["very low", "low", "medium", "high", "very high"]),
        )

        st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
            help="Set the number of most frequent fuzzy rules for synergy analysis.",
            key=ConfigStateKeys.NumberOfTopRules,
        )

    st.subheader("Select outputs to save")

    # Save options
    st.toggle(
        "Save feature importance options",
        help="Save the selected configuration of feature importance methods.",
        key=ConfigStateKeys.SaveFeatureImportanceOptions,
        value=True,
    )

    st.toggle(
        "Save feature importance results",
        help="Store the results from feature importance computations.",
        key=ConfigStateKeys.SaveFeatureImportanceResults,
        value=True,
    )


@st.experimental_fragment
def ml_options_form(use_hyperparam_search: bool):
    """
    The form for setting up the machine learning pipeline.

    Args:
        use_hyperparam_search (bool): Is the user using hyper-parameter search?
    """
    st.subheader("Select and cofigure which models to train")

    if models_exist(
        ml_model_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ConfigStateKeys.ExperimentName]
        )
    ):
        st.warning("You have trained models in this experiment.")
        st.checkbox(
            "Would you like to rerun the experiments? This will overwrite the existing models.",
            value=True,
            key=ConfigStateKeys.RerunML,
        )
    else:
        st.session_state[ConfigStateKeys.RerunML] = True

    if st.session_state[ConfigStateKeys.RerunML]:
        if use_hyperparam_search:
            st.success("**✨ Hyper-parameters will be searched automatically**")

        model_types = {}
        if st.toggle("Linear Model", value=False):
            lm_model_type = _linear_model_opts(use_hyperparam_search)
            model_types.update(lm_model_type)

        if st.toggle("Random Forest", value=False):
            rf_model_type = _random_forest_opts(use_hyperparam_search)
            model_types.update(rf_model_type)

        if st.toggle("XGBoost", value=False):
            xgb_model_type = _xgboost_opts(use_hyperparam_search)
            model_types.update(xgb_model_type)

        if st.toggle("Support Vector Machine", value=False):
            svm_model_type = _svm_opts(use_hyperparam_search)
            model_types.update(svm_model_type)

        st.session_state[ConfigStateKeys.ModelTypes] = model_types
        st.subheader("Select outputs to save")
        st.toggle(
            "Save models",
            key=ConfigStateKeys.SaveModels,
            value=True,
            help="Save the models that are trained to disk?",
        )
        st.toggle(
            "Save plots",
            key=PlotOptionKeys.SavePlots,
            value=True,
            help="Save the plots to disk?",
        )


@st.experimental_fragment
def target_variable_dist_form(data, dep_var_name, data_analysis_plot_dir, plot_opts):
    """
    Form to create the target variable distribution plot.
    """

    show_kde = st.toggle("Show KDE", value=True, key=ConfigStateKeys.ShowKDE)
    n_bins = st.slider(
        "Number of Bins",
        min_value=5,
        max_value=50,
        value=10,
        key=ConfigStateKeys.NBins,
    )

    if st.checkbox(
        "Create Target Variable Distribution Plot",
        key=ConfigStateKeys.TargetVarDistribution,
    ):
        plt.style.use(plot_opts.plot_colour_scheme)
        plt.figure(figsize=(10, 6))
        displot = sns.displot(data=data, x=data.columns[-1], kde=show_kde, bins=n_bins)
        plt.title(
            f"{dep_var_name} Distribution",
            fontdict={
                "fontsize": plot_opts.plot_title_font_size,
                "family": plot_opts.plot_font_family,
            },
        )

        plt.xlabel(
            dep_var_name,
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        plt.ylabel(
            "Frequency",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        plt.xticks(
            fontsize=plot_opts.plot_axis_tick_size, family=plot_opts.plot_font_family
        )
        plt.yticks(
            fontsize=plot_opts.plot_axis_tick_size, family=plot_opts.plot_font_family
        )

        st.pyplot(displot)

        if st.button("Save Plot", key=ConfigStateKeys.SaveTargetVarDistribution):

            displot.savefig(data_analysis_plot_dir / f"{dep_var_name}_distribution.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def correlation_heatmap_form(data, data_analysis_plot_dir, plot_opts):
    """
    Form to create the correlation heatmap plot.
    """

    if st.toggle(
        "Select All Descriptors",
        value=False,
        key=ConfigStateKeys.SelectAllDescriptorsCorrelation,
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = []

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

    if st.checkbox(
        "Create Correlation Heatmap Plot", key=ConfigStateKeys.CorrelationHeatmap
    ):

        corr = corr_data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        plt.style.use(plot_opts.plot_colour_scheme)
        fig, ax = plt.subplots(figsize=(11, 9))

        ax.set_title(
            "Correlation Heatmap",
            fontsize=plot_opts.plot_title_font_size,
            family=plot_opts.plot_font_family,
            wrap=True,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=plot_opts.plot_axis_tick_size,
            family=plot_opts.plot_font_family,
        )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=plot_opts.plot_axis_tick_size,
            family=plot_opts.plot_font_family,
        )

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(
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

        st.pyplot(fig)

        if st.button("Save Plot", key=ConfigStateKeys.SaveHeatmap):

            fig.savefig(data_analysis_plot_dir / "correlation_heatmap.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def pairplot_form(data, data_analysis_plot_dir, plot_opts):
    """
    Form to create the pairplot plot.
    """

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

    if st.checkbox("Create Pairplot", key=ConfigStateKeys.PairPlot):

        plt.style.use(plot_opts.plot_colour_scheme)
        plt.figure(figsize=(10, 6))
        pairplot = sns.pairplot(pairplot_data, corner=True)
        st.pyplot(plt)

        if st.button("Save Plot", key=ConfigStateKeys.SavePairPlot):
            pairplot.savefig(data_analysis_plot_dir / "pairplot.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def tSNE_plot_form(
    data, random_state, data_analysis_plot_dir, plot_opts, scaler: Normalisations = None
):

    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]

    if scaler == Normalisations.NoNormalisation:
        scaler = st.selectbox(
            "Select Normalisation for Comparison (this will not affect the normalisation for ML models)",
            options=[Normalisations.Standardization, Normalisations.MinMax],
            key=ConfigStateKeys.SelectNormTsne,
        )

    if scaler == Normalisations.MinMax:
        X_scaled = MinMaxScaler().fit_transform(X)
    elif scaler == Normalisations.Standardization:
        X_scaled = StandardScaler().fit_transform(X)

    perplexity = st.slider(
        "Perplexity",
        min_value=5,
        max_value=50,
        value=30,
        help="The perplexity parameter controls the balance between local and global aspects of the data.",
        key=ConfigStateKeys.Perplexity,
    )

    if st.checkbox("Create t-SNE Plot", key=ConfigStateKeys.tSNEPlot):

        tsne_normalised = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )
        tsne_original = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )

        X_embedded_normalised = tsne_normalised.fit_transform(X_scaled)
        X_embedded = tsne_original.fit_transform(X)

        df_normalised = pd.DataFrame(X_embedded_normalised, columns=["x", "y"])
        df_normalised["target"] = y

        df = pd.DataFrame(X_embedded, columns=["x", "y"])
        df["target"] = y

        plt.style.use(plot_opts.plot_colour_scheme)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Normalized Data
        sns.scatterplot(
            data=df_normalised,
            x="x",
            y="y",
            hue="target",
            palette="viridis",
            ax=axes[0],
        )
        axes[0].set_title(
            "t-SNE Plot (Normalised Features)",
            fontsize=plot_opts.plot_title_font_size,
            family=plot_opts.plot_font_family,
        )
        axes[0].set_xlabel(
            "t-SNE Component 1",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )
        axes[0].set_ylabel(
            "t-SNE Component 2",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        # Plot 2: Original Data
        sns.scatterplot(
            data=df, x="x", y="y", hue="target", palette="viridis", ax=axes[1]
        )
        axes[1].set_title(
            "t-SNE Plot (Original Features)",
            fontsize=plot_opts.plot_title_font_size,
            family=plot_opts.plot_font_family,
        )
        axes[1].set_xlabel(
            "t-SNE Component 1",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )
        axes[1].set_ylabel(
            "t-SNE Component 2",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        plt.xticks(
            fontsize=plot_opts.plot_axis_tick_size, family=plot_opts.plot_font_family
        )
        plt.yticks(
            fontsize=plot_opts.plot_axis_tick_size, family=plot_opts.plot_font_family
        )

        plt.tight_layout()

        st.pyplot(fig)

        if st.button("Save Plot", key=ConfigStateKeys.SaveTSNEPlot):

            fig.savefig(data_analysis_plot_dir / "tsne_plot.png")
            plt.clf()
            st.success("Plots created and saved successfully.")


def _linear_model_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        fit_intercept = st.checkbox("Fit intercept")
        params = {
            "fit_intercept": fit_intercept,
        }
        st.divider()
    else:
        params = LINEAR_MODEL_GRID
    model_types["Linear Model"] = {
        "use": True,
        "params": params,
    }
    return model_types


def _random_forest_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        n_estimators_rf = st.number_input(
            "Number of estimators", value=100, key="n_estimators_rf"
        )
        min_samples_split = st.number_input("Minimum samples split", value=2)
        min_samples_leaf = st.number_input("Minimum samples leaf", value=1)
        col1, col2 = st.columns([0.25, 0.75], vertical_alignment="bottom", gap="small")
        use_rf_max_depth = col1.checkbox(
            "Set max depth",
            value=False,
            help="If disabled or 0, then nodes are expanded until all leaves are pure"
            " or until all leaves contain less than 'Minimum samples split'.",
        )
        max_depth_rf = col2.number_input(
            "Maximum depth",
            value="min",
            min_value=0,
            key="max_depth_rf",
            disabled=not use_rf_max_depth,
        )
        params = {
            "n_estimators": n_estimators_rf,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth_rf if max_depth_rf > 0 else None,
        }
        st.divider()
    else:
        params = RANDOM_FOREST_GRID
    model_types["Random Forest"] = {
        "use": True,
        "params": params,
    }
    return model_types


def _xgboost_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        if st.checkbox("Set XGBoost options"):
            st.write("Options:")
            n_estimators_xgb = st.number_input(
                "Number of estimators", value=100, key="n_estimators_xgb"
            )
            learning_rate = st.number_input("Learning rate", value=0.01)
            subsample = st.number_input("Subsample size", value=0.5)
            col1, col2 = st.columns(
                [0.25, 0.75], vertical_alignment="bottom", gap="small"
            )
            use_xgb_max_depth = col1.checkbox(
                "Set max depth",
                value=False,
                help="If disabled or 0, then nodes are expanded until all leaves are pure.",
            )
            max_depth_xbg = col2.number_input(
                "Maximum depth",
                value="min",
                min_value=0,
                key="max_depth_xgb",
                disabled=not use_xgb_max_depth,
            )
        else:
            n_estimators_xgb = None
            max_depth_xbg = None
            learning_rate = None
            subsample = None
        params = {
            "kwargs": {
                "n_estimators": n_estimators_xgb,
                "max_depth": max_depth_xbg,
                "learning_rate": learning_rate,
                "subsample": subsample,
            }
        }
        st.divider()
    else:
        params = XGB_GRID

    model_types["XGBoost"] = {
        "use": True,
        "params": params,
    }
    return model_types


def _svm_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        kernel = st.selectbox("Kernel", options=SVM_KERNELS)
        degree = st.number_input("Degree", min_value=0, value=3)
        c = st.number_input("C", value=1.0, min_value=0.0)
        params = {
            "kernel": kernel.lower(),
            "degree": degree,
            "C": c,
        }
        st.divider()
    else:
        params = SVM_GRID

    model_types["SVM"] = {
        "use": True,
        "params": params,
    }
    return model_types
