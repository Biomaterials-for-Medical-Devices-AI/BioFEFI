from argparse import Namespace
from multiprocessing import Process
from components.images.logos import header_logo, sidebar_logo
from components.logs import log_box
from components.forms import data_upload_form
from components.plots import ml_plots
from components.configuration import ml_options
from services.logs import get_logs
from feature_importance import feature_importance, fuzzy_interpretation
from feature_importance.feature_importance_options import FeatureImportanceOptions
from feature_importance.fuzzy_options import FuzzyOptions
from machine_learning import train
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from options.enums import ConfigStateKeys, ExecutionStateKeys
from options.file_paths import uploaded_file_path, log_dir, ml_plot_dir
from utils.logging_utils import Logger, close_logger
from utils.utils import set_seed
import streamlit as st
import os


def build_configuration() -> tuple[Namespace, Namespace, Namespace, str]:
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[Namespace, Namespace, Namespace, str]: The configuration for fuzzy, FI and ML pipelines,
        and the experiment name.
    """

    fuzzy_opt = FuzzyOptions()
    fuzzy_opt.initialize()
    if st.session_state.get(ConfigStateKeys.FuzzyFeatureSelection, False):
        fuzzy_opt.parser.set_defaults(
            fuzzy_feature_selection=st.session_state[
                ConfigStateKeys.FuzzyFeatureSelection
            ],
            num_fuzzy_features=st.session_state[ConfigStateKeys.NumberOfFuzzyFeatures],
            granular_features=st.session_state[ConfigStateKeys.GranularFeatures],
            num_clusters=st.session_state[ConfigStateKeys.NumberOfClusters],
            cluster_names=st.session_state[ConfigStateKeys.ClusterNames],
            num_rules=st.session_state[ConfigStateKeys.NumberOfTopRules],
            save_fuzzy_set_plots=st.session_state[ConfigStateKeys.SaveFuzzySetPlots],
            # fuzzy_log_dir=
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
            is_granularity=st.session_state[ConfigStateKeys.GranularFeatures],
        )
    fuzzy_opt = fuzzy_opt.parse()

    fi_opt = FeatureImportanceOptions()
    fi_opt.initialize()
    if st.session_state.get(ConfigStateKeys.IsFeatureImportance, False):
        fi_opt.parser.set_defaults(
            num_features_to_plot=st.session_state[
                ConfigStateKeys.NumberOfImportantFeatures
            ],
            permutation_importance_scoring=st.session_state[
                ConfigStateKeys.ScoringFunction
            ],
            permutation_importance_repeat=st.session_state[
                ConfigStateKeys.NumberOfRepetitions
            ],
            shap_reduce_data=st.session_state[ConfigStateKeys.ShapDataPercentage],
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
            is_feature_importance=st.session_state[ConfigStateKeys.IsFeatureImportance],
            # fi_log_dir=
            angle_rotate_xaxis_labels=st.session_state[
                ConfigStateKeys.RotateXAxisLabels
            ],
            angle_rotate_yaxis_labels=st.session_state[
                ConfigStateKeys.RotateYAxisLabels
            ],
            save_feature_importance_plots=st.session_state[
                ConfigStateKeys.SaveFeatureImportancePlots
            ],
            save_feature_importance_options=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceOptions
            ],
            save_feature_importance_results=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceResults
            ],
            local_importance_methods=st.session_state[
                ConfigStateKeys.LocalImportanceFeatures
            ],
            feature_importance_ensemble=st.session_state[
                ConfigStateKeys.EnsembleMethods
            ],
            global_importance_methods=st.session_state[
                ConfigStateKeys.GlobalFeatureImportanceMethods
            ],
        )
    fi_opt = fi_opt.parse()

    ml_opt = MLOptions()
    ml_opt.initialize()
    if st.session_state.get(ConfigStateKeys.IsMachineLearning, False):
        path_to_data = uploaded_file_path(
            st.session_state[ConfigStateKeys.UploadedFileName].name,
            st.session_state[ConfigStateKeys.ExperimentName],
        )
        ml_opt.parser.set_defaults(
            n_bootstraps=st.session_state[ConfigStateKeys.NumberOfBootstraps],
            save_actual_pred_plots=st.session_state[ConfigStateKeys.SavePlots],
            normalization=st.session_state[ConfigStateKeys.Normalization],
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            data_path=path_to_data,
            data_split=st.session_state[ConfigStateKeys.DataSplit],
            model_types=st.session_state[ConfigStateKeys.ModelTypes],
            ml_log_dir=ml_plot_dir(st.session_state[ConfigStateKeys.ExperimentName]),
            problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
            random_state=st.session_state[ConfigStateKeys.RandomSeed],
            is_machine_learning=st.session_state[ConfigStateKeys.IsMachineLearning],
        )
    ml_opt = ml_opt.parse()

    return fuzzy_opt, fi_opt, ml_opt, st.session_state[ConfigStateKeys.ExperimentName]


def save_upload(file_to_upload: str, content: str):
    """Save a file given to the UI to disk.

    Args:
        file_to_upload (str): The name of the file to save.
        content (str): The contents to save to the file.
    """
    base_dir = os.path.dirname(file_to_upload)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(file_to_upload, "w") as f:
        f.write(content)


def pipeline(
    fuzzy_opts: Namespace, fi_opts: Namespace, ml_opts: Namespace, experiment_name: str
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (Namespace): Options for fuzzy feature importance.
        fi_opts (Namespace): Options for feature importance.
        ml_opts (Namespace): Options for machine learning.
        experiment_name (str): The name of the experiment.
    """
    seed = ml_opts.random_state
    set_seed(seed)
    logger_instance = Logger(log_dir(experiment_name))
    logger = logger_instance.make_logger()

    # Machine learning
    if ml_opts.is_machine_learning:
        data = DataBuilder(ml_opts, logger).ingest()
        trained_models = train.run(ml_opts, data, logger)

        # Feature importance
        if fi_opts.is_feature_importance:
            gloabl_importance_results, local_importance_results, ensemble_results = (
                feature_importance.run(fi_opts, data, trained_models, logger)
            )

        # Fuzzy interpretation
        if fuzzy_opts.fuzzy_feature_selection:
            fuzzy_rules = fuzzy_interpretation.run(
                fuzzy_opts, ml_opts, data, trained_models, ensemble_results, logger
            )

    # Close the logger
    close_logger(logger_instance, logger)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        p.terminate()


## Page contents
st.set_page_config(
    page_title="BioFEFI",
    page_icon="static/BioFEFI_Logo_Transparent_160x160.png",
)
header_logo()
sidebar_logo()
with st.sidebar:
    st.header("Options")
    # st.checkbox("Feature Engineering", key=ConfigStateKeys.IsFeatureEngineering)

    # Machine Learning Options
    ml_options()

    # Feature Importance Options
    fi_on = st.checkbox("Feature Importance", key=ConfigStateKeys.IsFeatureImportance)
    if fi_on:
        with st.expander("Feature importance options"):
            st.write("Global feature importance methods:")
            global_methods = {}
            use_permutation = st.checkbox("Permutation Importance")
            global_methods["Permutation Importance"] = {
                "type": "global",
                "value": use_permutation,
            }
            use_shap = st.checkbox("SHAP")
            global_methods["SHAP"] = {"type": "global", "value": use_shap}
            st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = (
                global_methods
            )

            st.write("Feature importance ensemble methods:")
            ensemble_methods = {}
            use_mean = st.checkbox("Mean")
            ensemble_methods["Mean"] = use_mean
            use_majority = st.checkbox("Majority vote")
            ensemble_methods["Majority Vote"] = use_majority
            st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

            st.write("Local feature importance methods:")
            local_importance_methods = {}
            use_lime = st.checkbox("LIME")
            local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
            use_local_shap = st.checkbox("Local SHAP")
            local_importance_methods["SHAP"] = {
                "type": "local",
                "value": use_local_shap,
            }
            st.session_state[ConfigStateKeys.LocalImportanceFeatures] = (
                local_importance_methods
            )

            num_important_features = st.number_input(
                "Number of most important features to plot",
                min_value=1,
                value=10,
                key=ConfigStateKeys.NumberOfImportantFeatures,
            )
            scoring_function = st.selectbox(
                "Scoring function for permutation importance",
                [
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "accuracy",
                    "f1",
                ],
                key=ConfigStateKeys.ScoringFunction,
            )
            num_repetitions = st.number_input(
                "Number of repetitions for permutation importance",
                min_value=1,
                value=5,
                key=ConfigStateKeys.NumberOfRepetitions,
            )
            shap_data_percentage = st.slider(
                "Percentage of data to consider for SHAP",
                0,
                100,
                100,
                key=ConfigStateKeys.ShapDataPercentage,
            )
            angle_rotate_xaxis_labels = st.number_input(
                "Angle to rotate X-axis labels",
                min_value=0,
                max_value=90,
                value=10,
                key=ConfigStateKeys.RotateXAxisLabels,
            )
            angle_rotate_yaxis_labels = st.number_input(
                "Angle to rotate Y-axis labels",
                min_value=0,
                max_value=90,
                value=60,
                key=ConfigStateKeys.RotateYAxisLabels,
            )
            save_feature_importance_plots = st.checkbox(
                "Save feature importance plots",
                key=ConfigStateKeys.SaveFeatureImportancePlots,
            )
            save_feature_importance_options = st.checkbox(
                "Save feature importance options",
                key=ConfigStateKeys.SaveFeatureImportanceOptions,
            )
            save_feature_importance_results = st.checkbox(
                "Save feature importance results",
                key=ConfigStateKeys.SaveFeatureImportanceResults,
            )

            # Fuzzy Options
            st.subheader("Fuzzy Options")
            fuzzy_feature_selection = st.checkbox(
                "Fuzzy feature selection", key=ConfigStateKeys.FuzzyFeatureSelection
            )
            if fuzzy_feature_selection:
                num_fuzzy_features = st.number_input(
                    "Number of features for fuzzy interpretation",
                    min_value=1,
                    value=5,
                    key=ConfigStateKeys.NumberOfFuzzyFeatures,
                )
                granular_features = st.checkbox(
                    "Granular features", key=ConfigStateKeys.GranularFeatures
                )
                num_clusters = st.number_input(
                    "Number of clusters for target variable",
                    min_value=2,
                    value=3,
                    key=ConfigStateKeys.NumberOfClusters,
                )
                cluster_names = st.text_input(
                    "Names of clusters (comma-separated)",
                    key=ConfigStateKeys.ClusterNames,
                )
                num_top_rules = st.number_input(
                    "Number of top occurring rules for fuzzy synergy analysis",
                    min_value=1,
                    value=10,
                    key=ConfigStateKeys.NumberOfTopRules,
                )
                save_fuzzy_set_plots = st.checkbox(
                    "Save fuzzy set plots", key=ConfigStateKeys.SaveFuzzySetPlots
                )

    seed = st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )
data_upload_form()


# If the user has uploaded a file and pressed the run button, run the pipeline
if (
    uploaded_file := st.session_state.get(ConfigStateKeys.UploadedFileName)
) and st.session_state.get(ExecutionStateKeys.RunPipeline, False):
    experiment_name = st.session_state.get(ConfigStateKeys.ExperimentName)
    upload_path = uploaded_file_path(uploaded_file.name, experiment_name)
    save_upload(upload_path, uploaded_file.read().decode("utf-8"))
    config = build_configuration()
    process = Process(target=pipeline, args=config, daemon=True)
    process.start()
    cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
    with st.spinner("Running pipeline..."):
        # wait for the process to finish or be cancelled
        process.join()
    st.session_state[ConfigStateKeys.LogBox] = get_logs(log_dir(experiment_name))
    log_box()
    ml_plots(ml_plot_dir(experiment_name))
