import json, dataclasses
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import shap

from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_options_dir,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_result_dir,
)
from biofefi.options.plotting import PlottingOptions
from biofefi.utils.logging_utils import Logger
from biofefi.utils.utils import log_options


def save_plot_options(path: Path, options: PlottingOptions):
    """Save plot options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (PlottingOptions): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file)


def load_plot_options(path: Path) -> PlottingOptions:
    """Load plotting options from the given path.
    The path will be to a `json` file containing the plot options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        PlottingOptions: The plotting options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    options = PlottingOptions(**options_json)
    return options


def save_importance_results(
    feature_importance_df: pd.DataFrame,
    model_type,
    importance_type: str,
    feature_importance_type: str,
    experiment_name: str,
    fi_opt: FeatureImportanceOptions,
    plot_opt: PlottingOptions,
    logger: Logger,
    shap_values=None,
):
    """Save the feature importance results to a CSV file and the plots.

    Args:
        feature_importance_df (pd.DataFrame): DataFrame of feature importance results.
        model_type (_type_): Type of model.
        importance_type (str): Type of feature importance method.
        feature_importance_type (str): Type of feature importance method (Again for some reason).
        experiment_name (str): Name of the experiment, to know where to save outputs.
        fi_opt (FeatureImportanceOptions): Feature importance options.
        plot_opt (PlottingOptions): Plotting options.
        logger (Logger): The logger.
        shap_values (_type_, optional): SHAP values. Defaults to None.
    """

    biofefi_base_dir = biofefi_experiments_base_dir()
    logger.info(f"Saving importance results and plots of {feature_importance_type}...")

    # Save plots when the flag is set to True and importance type is not fuzzy
    if fi_opt.save_feature_importance_plots and importance_type != "fuzzy":
        save_dir = fi_plot_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        plt.style.use(plot_opt.plot_colour_scheme)
        fig, ax = plt.subplots(layout="constrained")

        feature_importance_df.sort_values(by=0, ascending=False).head(
            fi_opt.num_features_to_plot
        ).plot(
            kind="bar",
            legend=False,
            ax=ax,
            title=f"{feature_importance_type} - {model_type}",
            ylabel="Importance",
        )
        # rotate x-axis labels for better readability
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=plot_opt.angle_rotate_xaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=plot_opt.angle_rotate_yaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.set_xlabel(ax.get_xlabel(), family=plot_opt.plot_font_family)
        ax.set_ylabel(ax.get_ylabel(), family=plot_opt.plot_font_family)
        ax.set_title(ax.get_title(), family=plot_opt.plot_font_family)
        fig.savefig(save_dir / f"{model_type}-bar.png")

        if feature_importance_type == "SHAP":
            # Plot bee swarm plot
            fig, ax = plt.subplots(layout="constrained")
            ax.set_title(
                f"{feature_importance_type} - {model_type}",
                family=plot_opt.plot_font_family,
            )
            shap.plots.beeswarm(
                shap_values, max_display=fi_opt.num_features_to_plot, show=False
            )
            ax.set_xlabel(ax.get_xlabel(), family=plot_opt.plot_font_family)
            ax.set_ylabel(ax.get_ylabel(), family=plot_opt.plot_font_family)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                family=plot_opt.plot_font_family,
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                family=plot_opt.plot_font_family,
            )
            fig.savefig(save_dir / f"{model_type}-beeswarm.png")

    # Save the results to a CSV file - create folders if they don't exist
    if fi_opt.save_feature_importance_results and importance_type != "fuzzy":
        save_dir = fi_result_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    if fi_opt.save_feature_importance_results and importance_type == "fuzzy":
        save_dir = fuzzy_result_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    # Save the metrics to a log file
    if fi_opt.save_feature_importance_options:
        options_path = fi_options_dir(biofefi_base_dir / experiment_name)
        if not options_path.exists():
            options_path.mkdir(parents=True, exist_ok=True)
        log_options(options_path, fi_opt)


def plot_inidvidual_importance(
    df: pd.DataFrame,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot individual feature importance as a bar chart.

    Args:
        df (pd.DataFrame): The data to plot
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The number of features to plot importance.
        title (str): The title of the plot.

    Returns:
        Figure: The bar chart showing the top `num_features_to_plot`.
    """

    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")

    df.sort_values(by=0, ascending=False).head(num_features_to_plot).plot(
        kind="bar",
        legend=False,
        ax=ax,
        title=title,
        ylabel="Importance",
    )
    # rotate x-axis labels for better readability
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=plot_opts.angle_rotate_xaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=plot_opts.angle_rotate_yaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_xlabel(ax.get_xlabel(), family=plot_opts.plot_font_family)
    ax.set_ylabel(ax.get_ylabel(), family=plot_opts.plot_font_family)
    ax.set_title(ax.get_title(), family=plot_opts.plot_font_family)
    return fig


def plot_global_shap(
    shap_values: Any,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot global SHAP values as a beeswarm plot.

    Args:
        shap_values (Any): The SHAP values to plot.
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The top number of features to show.
        title (str): The title of the plot.

    Returns:
        Figure: The beeswarm plot.
    """
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")
    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
    )
    shap.plots.beeswarm(shap_values, max_display=num_features_to_plot, show=False)
    ax.set_xlabel(ax.get_xlabel(), family=plot_opts.plot_font_family)
    ax.set_ylabel(ax.get_ylabel(), family=plot_opts.plot_font_family)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        family=plot_opts.plot_font_family,
    )
    return fig
