import json, dataclasses
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import shap

from biofefi.options.plotting import PlottingOptions


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


def plot_lime_importance(
    df: pd.DataFrame,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot LIME importance.

    Args:
        df (pd.DataFrame): The LIME data to plot
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The top number of features to plot.
        title (str): The title of the plot.

    Returns:
        Figure: The LIME plot.
    """
    # Calculate most important features
    most_importance_features = (
        df.abs()
        .mean()
        .sort_values(ascending=False)
        .head(num_features_to_plot)
        .index.to_list()
    )

    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")

    sns.violinplot(data=df.loc[:, most_importance_features], fill=True, ax=ax)

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
    ax.set_ylabel("Importance", family=plot_opts.plot_font_family)
    ax.set_title(title, family=plot_opts.plot_font_family)
    return fig


def plot_local_shap_importance(
    shap_values: shap.Explainer,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot a beeswarm plot of the local SHAP values.

    Args:
        shap_values (shap.Explainer): The SHAP explainer to produce the plot from.
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The number of top features to plot.
        title (str): The plot title.

    Returns:
        Figure: The beeswarm plot of local SHAP values.
    """
    # Plot bee swarm plot
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


def plot_global_shap_importance(
    shap_values: pd.DataFrame,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Produce a bar chart of global SHAP values.

    Args:
        shap_values (pd.DataFrame): The `DataFrame` containing the global SHAP values.
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The number of top features to plot.
        title (str): The plot title.

    Returns:
        Figure: The bar chart of global SHAP values.
    """
    # Plot bar chart
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")
    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
    )
    plot_data = (
        shap_values.sort_values(by=0, ascending=False).head(num_features_to_plot).T
    )
    sns.barplot(data=plot_data, fill=True, ax=ax)
    ax.set_xlabel(ax.get_xlabel(), family=plot_opts.plot_font_family)
    ax.set_ylabel("Abs. SHAP Importance", family=plot_opts.plot_font_family)
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

    return fig
