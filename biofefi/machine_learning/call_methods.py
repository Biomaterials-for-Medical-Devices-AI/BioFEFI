import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import ProblemTypes
from biofefi.options.execution import ExecutionOptions
from biofefi.options.file_paths import ml_plot_dir
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions


def plot_scatter(
    y,
    yp,
    r2: float,
    set_name: str,
    dependent_variable: str,
    model_name: str,
    directory: str,
    plot_opts: PlottingOptions | None = None,
):
    """_summary_

    Args:
        y (_type_): True y values.
        yp (_type_): Predicted y values.
        r2 (float): R-squared between `y`and `yp`.
        set_name (str): "Train" or "Test".
        dependent_variable (str): The name of the dependent variable.
        model_name (str): Name of the model.
        directory (str): The directory to save the plot.
        plot_opts (PlottingOptions | None, optional): Options for styling the plot. Defaults to None.
    """

    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=yp)

    # Add the best fit line
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)

    # Set labels and title
    plt.xlabel("Measured " + dependent_variable, fontsize=13)
    plt.ylabel("Predicted " + dependent_variable, fontsize=13)
    figure_title = "Prediction Error for " + model_name + " - " + set_name
    plt.title(figure_title, fontsize=13)

    # Add legend
    legend = "R2: " + str(float("{0:.2f}".format(r2["value"])))
    plt.legend(["Best fit", legend], loc="upper left", fontsize=13)

    # Add grid
    plt.grid(axis="both")

    # Save the figure
    plt.savefig(f"{directory}/{model_name}-{set_name}.png")
    plt.close()


def save_actual_pred_plots(
    data: DataBuilder,
    ml_results,
    opt: ExecutionOptions,
    logger,
    ml_metric_results,
    plot_opts: PlottingOptions | None = None,
    ml_opts: MachineLearningOptions | None = None,
) -> None:
    """Save Actual vs Predicted plots for Regression models
    Args:
        data: Data object
        ml_results: Results of the model
        opt: Options
        logger: Logger
        ml_metric_results: metrics of machine learning models
    Returns:
        None
    """
    if opt.problem_type == ProblemTypes.Regression:

        # Create results directory if it doesn't exist
        directory = ml_plot_dir(opt.experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        # Convert train and test sets to numpy arrays for easier handling
        y_test = [np.array(df) for df in data.y_test]
        y_train = [np.array(df) for df in data.y_train]

        # Scatter plot of actual vs predicted values
        for model_name, model_options in ml_opts.model_types.items():
            if model_options["use"]:
                logger.info(f"Saving actual vs prediction plots of {model_name}...")

                for i in range(ml_opts.n_bootstraps):
                    y_pred_test = ml_results[i][model_name]["y_pred_test"]
                    y_pred_train = ml_results[i][model_name]["y_pred_train"]

                    # Plotting the training and test results
                    plot_scatter(
                        y_test[i],
                        y_pred_test,
                        ml_metric_results[model_name][i]["R2"]["test"],
                        "Test",
                        opt.dependent_variable,
                        model_name,
                        directory,
                    )
                    plot_scatter(
                        y_train[i],
                        y_pred_train,
                        ml_metric_results[model_name][i]["R2"]["train"],
                        "Train",
                        opt.dependent_variable,
                        model_name,
                        directory,
                    )
