import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_scatter(y, yp, r2, set_name, dependent_variable, model_name, directory):

    # Plotting the dataset
    fig, ax = plt.subplots()
    ax.scatter(y, yp)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured " + dependent_variable, fontsize=13)
    ax.set_ylabel("Predicted " + dependent_variable, fontsize=13)
    figure_title = "Prediction Error for" + model_name + " - " + set_name
    plt.title(figure_title, fontsize=13)
    legend = "R2: " + str(float("{0:.2f}".format(r2)))
    plt.legend(["Best fit", legend], loc="upper left", fontsize=13)
    plt.grid(axis="both")
    plt.savefig(f"{directory}/{model_name}.png")
    plt.close()


def save_actual_pred_plots(data, results, opt: argparse.Namespace, logger) -> None:
    """Save Actual vs Predicted plots for Regression models
    Args:
        data: Data object
        results: Results of the model
        opt: Options
        logger: Logger
    Returns:
        None
    """
    if opt.problem_type == "regression":

        # Create results directory if it doesn't exist
        directory = opt.ml_log_dir
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        # Convert train and test sets to numpy arrays for easier handling
        y_test = [np.array(df) for df in data.y_test]
        y_train = [np.array(df) for df in data.y_train]

        # Scatter plot of actual vs predicted values
        for model_name, model_options in opt.model_types.items():
            if model_options["use"]:
                logger.info(f"Saving actual vs prediction plots of {model_name}...")

                for i in range(opt.n_bootstraps):
                    y_pred_test = results[i][model_name]["y_pred_test"]
                    y_pred_train = results[i][model_name]["y_pred_train"]
                    dependent_variable = opt.dependent_variable

                    # Plotting the training and test results
                    plot_scatter(
                        y_test[i],
                        y_pred_test,
                        results[i][model_name]["r2_test"],
                        "Test",
                        dependent_variable,
                        model_name,
                        directory,
                    )
                    plot_scatter(
                        y_train[i],
                        y_pred_train,
                        results[i][model_name]["r2_train"],
                        "Train",
                        dependent_variable,
                        model_name,
                        directory,
                    )

                # sns.scatterplot(
                #     x=y_test[i], y=y_pred_test, marker="o", s=30, color="black"
                # )
