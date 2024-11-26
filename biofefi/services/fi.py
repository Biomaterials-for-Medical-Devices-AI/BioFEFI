from typing import Any
import pandas as pd
import shap

from biofefi.options.fi import FeatureImportanceOptions
from biofefi.utils.logging_utils import Logger


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    shap_type: str,
    fi_opt: FeatureImportanceOptions,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate SHAP values for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): The dataset.
        shap_type (str): The type of SHAP (local or global).
        fi_opt (FeatureImportanceOptions): The options.
        logger (Logger): The logger.

    Raises:
        ValueError: SHAP type is not "local" or "global"

    Returns:
        tuple[pd.DataFrame, Any]: SHAP dataframe and SHAP values.
    """
    logger.info(f"Calculating SHAP Importance for {model.__class__.__name__} model..")

    if fi_opt.shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(
            X, int(X.shape[0] * fi_opt.shap_reduce_data / 100)
        )
        explainer = shap.Explainer(model.predict, X_reduced)

    shap_values = explainer(X)

    # add an option to check if feature importance is local or global
    if shap_type == "local":
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
        # TODO: scale coefficients between 0 and +1 (low to high impact)
    elif shap_type == "global":
        # Calculate Average Importance + set column names as index
        shap_df = (
            pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame()
        )
    else:
        raise ValueError("SHAP type must be either local or global")

    logger.info(f"SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values
