from dataclasses import dataclass
from typing import Any


@dataclass
class MachineLearningOptions:
    n_bootstraps: int = 3
    save_actual_pred_plots: bool = True
    model_types: dict = {
        "Linear Model": {"use": False, "params": {"fit_intercept": False}},
        "Random Forest": {
            "use": True,
            "params": {
                "n_estimators": 300,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_depth": 6,
            },
        },
        "XGBoost": {
            "use": True,
            "params": {
                "kwargs": {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.01,
                    "subsample": 0.5,
                }
            },
        },
    }
    ml_log_dir: str = "ml"
    save_models: bool = True
    ml_plot_dir: str = "ml"
