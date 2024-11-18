from dataclasses import dataclass
from typing import Any


@dataclass
class MachineLearningOptions:
    n_bootstraps: int = 3
    save_actual_pred_plots: bool = True
    model_types: dict
    ml_log_dir: str = "ml"
    save_models: bool = True
    ml_plot_dir: str = "ml"
