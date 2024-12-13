from dataclasses import dataclass
from typing import List

from biofefi.options.enums import ProblemTypes


@dataclass
class SyntheticDataOptions:

    # --- Problem Type Options ---
    problem_type: str = ProblemTypes.Classification  # Classification or Regression

    # ---- General Options ----
    n_samples: int = 500
    is_shuffle: bool = True
    random_state: int = 46

    # --- Classification Options ---
    n_features: int = 20
    n_informative: int = 2
    n_redudant: int = 2
    n_repeated: int = 0
    n_classes: int = 2
    n_clusters_per_class: int = 2
    data_weights: List[float] | None = None
    flip_y: float = 0.01
    class_sep: float = 1.0
    data_scale: float | None = None

    # --- Regression Options ---
    reg_n_informative: int = 10
    reg_n_targets: int = 1
    reg_bias: float = 0.0
    reg_noise: float = 0.0
    reg_coef: bool = False
