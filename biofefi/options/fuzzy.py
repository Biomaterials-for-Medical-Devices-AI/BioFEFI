from dataclasses import dataclass


@dataclass
class FuzzyOptions:
    fuzzy_feature_selection: bool = True
    number_fuzzy_features: int = 10
    granular_features: bool = True
    number_clusters: int = 5
    cluster_names: list = ["very low", "low", "medium", "high", "very high"]
    number_rules: int = 5
    save_fuzzy_set_plots: bool = True
    fuzzy_log_dir: str = "fuzzy"
