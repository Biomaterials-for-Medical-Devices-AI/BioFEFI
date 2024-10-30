from biofefi.machine_learning.ml_options import MLOptions
import ast


class SyntheticDataOptions(MLOptions):
    """
    Options for creating synthetic data for machine learning tasks.
    """

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        MLOptions.initialize(self)

        # ------- General Options ------- >>>

        self.parser.add_argument(
            "--num_samples",
            type=int,
            default=300,
            choices=[300, 500, 700],
            help="Number of samples to create for synthetic data",
        )

        self.parser.add_argument(
            "--num_features",
            type=int,
            default=15,
            choices=[15, 20, 25, 30],
            help="Number of features to create for synthetic data",
        )

        self.parser.add_argument(
            "--is_shuffle",
            type=bool,
            default=True,
            help="Flag to shuffle the synthetic data",
        )

        # --- Additional General Options can be added above --- >>>
        # ------------------------------------------------------------

        # ------- Classification Options for Synthetic Data ------- >>>

        self.parser.add_argument(
            "--num_informative",
            type=int,
            default=2,
            help="Number of informative features to create for synthetic data",
        )

        self.parser.add_argument(
            "--num_redundant",
            type=int,
            default=2,
            choices=[2, 3, 4],
            help="Number of redundant features to create for synthetic data."
            "These features are generated as random linear combinations of the informative features.",
        )

        self.parser.add_argument(
            "--num_repeated",
            type=int,
            default=0,
            help="Number of duplicated features to create for synthetic data."
            "These features are generated as exact copies of other features.",
        )

        self.parser.add_argument(
            "--num_classes",
            type=int,
            default=2,
            choices=[2, 3, 4],
            help="Number of classes to create for synthetic data for a classification problem.",
        )

        self.parser.add_argument(
            "--num_clusters_per_class",
            type=int,
            default=2,
            help="Number of clusters per class to create for synthetic data for a classification problem.",
        )

        self.parser.add_argument(
            "--data_weights",
            type=lambda x: ast.literal_eval(x),
            default=None,
            help="Weights associated with classes in the form of a list of floats or None",
        )

        self.parser.add_argument(
            "--flip_y",
            type=float,
            default=0.01,
            help="The fraction of samples whose class are randomly randomly."
            "Larger values introduce noise in the labels and make the classification task harder.",
        )

        self.parser.add_argument(
            "--class_sep",
            type=float,
            default=1.0,
            help="The factor multiplying the hypercube size."
            "Larger values spread out the clusters/classes and make the classification task easier.",
        )

        self.parser.add_argument(
            "--data_scale",
            type=int,
            default=None,
            help="To scale the synthetic data or not by default.",
        )

        # --- Additional Classification Options can be added above --- >>>
        # ------------------------------------------------------------

        # ------- Regression Options for Synthetic Data ------- >>>

        self.parser.add_argument(
            "--reg_informative",
            type=int,
            default=10,
            help="Number of informative features to create for synthetic data",
        )

        self.parser.add_argument(
            "--n_target",
            type=int,
            default=1,
            help="Number of target values to create for synthetic data",
        )

        self.parser.add_argument(
            "--reg_bias",
            type=float,
            default=0.0,
            help="The bias term in the underlying linear model",
        )

        self.parser.add_argument(
            "--effective_rank",
            type=int,
            default=None,
            help="If not None, the number of singular vectors to"
            "be random linear combinations of the informative features.",
        )

        self.parser.add_argument(
            "--reg_noise",
            type=float,
            default=0.0,
            help="The standard deviation of the gaussian noise applied to the output.",
        )

        self.parser.add_argument(
            "--reg_coef",
            type=bool,
            default=False,
            help="If True, the coefficients of the underlying linear model are returned.",
        )

        # --- Additional Regression Options can be added above --- >>>
        # ------------------------------------------------------------
