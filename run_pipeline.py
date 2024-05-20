from feature_importance import feature_importance
from feature_importance.feature_importance_options import \
    FeatureImportanceOptions
from machine_learning import train
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from utils.logging_utils import Logger
from utils.utils import set_seed

ml_opt = MLOptions().parse()
fi_opt = FeatureImportanceOptions().parse()
seed = ml_opt.random_state
ml_logger = Logger(ml_opt.log_dir, ml_opt.experiment_name).make_logger()
fi_logger = Logger(fi_opt.log_dir, fi_opt.experiment_name).make_logger()


# Set seed for reproducibility
set_seed(seed)
data = DataBuilder(ml_opt, logger=ml_logger).ingest()
trained_models = train.run(ml_opt, data, ml_logger)
gloabl_importance_results, ensemble_results, local_importance_results = feature_importance.run(fi_opt, data, trained_models,fi_logger)

print(ensemble_results['Majority Vote'])



