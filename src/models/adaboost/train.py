import os
import sys

from sklearn.ensemble import AdaBoostClassifier

from dvclive.live import Live
import dvc.api


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.base_trainer import AdaBoostTrainer
from models.evaluate import evaluate_with_cv
from models.utils import load_all_data

dvclive_path = os.path.join(parent_dir, "../dvclive/adaboost")
models_dir = os.path.join(parent_dir, "../models/adaboost/")
params = dvc.api.params_show()

# Load all data for cross-validation
X, y = load_all_data(os.path.join(parent_dir, "../data/processed"))

with Live(dvclive_path) as live:
    # Initialize model
    adaboost = AdaBoostClassifier(random_state=params["random_state"])

    # Initialize trainer
    trainer = AdaBoostTrainer(
        model=adaboost,
        model_name="adaboost",
        dvclive_path=dvclive_path,
        models_dir=models_dir
    )

    # Run training pipeline
    model_path = os.path.join(models_dir, "adaboost.joblib")
    trainer.run_training_pipeline(
        X=X,
        y=y,
        evaluate_fn=evaluate_with_cv,
        live=live,
        model_path=model_path
    )
