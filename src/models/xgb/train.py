import os
import sys

from xgboost import XGBClassifier

from dvclive.live import Live
import dvc.api


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.base_trainer import XGBTrainer
from models.evaluate import evaluate_with_cv
from models.utils import load_all_data

dvclive_path = os.path.join(parent_dir, "../dvclive/xgb")
models_dir = os.path.join(parent_dir, "../models/xgb/")
params = dvc.api.params_show()

# Load all data for cross-validation
X, y = load_all_data(os.path.join(parent_dir, "../data/processed"))

with Live(dvclive_path) as live:
    # Initialize model
    xgb = XGBClassifier(random_state=params["random_state"], eval_metric='logloss')

    # Initialize trainer
    trainer = XGBTrainer(
        model=xgb,
        model_name="xgb",
        dvclive_path=dvclive_path,
        models_dir=models_dir
    )

    # Run training pipeline
    model_path = os.path.join(models_dir, "xgb.json")
    trainer.run_training_pipeline(
        X=X,
        y=y,
        evaluate_fn=evaluate_with_cv,
        live=live,
        model_path=model_path
    )
