import os
import tensorflow_hub as hub

from keypoints_extraction.preprocessor import MoveNetPreprocessor


DATA_DIR = "data"
IMAGES_OUT_DIR = os.path.join(DATA_DIR, "poses_images_out")
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")

model = hub.load(
    "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
)
movenet = model.signatures["serving_default"]

preprocessor = MoveNetPreprocessor(
    model=movenet,
    images_in_folder=os.path.join(DATA_DIR, "original"),
    images_out_folder=IMAGES_OUT_DIR,
    csvs_out_path=CSVS_OUT,
    batch_size=4,
)

preprocessor.process()
