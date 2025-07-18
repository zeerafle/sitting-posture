import os
import tensorflow_hub as hub
import polars as pl

from keypoints_extraction.preprocessor import MoveNetPreprocessor
from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SymmetryFeatureExtractor,
    SpineAlignmentFeatureExtractor
)
from feature_extraction.view_features import ViewSpecificFeatureExtractor
from feature_extraction.base import FeatureExtractionPipeline


DATA_DIR = "data"
IMAGES_OUT_DIR = os.path.join(DATA_DIR, "poses_images_out")
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "data_with_features.csv")

# Step 1: Extract pose landmarks
print("Loading MoveNet model...")
model = hub.load(
    "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
)
movenet = model.signatures["serving_default"]

print("Processing images and extracting landmarks...")
preprocessor = MoveNetPreprocessor(
    model=movenet,
    images_in_folder=os.path.join(DATA_DIR, "original"),
    images_out_folder=IMAGES_OUT_DIR,
    csvs_out_path=CSVS_OUT,
    batch_size=4,
)

preprocessor.process()

# Step 2: Extract hand-engineered features
print("Extracting hand-engineered features...")
landmarks_df = pl.read_csv(CSVS_OUT)

# Create feature extraction pipeline
feature_extractors = [
    AngleFeatureExtractor(),
    DistanceFeatureExtractor(),
    SymmetryFeatureExtractor(),
    ViewSpecificFeatureExtractor(),
    SpineAlignmentFeatureExtractor()
]

pipeline = FeatureExtractionPipeline(feature_extractors)

# Extract all features
enriched_df = pipeline.extract_all(landmarks_df)

# Save the enriched dataset
enriched_df.write_csv(FEATURES_OUT)

print(f"Feature extraction complete! Data saved to {FEATURES_OUT}")
print(f"Total features extracted: {len(pipeline.get_all_feature_names())}")
print(f"Feature names: {pipeline.get_all_feature_names()}")
