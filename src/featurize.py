import os
import polars as pl

from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SpineAlignmentFeatureExtractor
)
from feature_extraction.base import FeatureExtractionPipeline


DATA_DIR = "data"
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "data_with_features.csv")

print("Extracting hand-engineered features...")
landmarks_df = pl.read_csv(CSVS_OUT)

# Create feature extraction pipeline
feature_extractors = [
    AngleFeatureExtractor(),
    DistanceFeatureExtractor(),
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
