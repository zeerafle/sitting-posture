import os
import sys
import time
import polars as pl
from loguru import logger

from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SpineAlignmentFeatureExtractor
)
from feature_extraction.base import FeatureExtractionPipeline

# Configure Loguru
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_path, exist_ok=True)
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(os.path.join(log_path, "feature_extraction_{time}.log"), rotation="500 MB", retention="30 days", compression="zip", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

DATA_DIR = "data"
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "data_with_features.csv")

logger.info("Starting feature extraction process")
logger.debug(f"Input data path: {CSVS_OUT}")
logger.debug(f"Output path for enriched data: {FEATURES_OUT}")

# Load the keypoints data
start_time = time.time()
try:
    logger.info(f"Loading landmarks data from {CSVS_OUT}")
    landmarks_df = pl.read_csv(CSVS_OUT)
    load_time = time.time() - start_time
    rows, cols = landmarks_df.shape
    logger.success(f"Loaded landmarks dataframe with shape {rows}x{cols} in {load_time:.2f}s")
except Exception as e:
    logger.error(f"Failed to load landmarks data: {str(e)}")
    sys.exit(1)

# Create feature extraction pipeline
logger.info("Setting up feature extraction pipeline")
try:
    # Log information about each extractor
    logger.info("Initializing feature extractors")
    feature_extractors = [
        AngleFeatureExtractor(),
        DistanceFeatureExtractor(),
        SpineAlignmentFeatureExtractor()
    ]

    for extractor in feature_extractors:
        logger.debug(f"Added {extractor.__class__.__name__} to pipeline")

    pipeline = FeatureExtractionPipeline(feature_extractors)
    logger.debug(f"Pipeline initialized with {len(feature_extractors)} extractors")
except Exception as e:
    logger.error(f"Error initializing feature extraction pipeline: {str(e)}", exc_info=True)
    sys.exit(1)

# Extract all features
logger.info("Starting feature extraction")
extraction_start = time.time()
try:
    enriched_df = pipeline.extract_all(landmarks_df)
    extraction_time = time.time() - extraction_start

    # Get details about extracted features
    all_features = pipeline.get_all_feature_names()
    original_cols = landmarks_df.shape[1]
    new_cols = enriched_df.shape[1]
    added_cols = new_cols - original_cols

    logger.success(f"Feature extraction completed in {extraction_time:.2f}s")
    logger.info(f"Added {added_cols} new feature columns")
    logger.debug(f"Total columns in enriched dataframe: {new_cols}")
except Exception as e:
    logger.error(f"Error during feature extraction: {str(e)}", exc_info=True)
    sys.exit(1)

# Save the enriched dataset
logger.info(f"Saving enriched dataset to {FEATURES_OUT}")
save_start = time.time()
try:
    enriched_df.write_csv(FEATURES_OUT)
    save_time = time.time() - save_start
    file_size_mb = os.path.getsize(FEATURES_OUT) / (1024 * 1024)
    logger.success(f"Enriched data saved successfully in {save_time:.2f}s ({file_size_mb:.2f} MB)")
except Exception as e:
    logger.error(f"Error saving enriched dataset: {str(e)}")
    sys.exit(1)

# Log feature information
logger.info(f"Total features extracted: {len(all_features)}")
logger.debug("Extracted features:")
for i, feature in enumerate(all_features, 1):
    logger.debug(f"  {i}. {feature}")

total_time = time.time() - start_time
logger.success(f"Feature extraction process completed in {total_time:.2f}s")
