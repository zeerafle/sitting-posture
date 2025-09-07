import os
import sys
import time
import pandas as pd
from loguru import logger

from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SpineAlignmentFeatureExtractor
)

# Configure Loguru
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_path, exist_ok=True)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    os.path.join(log_path, "feature_extraction_{time}.log"),
    rotation="500 MB",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

DATA_DIR = "data"
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "data_with_features.csv")


def _compute_row_features(row, angle_ex, dist_ex, spine_ex) -> dict:
    # Angle features
    features = {
        "shoulder_angle": angle_ex._calculate_shoulder_angle(row),
        "hip_angle": angle_ex._calculate_hip_angle(row),
    }

    aL_A, aL_B, aL_C = angle_ex._calculate_SEWAngleABC_left(row)
    aR_A, aR_B, aR_C = angle_ex._calculate_SEWAngleABC_right(row)
    features.update({
        "sewangle_a_left": aL_A,
        "sewangle_b_left": aL_B,
        "sewangle_c_left": aL_C,
        "sewangle_a_right": aR_A,
        "sewangle_b_right": aR_B,
        "sewangle_c_right": aR_C,
        "phi1": angle_ex._calculate_phi1_angle(row),
        "phi2": angle_ex._calculate_phi2_angle(row),
        "phi3": angle_ex._calculate_phi3_angle(row),
        "phi4": angle_ex._calculate_phi4_angle(row),
        "phi5": angle_ex._calculate_phi5_angle(row),
    })

    # Distance features
    features.update({
        "torso_length": dist_ex._torso_length(row),
        "nose_to_shoulder_left_distance": dist_ex._nose_to_shoulder_left_distance(row),
        "nose_to_shoulder_right_distance": dist_ex._nose_to_shoulder_right_distance(row),
        "shoulder_to_elbow_left_distance": dist_ex._shoulder_to_elbow_left_distance(row),
        "shoulder_to_elbow_right_distance": dist_ex._shoulder_to_elbow_right_distance(row),
        "elbow_to_wrist_left_distance": dist_ex._elbow_to_wrist_left_distance(row),
        "elbow_to_wrist_right_distance": dist_ex._elbow_to_wrist_right_distance(row),
        "wrist_to_shoulder_left_distance": dist_ex._wrist_to_shoulder_left_distance(row),
        "wrist_to_shoulder_right_distance": dist_ex._wrist_to_shoulder_right_distance(row),
        "shoulder_to_mid_left_distance": dist_ex._shoulder_to_mid_left_distance(row),
        "shoulder_to_mid_right_distance": dist_ex._shoulder_to_mid_right_distance(row),
        "mid_to_middle_hip_distance": dist_ex._mid_to_middle_hip_distance(row),
        "middle_hip_to_shoulder_left_distance": dist_ex._middle_hip_to_shoulder_left_distance(row),
        "middle_hip_to_shoulder_right_distance": dist_ex._middle_hip_to_shoulder_right_distance(row),
        "nose_to_middle_shoulder_distance": dist_ex._nose_to_middle_shoulder_distance(row),
    })

    # Spine alignment features
    features.update({
        "t_tl_diff_y": abs(
            (row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2
            - (  # TL midpoint between T and L (0.5)
                ((row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2)
                + (
                    (row.get("LEFT_HIP_y", 0) + row.get("RIGHT_HIP_y", 0)) / 2
                    - (row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2
                ) * 0.5
            )
        ),
        "tl_l_diff_y": abs(
            (  # TL midpoint between T and L (0.5)
                ((row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2)
                + (
                    (row.get("LEFT_HIP_y", 0) + row.get("RIGHT_HIP_y", 0)) / 2
                    - (row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2
                ) * 0.5
            )
            - ((row.get("LEFT_HIP_y", 0) + row.get("RIGHT_HIP_y", 0)) / 2)
        ),
        "t_l_diff_y": abs(
            (row.get("LEFT_SHOULDER_y", 0) + row.get("RIGHT_SHOULDER_y", 0)) / 2
            - ((row.get("LEFT_HIP_y", 0) + row.get("RIGHT_HIP_y", 0)) / 2)
        ),
    })

    return features


def main():
    logger.info("Starting feature extraction process")
    logger.debug(f"Input data path: {CSVS_OUT}")
    logger.debug(f"Output path for enriched data: {FEATURES_OUT}")

    # Load the keypoints data with pandas
    start_time = time.time()
    try:
        logger.info(f"Loading landmarks data from {CSVS_OUT}")
        landmarks_df = pd.read_csv(CSVS_OUT)
        load_time = time.time() - start_time
        rows, cols = landmarks_df.shape
        logger.success(f"Loaded landmarks dataframe with shape {rows}x{cols} in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load landmarks data: {str(e)}")
        sys.exit(1)

    # Initialize extractors (we will call their helper methods on pandas rows)
    logger.info("Setting up feature extractors")
    try:
        angle_extractor = AngleFeatureExtractor()
        distance_extractor = DistanceFeatureExtractor()
        spine_extractor = SpineAlignmentFeatureExtractor()
        logger.debug("Feature extractors initialized")
    except Exception as e:
        logger.error(f"Error initializing feature extractors: {str(e)}", exc_info=True)
        sys.exit(1)

    # Extract all features using pandas
    logger.info("Starting feature extraction (pandas)")
    extraction_start = time.time()
    try:
        features_df = landmarks_df.apply(
            lambda row: _compute_row_features(row, angle_extractor, distance_extractor, spine_extractor),
            axis=1,
            result_type="expand"
        )

        # Combine original landmarks with new features
        enriched_df = pd.concat([landmarks_df, features_df], axis=1)
        extraction_time = time.time() - extraction_start

        # Feature names from extractors
        all_features = (
            angle_extractor.get_feature_names()
            + distance_extractor.get_feature_names()
            + spine_extractor.get_feature_names()
        )

        original_cols = landmarks_df.shape[1]
        new_cols = enriched_df.shape[1]
        added_cols = new_cols - original_cols

        logger.success(f"Feature extraction completed in {extraction_time:.2f}s")
        logger.info(f"Added {added_cols} new feature columns")
        logger.debug(f"Total columns in enriched dataframe: {new_cols}")
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}", exc_info=True)
        sys.exit(1)

    # Save the enriched dataset with pandas
    logger.info(f"Saving enriched dataset to {FEATURES_OUT}")
    save_start = time.time()
    try:
        enriched_df.to_csv(FEATURES_OUT, index=False)
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


if __name__ == "__main__":
    main()
