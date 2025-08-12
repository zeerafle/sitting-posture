import os
import time
import random
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import polars as pl
from glob import glob

# Import components from existing modules
from keypoints_extraction.utils import load_image, detect
from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SpineAlignmentFeatureExtractor
)
from feature_extraction.base import FeatureExtractionPipeline
from prepare import landmarks_to_embedding
from data import BodyPart

# Constants
DATA_DIR = "data/original"
MODEL_DIR = "models/adaboost"
CLASSES = ["ergonomis", "non-ergonomis"]
VIEWS = ["front", "left", "right"]
SAMPLES_PER_CLASS_PER_VIEW = 3

def get_random_images():
    """Get random image paths for benchmarking."""
    selected_images = []

    for class_name in CLASSES:
        for view in VIEWS:
            view_dir = os.path.join(DATA_DIR, class_name, view)
            if not os.path.exists(view_dir):
                print(f"Warning: {view_dir} not found. Skipping.")
                continue

            images = glob(os.path.join(view_dir, "*.jpg")) + \
                     glob(os.path.join(view_dir, "*.jpeg")) + \
                     glob(os.path.join(view_dir, "*.png")) + \
                     glob(os.path.join(view_dir, "*.JPG")) + \
                     glob(os.path.join(view_dir, "*.JPEG")) + \
                     glob(os.path.join(view_dir, "*.PNG"))

            if len(images) < SAMPLES_PER_CLASS_PER_VIEW:
                print(f"Warning: Only {len(images)} images found in {view_dir}")
                selected = images
            else:
                selected = random.sample(images, SAMPLES_PER_CLASS_PER_VIEW)

            selected_images.extend([
                {
                    'path': img,
                    'class': class_name,
                    'view': view
                } for img in selected
            ])

    return selected_images

def process_image(image_info, model, feature_pipeline, scalers, classifiers):
    """Process a single image and time each step."""
    image_path = image_info['path']
    view = image_info['view']

    timing = {}

    # Step 1: Load image
    start_time = time.time()
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize_with_pad(image, 256, 256)
    timing['load_image'] = (time.time() - start_time) * 1000  # ms

    # Step 2: Extract keypoints
    start_time = time.time()
    person = detect(model, image)
    keypoints = np.array([
        [kp.coordinate.x, kp.coordinate.y, kp.score]
        for kp in person.keypoints
    ], dtype=np.float32)
    timing['extract_keypoints'] = (time.time() - start_time) * 1000  # ms

    # Step 3: Create landmarks dataframe
    start_time = time.time()
    # Create header names for the landmark coordinates
    list_name = [
        [bodypart.name + "_x", bodypart.name + "_y", bodypart.name + "_score"]
        for bodypart in BodyPart
    ]
    header_name = []
    for columns_name in list_name:
        header_name += columns_name

    # Create a single row dataframe with the keypoints
    landmarks_df = pl.DataFrame([keypoints.flatten()], schema=header_name)
    landmarks_df = landmarks_df.with_columns([
        pl.lit(image_path).alias('file_name'),
        pl.lit(image_info['class']).alias('class_name'),
        pl.lit(view).alias('view_type'),
    ])
    timing['create_dataframe'] = (time.time() - start_time) * 1000  # ms

    # Step 4: Feature engineering
    start_time = time.time()
    enriched_df = feature_pipeline.extract_all(landmarks_df)
    timing['feature_engineering'] = (time.time() - start_time) * 1000  # ms

    # Step 5: Landmark normalization (embedding)
    start_time = time.time()
    landmark_cols = [col for col in enriched_df.columns if
                    any(part in col for part in ["NOSE", "EYE", "EAR", "SHOULDER",
                                               "ELBOW", "WRIST", "HIP"])]

    # Get landmarks coordinates and scores for embedding
    landmarks_data = []
    for part in BodyPart:
        x = enriched_df.get_column(f"{part.name}_x")[0]
        y = enriched_df.get_column(f"{part.name}_y")[0]
        score = enriched_df.get_column(f"{part.name}_score")[0]
        landmarks_data.extend([x, y, score])

    # Create embedding
    embedding = landmarks_to_embedding(landmarks_data)
    timing['normalize_landmarks'] = (time.time() - start_time) * 1000  # ms

    # Step 6: Feature normalization
    start_time = time.time()
    feature_cols = [col for col in enriched_df.columns if col not in landmark_cols +
                   ['file_name', 'class_name', 'class_no', 'view_type']]

    features_data = enriched_df.select(feature_cols).to_numpy()
    normalized_features = scalers[view].transform(features_data)

    # Combine embedding with normalized features
    embedding_array = np.array(embedding).reshape(1, -1)
    combined_features = np.hstack((embedding_array, normalized_features))
    timing['normalize_features'] = (time.time() - start_time) * 1000  # ms

    # Step 7: Make prediction
    start_time = time.time()
    prediction = classifiers[view].predict(combined_features)
    timing['predict'] = (time.time() - start_time) * 1000  # ms

    # Total processing time
    timing['total'] = sum(timing.values())

    return timing

def main():
    print("Loading models...")
    # Load MoveNet model
    start_time = time.time()
    model = hub.load(
        "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
    )
    movenet = model.signatures["serving_default"]
    print(f"MoveNet model loaded in {(time.time() - start_time) * 1000:.2f} ms")

    # Load AdaBoost models and scalers
    classifiers = {}
    scalers = {}
    for view in VIEWS:
        # Load classifier
        model_path = os.path.join(MODEL_DIR, f"adaboost_{view}.joblib")
        classifiers[view] = joblib.load(model_path)

        # Load scaler
        scaler_path = os.path.join("data", "processed", view, "scalers", "feature_scaler.joblib")
        scalers[view] = joblib.load(scaler_path)

    # Create feature extraction pipeline
    feature_extractors = [
        AngleFeatureExtractor(),
        DistanceFeatureExtractor(),
        SpineAlignmentFeatureExtractor()
    ]
    feature_pipeline = FeatureExtractionPipeline(feature_extractors)

    # Get random images for benchmarking
    images = get_random_images()
    print(f"Selected {len(images)} images for benchmarking")

    # Process each image and collect timing information
    all_timings = []
    for i, image_info in enumerate(images, 1):
        print(f"Processing image {i}/{len(images)}: {os.path.basename(image_info['path'])}")
        try:
            timing = process_image(image_info, movenet, feature_pipeline, scalers, classifiers)
            all_timings.append(timing)
            print(f"  Total processing time: {timing['total']:.2f} ms")
        except Exception as e:
            print(f"  Error processing image: {str(e)}")

    if not all_timings:
        print("No images were successfully processed. Exiting.")
        return

    # Calculate average timings
    avg_timings = {key: sum(t[key] for t in all_timings) / len(all_timings)
                  for key in all_timings[0].keys()}

    # Print results
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Average processing time per frame: {avg_timings['total']:.2f} ms")
    print("\nBreakdown:")
    for step, time_ms in avg_timings.items():
        if step != 'total':
            print(f"- {step}: {time_ms:.2f} ms ({time_ms / avg_timings['total'] * 100:.1f}%)")

if __name__ == "__main__":
    main()
