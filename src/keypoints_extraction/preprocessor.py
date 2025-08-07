import os
import sys
import tempfile
import csv
import tqdm

import tensorflow as tf
import numpy as np
import polars as pl
import cv2

from data import BodyPart
from keypoints_extraction.utils import load_image, draw_prediction_on_image, detect


class MoveNetPreprocessor(object):
    """Helper class to preprocess pose sample images for classification."""

    def __init__(
        self, model, images_in_folder, images_out_folder, csvs_out_path, batch_size=32
    ):
        """Creates a preprocessor to detection pose from images and save as CSV.

        Args:
          images_in_folder: Path to the folder with the input images.
          images_out_folder: Path to write the images overlay with detected landmarks.
          csvs_out_path: Path to write the CSV containing the detected landmark coordinates.
          batch_size: Number of images to process in a batch.
        """

        self._model = model
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._batch_size = batch_size
        self._messages = []

        # Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # Get list of pose classes and print image statistics
        self._pose_class_names = sorted(
            [
                n
                for n in os.listdir(self._images_in_folder)
                if not n.startswith(".")
                and os.path.isdir(os.path.join(self._images_in_folder, n))
            ]
        )

        # view types
        self._view_types = ["front", "left", "right"]

    def process(self, per_pose_class_limit=None, detection_threshold=0):
        """Preprocesses images in the given folder.
        Args:
          per_pose_class_limit: Number of images to load.
          detection_threshold: Only keep images with all landmark confidence score above this threshold.
        """
        for pose_class_name in self._pose_class_names:
            print(f"Preprocessing {pose_class_name}", file=sys.stderr)

            class_folder = os.path.join(self._images_in_folder, pose_class_name)

            # Process each view type
            for view_type in self._view_types:
                view_folder = os.path.join(class_folder, view_type)
                if not os.path.exists(view_folder):
                    print(
                        f"Warning: View folder {view_folder} not found. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                print(f"  Processing {view_type} view", file=sys.stderr)

                # Create output directory for this class and view
                images_out_folder = os.path.join(
                    self._images_out_folder, pose_class_name, view_type
                )
                if not os.path.exists(images_out_folder):
                    os.makedirs(images_out_folder)

                # Create CSV file for this class and view
                csv_out_path = os.path.join(
                    self._csvs_out_folder_per_class,
                    f"{pose_class_name}_{view_type}.csv",
                )

                with open(csv_out_path, "w") as csv_out_file:
                    csv_out_writer = csv.writer(
                        csv_out_file,
                        delimiter=",",
                        quoting=csv.QUOTE_MINIMAL,
                        lineterminator="\n",
                    )
                    image_paths = sorted(
                        [
                            os.path.join(view_folder, n)
                            for n in os.listdir(view_folder)
                            if not n.startswith(".")
                            and n.endswith((".JPG", ".jpg", ".png", ".jpeg"))
                        ]
                    )

                    if per_pose_class_limit is not None:
                        image_paths = image_paths[:per_pose_class_limit]

                    if not image_paths:
                        print(
                            f"  No images found in {view_folder}. Skipping.",
                            file=sys.stderr,
                        )
                        continue

                    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
                    dataset = dataset.map(
                        load_image, num_parallel_calls=tf.data.AUTOTUNE
                    )
                    dataset = dataset.batch(self._batch_size)
                    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

                    valid_image_count = 0
                    batch_start_idx = 0

                    for batch in tqdm.tqdm(
                        dataset, desc=f"{pose_class_name}-{view_type}"
                    ):
                        persons = self._detect_batch(batch, detection_threshold)
                        for i, person in enumerate(persons):
                            # Calculate actual index in image_paths
                            current_idx = batch_start_idx + i
                            if current_idx >= len(image_paths):
                                continue  # Skip if index is out of range

                            image_path = image_paths[current_idx]
                            image_name = os.path.basename(image_path)

                            if person is None:
                                self._messages.append(
                                    f"Skipped {image_path}. No pose was confidently detected."
                                )
                                continue

                            valid_image_count += 1
                            output_overlay = draw_prediction_on_image(
                                batch[i].numpy().astype(np.uint8),
                                person,
                                close_figure=True,
                                keep_input_size=True,
                            )
                            output_frame = cv2.cvtColor(
                                output_overlay, cv2.COLOR_RGB2BGR
                            )
                            cv2.imwrite(
                                os.path.join(images_out_folder, image_name),
                                output_frame,
                            )

                            pose_landmarks = np.array(
                                [
                                    [
                                        keypoint.coordinate.x,
                                        keypoint.coordinate.y,
                                        keypoint.score,
                                    ]
                                    for keypoint in person.keypoints
                                ],
                                dtype=np.float32,
                            )
                            coordinates = pose_landmarks.flatten().astype(str).tolist()
                            csv_out_writer.writerow(
                                [image_name] + coordinates + [view_type]
                            )

                        # Update the starting index for the next batch
                        batch_start_idx += len(persons)

                    if not valid_image_count:
                        print(
                            f'No valid images found for the "{pose_class_name}" class, "{view_type}" view.',
                            file=sys.stderr,
                        )

        # Print the error message collected during preprocessing.
        if self._messages:
            print("\n".join(self._messages))

        # Combine all per-class and per-view CSVs into a single output
        all_landmarks_df = self._all_landmarks_as_dataframe()
        all_landmarks_df.write_csv(self._csvs_out_path)

    def _detect_batch(self, batch, detection_threshold):
        """Detects poses in a batch of images."""
        persons = []
        for image in batch:
            person = detect(self._model, image)
            if (
                min([keypoint.score for keypoint in person.keypoints])
                < detection_threshold
            ):
                persons.append(None)
            else:
                persons.append(person)
        return persons

    def _all_landmarks_as_dataframe(self):
        """Combines all CSVs into a single dataframe with class and view information."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            for view_type in self._view_types:
                csv_out_path = os.path.join(
                    self._csvs_out_folder_per_class, f"{class_name}_{view_type}.csv"
                )
                if not os.path.exists(csv_out_path):
                    continue

                per_class_df = pl.read_csv(csv_out_path, has_header=False)
                if len(per_class_df) == 0:
                    continue

                # Add the labels
                # Note: the view_type is already in the last column of the CSV
                per_class_df = per_class_df.with_columns(
                    [
                        pl.lit(class_index).alias("class_no"),
                        pl.lit(class_name).alias("class_name"),
                    ]
                )

                # Add the path structure to the filename
                folder_name = pl.lit(os.path.join(class_name, view_type, ""))
                original_filename = pl.col(per_class_df.columns[0]).cast(pl.Utf8)
                new_filename = folder_name + original_filename
                per_class_df = per_class_df.with_columns(
                    [new_filename.alias(per_class_df.columns[0])]
                )

                if total_df is None:
                    total_df = per_class_df
                else:
                    total_df = pl.concat([total_df, per_class_df])

        # Create header names for the landmark coordinates
        list_name = [
            [bodypart.name + "_x", bodypart.name + "_y", bodypart.name + "_score"]
            for bodypart in BodyPart
        ]
        header_name = []
        for columns_name in list_name:
            header_name += columns_name

        # Create the complete header list - the view_type column already exists from the CSV
        header_name = (
            ["file_name"] + header_name + ["view_type", "class_no", "class_name"]
        )

        # Map old column names to new header names
        header_map = {
            total_df.columns[i]: header_name[i] for i in range(len(header_name))
        }
        total_df = total_df.rename(header_map)

        return total_df
