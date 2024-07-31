# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module contains the data types used in pose estimation."""

import enum
from typing import List, NamedTuple, Tuple

import numpy as np


class BodyPart(enum.Enum):
    """Enum representing human body keypoints detected by pose estimation models."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12


class Point(NamedTuple):
    """A point in 2D space."""

    x: float
    y: float


class KeyPoint(NamedTuple):
    """A detected human keypoint."""

    body_part: BodyPart
    coordinate: Point
    score: float


class Person(NamedTuple):
    """A pose detected by a pose estimation model."""

    keypoints: List[KeyPoint]
    score: float
    id: int | None = None


def unnormalize_keypoints_from_padded(
    keypoints_x: np.ndarray,
    keypoints_y: np.ndarray,
    padded_height: int,
    padded_width: int,
    original_height: float,
    original_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unnormalizes keypoints from a padded resized image back to the original image size.

    Args:
      keypoints_x: A numpy array with shape [17] containing normalized keypoints of x.
      keypoints_y: A numpy array with shape [17] containing normalized keypoints of y.
      padded_height: The height of the padded image.
      padded_width: The width of the padded image.
      original_height: The original height of the image.
      original_width: The original width of the image.

    Returns:
      Two numpy array with shape [17] containing keypoints in the original image size.
    """
    # Calculate the scale used for padding
    scale = min(padded_height / original_height, padded_width / original_width)

    # Calculate the dimensions of the cropped image
    crop_height = int(original_height * scale)
    crop_width = int(original_width * scale)

    # Calculate the offsets for cropping
    offset_height = (padded_height - crop_height) // 2
    offset_width = (padded_width - crop_width) // 2

    # Adjust the keypoints to remove the padding
    keypoints_x = (keypoints_x * padded_width - offset_width) / scale
    keypoints_y = (keypoints_y * padded_height - offset_height) / scale

    return keypoints_x, keypoints_y


def person_from_keypoints_with_scores(
    keypoints_with_scores: np.ndarray,
    image_height: float,
    image_width: float,
    keypoint_score_threshold: float = 0.1,
) -> Person:
    """Creates a Person instance from single pose estimation model output.

    Args:
      keypoints_with_scores: Output of the TFLite pose estimation model. A numpy
        array with shape [17, 3]. Each row represents a keypoint: [y, x, score].
      image_height: height of the image in pixels.
      image_width: width of the image in pixels.
      keypoint_score_threshold: Only use keypoints with above this threshold to
        calculate the person average score.

    Returns:
      A Person instance.
    """

    # only take the first 13 keypoints (legs not included)
    kpts_x = keypoints_with_scores[:13, 1]
    kpts_y = keypoints_with_scores[:13, 0]
    scores = keypoints_with_scores[:13, 2]

    # Convert keypoints to the input image coordinate system.
    keypoints = []
    ori_kpts_x, ori_kpts_y = unnormalize_keypoints_from_padded(
        kpts_x, kpts_y, 256, 256, image_height, image_width
    )
    for i in range(scores.shape[0]):
        keypoints.append(
            KeyPoint(
                BodyPart(i), Point(int(ori_kpts_x[i]), int(ori_kpts_y[i])), scores[i]
            )
        )

    # Calculate person score by averaging keypoint scores.
    scores_above_threshold = list(
        filter(lambda x: x > keypoint_score_threshold, scores)
    )
    person_score = np.average(scores_above_threshold)

    return Person(keypoints, person_score)


class Category(NamedTuple):
    """A classification category."""

    label: str
    score: float
