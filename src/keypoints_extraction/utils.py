import numpy as np
from typing import List, Tuple
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from data import Person, person_from_keypoints_with_scores

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
}

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]


def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] | None = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
    """Draws landmarks and edges on the input image and return it.

    Args:
      image: The input RGB image.
      list_persons: The list of all "Person" entities to be visualize.
      keypoint_color: the colors in which the landmarks should be plotted.
      keypoint_threshold: minimum confidence score for a keypoint to be drawn.
      instance_threshold: minimum confidence score for a person to be drawn.

    Returns:
      Image with keypoints and edges.
    """
    for person in list_persons:
        if person.score < instance_threshold:
            continue

        keypoints = person.keypoints

        # Assign a color to visualize keypoints.
        if keypoint_color is None:
            if person.id is None:
                # If there's no person id, which means no tracker is enabled, use
                # a default color.
                person_color = (0, 255, 0)
            else:
                # If there's a person id, use different color for each person.
                person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
        else:
            person_color = keypoint_color

        # Draw all the landmarks
        for i in range(len(keypoints)):
            if keypoints[i].score >= keypoint_threshold:
                cv2.circle(image, keypoints[i].coordinate, 17, person_color, 4)

        # Draw all the edges
        for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                keypoints[edge_pair[0]].score > keypoint_threshold
                and keypoints[edge_pair[1]].score > keypoint_threshold
            ):
                cv2.line(
                    image,
                    keypoints[edge_pair[0]].coordinate,
                    keypoints[edge_pair[1]].coordinate,
                    edge_color,
                    15,
                )

    return image


def draw_prediction_on_image(image, person, close_figure=True, keep_input_size=False):
    """Draws the keypoint predictions on image.

    Args:
      image: An numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      person: A person entity returned from the MoveNet.SinglePose model.
      close_figure: Whether to close the plt figure after the function returns.
      keep_input_size: Whether to keep the size of the input image.

    Returns:
      An numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    # Draw the detection result on top of the image.
    image_np = visualize(image, [person], keypoint_color=None)

    # Plot the image with detection results.
    height, width, _ = image.shape
    aspect_ratio = float(width) / height
    display_height = 512
    display_width = int(display_height * aspect_ratio)
    resized_image = cv2.resize(image_np, (display_width, display_height))
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    _ = ax.imshow(resized_image)

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = tf.image.resize_with_pad(image_np, 256, 256).numpy()

    return image_np


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    return img


def detect(model, input_tensor: tf.Tensor) -> Person:
    """Runs detection on an input image.

    Args:
      input_tensor: A [height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
      A Person entity detected by the MoveNet.SinglePose.
    """
    image_height, image_width, _ = input_tensor.shape
    # Resize with pad to keep the aspect ratio and fit the expected size.
    input_tensor = tf.cast(
        tf.image.resize_with_pad(input_tensor, 256, 256), dtype=tf.int32
    )
    # Add a batch dimension.
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    keypoints_with_scores = model(input_tensor)["output_0"]
    # Reshape the keypoints_with_scores to be a [17, 3] tensor.
    keypoints_with_scores = tf.squeeze(keypoints_with_scores)
    return person_from_keypoints_with_scores(
        keypoints_with_scores.numpy(),
        image_height,
        image_width,
    )
