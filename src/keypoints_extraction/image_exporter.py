import os
import cv2
import numpy as np
from loguru import logger
from keypoints_extraction.utils import draw_prediction_on_image

class ImageExporter:
    def __init__(self, out_folder: str):
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder

    def export(self, image: np.ndarray, person, image_name: str):
        overlay = draw_prediction_on_image(
            image.astype(np.uint8), person, close_figure=True, keep_input_size=True
        )
        frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(self.out_folder, image_name)
        cv2.imwrite(out_path, frame)
        logger.debug(f"Wrote overlay to {out_path}")
