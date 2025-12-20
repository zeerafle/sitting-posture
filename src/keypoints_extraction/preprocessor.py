import os
import tempfile
import time
import tqdm
import tensorflow as tf
import numpy as np
from loguru import logger

from keypoints_extraction.pose_detector import PoseDetector
from keypoints_extraction.image_exporter import ImageExporter
from keypoints_extraction.csv_writer import CSVWriter
from keypoints_extraction.dataframe_combiner import DataFrameCombiner


class MoveNetPreprocessor:
    def __init__(
        self, model, images_in_folder, images_out_folder, csvs_out_path, batch_size=32
    ):
        logger.info("Initializing MoveNetPreprocessor")
        self._model = model
        self._in = images_in_folder
        self._out = images_out_folder
        self._csv_out = csvs_out_path
        self._batch = batch_size

        # temp folder for per-class CSVs
        self._csv_tmp = tempfile.mkdtemp()
        self._classes = sorted(
            n
            for n in os.listdir(self._in)
            if not n.startswith(".") and os.path.isdir(os.path.join(self._in, n))
        )
        self._views = ["front", "left", "right"]

    def process(self, per_pose_class_limit=None, detection_threshold=0):
        start_total = time.time()
        detector = PoseDetector(self._model, detection_threshold)

        for cls in self._classes:
            for view in self._views:
                view_in = os.path.join(self._in, cls, view)
                if not os.path.isdir(view_in):
                    logger.warning(f"{view_in} missing, skip")
                    continue

                imgs = sorted(
                    p for p in os.listdir(view_in) if p.lower().endswith((".jpg", ".png"))
                )
                if per_pose_class_limit:
                    imgs = imgs[:per_pose_class_limit]

                imgs = [tf.io.read_file(os.path.join(view_in, f)) for f in imgs]
                dataset = (
                    tf.data.Dataset.from_tensor_slices(imgs)
                    .map(lambda x: tf.io.decode_image(x), num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self._batch)
                    .prefetch(tf.data.AUTOTUNE)
                )

                out_folder = os.path.join(self._out, cls, view)
                img_exp = ImageExporter(out_folder)
                csv_path = os.path.join(self._csv_tmp, f"{cls}_{view}.csv")
                writer = CSVWriter(csv_path)

                for batch in tqdm.tqdm(dataset, desc=f"{cls}-{view}"):
                    np_imgs = batch.numpy()
                    persons = detector.detect_batch(np_imgs)
                    for i, p in enumerate(persons):
                        if p is None:
                            continue
                        name = os.path.basename(batch[i].numpy())
                        img_exp.export(np_imgs[i], p, name)
                        writer.write(name, p, view)

                writer.close()

        combiner = DataFrameCombiner(self._csv_tmp, self._classes, self._views)
        combiner.combine(self._csv_out)
        logger.success(f"All done in {time.time()-start_total:.2f}s")
