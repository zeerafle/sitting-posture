import time
from loguru import logger
from keypoints_extraction.utils import detect

class PoseDetector:
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def detect_batch(self, images: list):
        logger.debug(f"Detecting {len(images)} images")
        start = time.time()
        results = []
        for img in images:
            person = detect(self.model, img)
            min_score = min(k.score for k in person.keypoints)
            results.append(person if min_score >= self.threshold else None)
        logger.debug(f"Detection took {time.time() - start:.2f}s. Valid: {sum(p is not None for p in results)}/{len(results)}")
        return results
