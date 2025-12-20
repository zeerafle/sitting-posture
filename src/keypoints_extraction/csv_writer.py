import csv
import numpy as np

class CSVWriter:
    def __init__(self, csv_path: str):
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file, lineterminator="\n")

    def write(self, image_name: str, person, view_type: str):
        coords = np.array([[kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints])
        row = [image_name] + coords.flatten().astype(str).tolist() + [view_type]
        self._writer.writerow(row)

    def close(self):
        self._file.close()
