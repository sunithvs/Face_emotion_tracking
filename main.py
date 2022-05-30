import math
from typing import Union, Tuple

import cv2
import numpy as np
import mediapipe as mp


def to_pixel_coordinates(x: float, y: float, width: int, height: int) -> Union[None, Tuple[int, int]]:
    return min(math.floor(x * width), width - 1), min(math.floor(y * height), height - 1)


def get_rect(image: np.ndarray, detection):
    box = detection.location_data.relative_bounding_box
    return [
        to_pixel_coordinates(point[0], point[1], image.shape[1], image.shape[0])
        for point in ((box.xmin, box.ymin), (box.xmin + box.width, box.ymin + box.height))
    ]


def get_faces():
    faces = []
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    img = cv2.imread("data/img_1.png")

    face_detection_results = face_detection.process(img[:, :, ::-1])

    if not face_detection_results.detections:
        return

    for face in face_detection_results.detections:
        (x1, y1), (x2, y2) = get_rect(img, face)

        faces.append(img[y1:y2, x1:x2])
    return faces


def main():
    print(get_faces())


if __name__ == "__main__":
    main()
