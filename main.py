import math
from typing import Union, Tuple

import cv2
import numpy as np
import mediapipe as mp

# import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

img = cv2.imread("data/img_1.png")
img = img[:, :, ::-1]
img_cpy = img.copy()

face_detection_results = face_detection.process(img)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_rect(image: np.ndarray, detection):
    image_rows, image_cols, _ = image.shape

    location = detection.location_data

    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)

    return rect_start_point, rect_end_point


# print(face_detection_results.detections)
if face_detection_results.detections:

    for face_no, face in enumerate(face_detection_results.detections):
        print(f'FACE NUMBER: {face_no + 1}')

        box = face.location_data.relative_bounding_box

        (x1, y1), (x2, y2) = get_rect(img, face)

        print(box)
        print((x1, y1), (x2, y2), img.shape)

        # cv2.imshow("head", cv2.rectangle(img_cpy, (x1, y1), (x2, y2), color=(255,0,0)))
        cv2.imshow("crop", img_cpy[y1:y2, x1:x2])
        # cv2.imshow("box", )
        cv2.waitKey()
