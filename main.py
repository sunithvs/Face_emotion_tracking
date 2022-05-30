import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
img = cv2.imread("data/img_1.png")

face_detection_results = face_detection.process(img[:, :, ::-1])

img_copy = img[:, :, ::-1].copy()
# print(face_detection_results.detections)
if face_detection_results.detections:

    for face_no, face in enumerate(face_detection_results.detections):
        # print(face_no, face)
        print(f'FACE NUMBER: {face_no + 1}')
        print('==============================')

        print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')

        face_data = face.location_data

        print(f'nFACE BOUNDING BOX:n{face_data.relative_bounding_box}')

        for i in range(2):
            print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
            print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')

img_copy = img[:, :, ::-1].copy()

if face_detection_results.detections:

    for face_no, face in enumerate(face_detection_results.detections):
        mp_drawing.draw_detection(image=img_copy, detection=face,
                                  keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                               thickness=2,
                                                                               circle_radius=2))
fig = plt.figure(figsize=[10, 10])

plt.title("Resultant Image")
plt.axis('off')
plt.imshow(img_copy)
plt.show()
