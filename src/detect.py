#!/usr/bin/env python3

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# loading haarcascade_frontalface_default.xml
face_model = cv2.CascadeClassifier(
    '../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml')
model = load_model('masknet-sigmoid.h5')
mask_label = {0: 'MASK', 1: 'NO MASK'}
color_label = {0: (0, 255, 0), 1: (255, 0, 0)}


def detect():
    # Read in an image
    # img = cv2.imread(f'../input/face-mask-detection/images/maksssksksss{image_id}.png')
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    # Detect faces
    faces = face_model.detectMultiScale(gray, minSize=(25, 25))
    new_img = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)  # colored output image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = new_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        color = color_label[mask_result.argmax()]
        cv2.putText(
            img,
            mask_label[mask_result.argmax()],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.imshow('Video', img)


video_capture = cv2.VideoCapture(0)
while True:
    detect()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
