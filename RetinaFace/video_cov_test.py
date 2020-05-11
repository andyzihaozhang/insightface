import cv2
import sys
import numpy as np
import time
import os
import glob
from retinaface_cov import RetinaFaceCoV

thresh = 0.8
mask_thresh = 0.5
scales = [0.5]
gpu_id = -1  # do not use GPU

detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpu_id, 'net3l')

vc = cv2.VideoCapture(0)

loop = 0
start = time.time()

while True:
    loop += 1

    ret, frame = vc.read()
    faces, landmarks = detector.detect(frame, thresh, scales=scales)

    if faces is not None:
        text = "{} face(s) found".format(faces.shape[0])
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for face, landmark in zip(faces, landmarks):
            bbox = face[:4]
            # face_prob = face[4]
            mask_prob = face[5]

            # green bounding box for people wearing mask and red for people not wearing mask
            color = (0, 0, 255)
            if mask_prob > mask_thresh:
                color = (0, 255, 0)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            for x, y in landmark:
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # end timer
        end = time.time()
        print("Average fps: ", loop/(end-start))
        break

# Release handle to the web camera
vc.release()
cv2.destroyAllWindows()
