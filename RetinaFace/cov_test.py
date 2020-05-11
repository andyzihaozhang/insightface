import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV

thresh = 0.8
mask_thresh = 0.2
scales = [640, 1080]

count = 1
gpuid = -1 # do not use GPU

detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

img = cv2.imread('mask_6_ppl.jpg')
print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])

im_scale = np.min([float(target_size)/float(im_size_min), float(max_size)/float(im_size_max)])
print('im_scale', im_scale)

scales = [im_scale]
flip = False

faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

if faces is not None:
    print('Found', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        face = faces[i]
        box = face[:4]
        face_prob = face[4]
        mask_prob = face[5]
        print("Face {}:\n bbox location = {}\n score = {}\n mask prob = {}".format(i+1, box, face_prob, mask_prob))

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        landmark = landmarks[i]
        for point in range(landmark.shape[0]):
            cv2.circle(img, (landmark[point][0], landmark[point][1]), 1, (0, 0, 255), 2)

    filename = 'out_mask_6_ppl.jpg'
    print('writing', filename)
    cv2.imwrite(filename, img)
