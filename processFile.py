#!/usr/bin/env python


from moviepy.editor import VideoFileClip
#import classLine
import datetime as dt
import time
import numpy as np
import cv2

from thresholdColorGrad import threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset

file_path = "../ac_laguna_mx5_2.mp4"
cap = cv2.VideoCapture(file_path)
threshold = [0 , 68]
kernel = 9
direction = [0.7, 1.3]
direction_delta = 0.01

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        key = cv2.waitKey(1000)
        print("key = ", key)
        if key == 116: # key "T"
            if threshold[0] > 0:
                threshold[0] = threshold[0] - 1
            if direction[0] > 0:
                direction[0] = direction[0] - direction_delta

        if key == 121: # key "Y"
            if threshold[0] < threshold[1]:
                threshold[0] = threshold[0] + 1

            if direction[0] < direction[1] - direction_delta:
                direction[0] = direction[0] + direction_delta

        if key == 117: # key "U"
            if threshold[1] > threshold[0]:
                threshold[1] = threshold[1] - 1

            if direction[1] > direction[0] + direction_delta:
                direction[1] = direction[1] - direction_delta

        if key == 105: # key "I"
            if threshold[1] < 255:
                threshold[1] = threshold[1] + 1

            if direction[1] < np.pi/2:
                direction[1] = direction[1] + direction_delta

        if key == 49: # key "End"
            if(kernel > 2):
                kernel = kernel - 2
        if key == 51: # key "PgDn"
            if(kernel < 31):
                kernel = kernel + 2

        #if key == 27: # ESC
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        img_roi = mask_roi(img)
        img_bin = hls_select(img_roi, ch='s', thresh=threshold)
        cv2.imshow("test", img_bin*255)
        print(threshold)
        print(direction)
        print(kernel)


    else:
        break

cap.release()
cv2.destroyAllWindows()