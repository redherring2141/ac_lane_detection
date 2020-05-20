#!/usr/bin/env python


from moviepy.editor import VideoFileClip
#import classLine
import datetime as dt
import time
import numpy as np
import cv2

from adjustThreshold import adjust_threshold
from thresholdColorGrad import threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset

file_path = "../ac_laguna_mx5_2.mp4"
cap = cv2.VideoCapture(file_path)
threshold = [0 , 68]
kernel = 9
direction = [0.7, 1.3]

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        key = cv2.waitKey(10)
        threshold, direction, kernel = adjust_threshold(key, threshold, direction, kernel)

        #if key == 27: # ESC
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


        #img_roi = mask_roi(img)
        img_roi = img[484:812,:]
        #img_hls = hls_select(img_roi, ch='s', thresh=threshold)
        #img_grad_abx = abs_sobel_th(img_roi, orient='x', ksize=kernel, thresh=threshold)
        #img_grad_aby = abs_sobel_th(img_roi, orient='y', ksize=kernel, thresh=threshold)
        img_grad_mag = mag_sobel_th(img_roi, ksize=kernel, thresh=threshold)
        #img_w
        cv2.imshow("test", img_grad_mag*255)



    else:
        break

cap.release()
cv2.destroyAllWindows()