#!/usr/bin/env python

import numpy as np
import cv2

image = cv2.imread("../../sample_1.jpg")

threshold = [0 , 68]
kernel = 9
direction = [0.7, 1.3]
direction_delta = 0.01

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
#cv2.waitKey(0)






while True:
    key = cv2.waitKey(1000)
    print("key = ", key)
    if key == 80: # key "Home"
        if threshold[0] > 0:
            threshold[0] = threshold[0] - 1
        if direction[0] > 0:
            direction[0] = direction[0] - direction_delta

    if key == 81: # key "PgUp"
        if threshold[0] < threshold[1]:
            threshold[0] = threshold[0] + 1

        if direction[0] < direction[1] - direction_delta:
            direction[0] = direction[0] + direction_delta

    if key == 86: # left arrow
        if threshold[1] > threshold[0]:
            threshold[1] = threshold[1] - 1

        if direction[1] > direction[0] + direction_delta:
            direction[1] = direction[1] - direction_delta

    if key == 85: # right arrow
        if threshold[1] < 255:
            threshold[1] = threshold[1] + 1

        if direction[1] < np.pi/2:
            direction[1] = direction[1] + direction_delta

    if key == 87: # key "End"
        if(kernel > 2):
            kernel = kernel - 2
    if key == 51: # key "PgDn"
        if(kernel < 31):
            kernel = kernel + 2

    if key == 27: # ESC
        break

    

    binary = np.zeros_like(image)
    binary[(gray >= threshold[0]) & (gray <= threshold[1])] = 1
    cv2.imshow("test", 255*binary)
    print(threshold)