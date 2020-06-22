import numpy as np
import cv2

direction_delta = 0.01
def adjust_threshold(key, threshold, direction, kernel):

    print("key = ", key)
    if key == 116: # key "T"
        if threshold[0] > 0:
            threshold[0] = threshold[0] - 1

    if key == 121: # key "Y"
        if threshold[0] < threshold[1]:
            threshold[0] = threshold[0] + 1

    if key == 117: # key "U"
        if threshold[1] > threshold[0]:
            threshold[1] = threshold[1] - 1

    if key == 105: # key "I"
        if threshold[1] < 255:
            threshold[1] = threshold[1] + 1


    if key == 103: # key "G"
        if direction[0] > 0:
            direction[0] = direction[0] - direction_delta

    if key == 104: # key "H"
        if direction[0] < direction[1] - direction_delta:
            direction[0] = direction[0] + direction_delta

    if key == 106: # key "J"
        if direction[1] > direction[0] + direction_delta:
            direction[1] = direction[1] - direction_delta

    if key == 107: # key "K"
        if direction[1] < np.pi/2:
            direction[1] = direction[1] + direction_delta


    if key == 98: # key "B"
        if(kernel > 2):
            kernel = kernel - 2
    if key == 110: # key "N"
        if(kernel < 31):
            kernel = kernel + 2

    print(threshold)
    print(direction)
    print(kernel)

    return threshold, direction, kernel