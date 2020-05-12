#!/usr/bin/env python


from moviepy.editor import VideoFileClip
import classLine
import datetime as dt
import time
import cv2

from adjustThreshold import adjust_threshold


# Create instance of class
frame = classLine.Line()

cap = cv2.VideoCapture("../../ac_laguna_mx5_2.mp4")

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        adjust_threshold(img)
        
        #cv2.imshow('Output', frame.process_frame(img))
        #cv2.imshow('Output', img)
        #cv2.imwrite('./sample_1.jpg', img)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break
        
    else:
        break

cap.release()
cv2.destroyAllWindows()