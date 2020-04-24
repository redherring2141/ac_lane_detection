#!/usr/bin/env python

import numpy as np
import pyscreenshot as ImageGrab
import cv2
import time
#import matplotlib as plt


#import cv2
#import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import pickle


#from calibrateCam import findPts, calibrate, calibrateCam
from transformImage import undistort, corners_unwarp
from thresholdColorGrad import threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset
from drawLane import draw_lane
#import classLine

'''
#Lab setting
X1 = 90
Y1 = 60
X2 = 990
Y2 = 900
FX = 0.5
FY = 0.5

'''

#3Secondz setting
X1 = 2000
Y1 = 310
X2 = 2900
Y2 = 830
FX = 1
FY = 1


nx = 9
ny = 6
ks = 17
th = (20,120)
w_win = 10
h_win = 92
margin = 25
font = cv2.FONT_HERSHEY_DUPLEX

if __name__ == '__main__':
    #screen = ImageGrab.grab()
    screen = ImageGrab.grab(bbox=(X1,Y1,X2,Y2))
    screen_x = screen.size[0] 
    screen_y = screen.size[1]
    screen_xoff = screen_x*0.095
    screen_yoff = 150

    warp_x = int(screen_x/3)
    warp_y = screen_y
    warp_size = (warp_x, warp_y)
    warp_xoff = screen_x/24
    warp_yoff = -50

    src = np.float32([(screen_x/2 - screen_xoff, screen_y*0.5),(screen_x/2 + screen_xoff, screen_y*0.5),
    (screen_x, screen_y*0.7), (0, screen_y*0.7)])


    dst = np.float32([(warp_x/2-warp_xoff,warp_y*0.85+warp_yoff),(warp_x/2+warp_xoff,warp_y*0.85+warp_yoff),
    (warp_x/2+warp_xoff,warp_y+warp_yoff), (warp_x/2-warp_xoff,warp_y+warp_yoff)])


    src2dstM = cv2.getPerspectiveTransform(src,dst)
    dst2srcM = cv2.getPerspectiveTransform(dst,src)
    prev_point = [0,0]


    while(True):
        #time.sleep(1)
        #screen_pil = ImageGrab.grab()
        screen_pil = ImageGrab.grab(bbox=(X1,Y1,X2,Y2))

        screen_np = np.array(screen_pil.convert('RGB'))
        # frame = cv2.cvtColor(cv2.resize(screen_np, (0,0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        img_org = cv2.resize(frame,(0,0), fx=FX, fy=FY)



        output = np.zeros((int(screen_y*FY*2), int(screen_x*FX),3),dtype='uint8')

        img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

        img_roi = mask_roi(img_org)

        
        img_hls_ch_s = hls_select(img_roi, ch='s', thresh=(30,80))*255

        img_grad_abx = abs_sobel_th(img_roi, orient='x', ksize=ks, thresh=(35,225))*255
        img_grad_aby = abs_sobel_th(img_roi, orient='y', ksize=ks, thresh=(35,225))*255
        img_grad_mag = mag_sobel_th(img_roi, ksize=ks, thresh=(35,115))*255
        img_grad_dir = dir_sobel_th(img_roi, ksize=7, thresh=(1.57,3.14))*255
        img_wlane_th = threshold(cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY), thresh=(125,255))*255

        img_combined = np.zeros_like(img_roi[:,:,0])
        img_combined[(img_grad_abx==255) | (img_hls_ch_s==255) | ((img_grad_mag==255) & (img_grad_dir==0)) | (img_wlane_th==255)]=255


        
        #img_bin = img_hls_ch_s
        img_bin = img_combined

        img_fin = np.zeros_like(img_org)
        img_fin[:,:,0] = img_bin
        img_fin[:,:,1] = img_bin
        img_fin[:,:,2] = img_bin
        #print(img_gray.shape)
        

        #img_fin = img_roi

        


        #img_fin = img_org
  
        output[0:int(screen_y*FY), 0:int(screen_x*FX)] = img_org
        output[int(screen_y*FY):int(screen_y*FY*2), 0:int(screen_x*FX)] = img_fin
        cv2.line(output, (0,int(screen_y)),(int(screen_x), int(screen_y)),(0,255,255),5)
   
        cv2.imshow('TEST_WINDOW', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


