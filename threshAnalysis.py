#!/usr/bin/env python


from moviepy.editor import VideoFileClip
#import classLine
import datetime as dt
import time
import numpy as np
import cv2

import matplotlib.pyplot as plt

from adjustThreshold import adjust_threshold
from thresholdColorGrad import combined_color, line_rgw_bin, line_wr_bin, combined_sobels, rgb2gray, rgb2luv, rgb2lab, rgb2hls, hls_wy_bin, threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import draw_hist, mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset
from transformImage import undistort, corners_unwarp

#file_path = "../../ac_laguna_mx5_2.mp4"
#file_path = "../../ac_inje_86_2.mp4"
file_path = "../../ac_inje_lancer_1_2.mp4"
cap = cv2.VideoCapture(file_path)
thresh = [64, 255]#hls
kernel = 7
direction = [0.7, 1.3]
#mode = 'adjust'
mode = 'fixed'
#crop = [484,812,0,1280]#laguna seca - top, bottom, left, right
#crop = [240,600,0,1280]#inje_86 - top, bottom, left, right
#[h_img, w_img] = [360, 1280]#inje_86
crop = [60,310,220,1060]#inje_lancer - top, bottom, left, right
[h_cropped, w_cropped] = [crop[1]-crop[0], crop[3]-crop[2]]#inje_lancer
resize_rate = 0.5
[h_resized, w_resized] = [int(h_cropped*resize_rate), int(w_cropped*resize_rate)]
[h_src, w_src] = [60,130]
[h_dst, w_dst] = [20,60]
[h_win, w_win] = [10,30]


font = cv2.FONT_HERSHEY_DUPLEX

if (cap.isOpened() == False):
    print("Error opening video stream or file")

if (mode == 'adjust'):
    guide1 = "[Adjust mode]"
    guide2 = "TY:-/+ lower threshold(0~upper), UI: -/+ upper threshold(lower~255)"
    guide3 = "GH:-/+ lower direction(-pi~upper), JK:-/+ upper direction(lower~pi)"
    guide4 = "BN:-/+ kernel size"
else:
    guide1 = "[Adjust mode]"
    guide2 = "TY:-/+ lower threshold(0~upper), UI: -/+ upper threshold(lower~255)"
    guide3 = "GH:-/+ lower direction(-pi~upper), JK:-/+ upper direction(lower~pi)"
    guide4 = "BN:-/+ kernel size"




while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:

        if (mode == 'adjust'):
            key = cv2.waitKey(10)
            thresh, direction, kernel = adjust_threshold(key, thresh, direction, kernel)

            thresh_hls = thresh
            #direction_hls = direction
            #kernel_hls = kernel

            thresh_lab = thresh
            #direction_lab = direction
            #kernel_lab = kernel

            thresh_luv = thresh
            #direction_luv = direction
            #kernel_luv = kernel

            thresh_abx = thresh
            #direction_abx = direction
            kernel_abx = kernel

            thresh_aby = thresh
            #direction_aby = direction
            kernel_aby = kernel

            thresh_mag = thresh
            #direction_mag = direction
            kernel_mag = kernel

            #thresh_dir = thresh
            direction_dir = direction
            kernel_dir = kernel
            guide1 = "[Adjust mode]"
            guide2 = "TY:-/+ lower threshold(0~upper), UI: -/+ upper threshold(lower~255)"
            guide3 = "GH:-/+ lower direction(-pi~upper), JK:-/+ upper direction(lower~pi)"
            guide4 = "BN:-/+ kernel size    "

        else:
            thresh_hls = (64,255)
            #direction_hls = direction
            #kernel_hls = kernel

            thresh_lab = (94,255)
            #direction_lab = direction
            #kernel_lab = kernel

            thresh_luv = (92,255)
            #direction_luv = direction
            #kernel_luv = kernel

            thresh_abx = (64,255)
            #direction_abx = direction
            kernel_abx = 9

            thresh_aby = (64,255)
            #direction_aby = direction
            kernel_aby = 7

            thresh_mag = (64,255)
            #direction_mag = direction
            kernel_mag = 7

            #thresh_dir = thresh
            direction_dir = (np.pi/4, np.pi/2)
            kernel_dir = 11

            guide1 = "[Fixed mode]"
            guide2 = "Color: "
            guide3 = "Gradient "
            guide4 = "Kernel: "



        #if key == 27: # ESC
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


        img_roi = img[crop[0]:crop[1],crop[2]:crop[3]]
        img_resized = cv2.resize(img_roi, (w_resized,h_resized), interpolation = cv2.INTER_AREA)
        img_blurred = cv2.GaussianBlur(img_resized,(kernel,kernel),25)
        img_hls = rgb2hls(img_blurred)[:,:,2]
        img_lab = rgb2lab(img_blurred)[:,:,0]
        img_color = combined_color(img_blurred)
        img_grad_abx = abs_sobel_th(img_color, orient='x', ksize=kernel_abx, thresh=thresh_abx)
        img_grad_aby = abs_sobel_th(img_color, orient='y', ksize=kernel_aby, thresh=thresh_aby)
        img_grad_mag = mag_sobel_th(img_color, ksize=kernel_mag, thresh=thresh_mag)
        img_combined = combined_sobels(img_grad_abx, img_grad_aby, img_grad_mag, img_hls, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))
        img_wr = line_wr_bin(img_blurred)
        img_bin = np.zeros_like(img_combined)
        img_bin[(img_combined == 1) | (img_wr == 1)] = 1

        img_fin = img_bin

        img_test = np.zeros((h_resized*6+100,w_resized*3,3),np.uint8)
        img_test[h_resized*0:h_resized*1,w_resized*0:w_resized*1,:] = img_resized
        img_test[h_resized*0:h_resized*1,w_resized*1:w_resized*2,:] = img_blurred
        img_test[h_resized*0:h_resized*1,w_resized*2:w_resized*3,:] = cv2.cvtColor(combined_color(img_blurred)*255, cv2.COLOR_GRAY2RGB)

        '''
        #RGB analysis - R th = 125, W th = 190 , G th = 140, Gray th = 0~124
        img_test[h_resized*1:h_resized*2,w_resized*0:w_resized*1,:]=cv2.cvtColor(img_blurred[:,:,0],cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*0:w_resized*1,:]=cv2.cvtColor(img_blurred[:,:,1],cv2.COLOR_GRAY2RGB)
        #img_test[h_resized*3:h_resized*4,w_resized*0:w_resized*1,:]=cv2.cvtColor(img_blurred[:,:,2],cv2.COLOR_GRAY2RGB)
        img_test[h_resized*3:h_resized*4,w_resized*0:w_resized*1,:]=cv2.cvtColor(rgb2gray(img_blurred),cv2.COLOR_GRAY2RGB)

        img_test[h_resized*1:h_resized*2,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(img_blurred[:,:,0],thresh_lab)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(img_blurred[:,:,1],thresh_lab)*255,cv2.COLOR_GRAY2RGB)
        #img_test[h_resized*3:h_resized*4,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(img_blurred[:,:,2],thresh_lab)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*3:h_resized*4,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(rgb2gray(img_blurred),thresh_lab)*255,cv2.COLOR_GRAY2RGB)

        img_test[h_resized*1:h_resized*2,w_resized*2:w_resized*3,:]=cv2.cvtColor(line_rgw_bin(img_blurred)*255,cv2.COLOR_GRAY2RGB)
        #img_test[h_resized*2:h_resized*3,w_resized*2:w_resized*3,:]=cv2.cvtColor(threshold(rgb2luv(img_blurred)[:,:,1],thresh_luv)*255,cv2.COLOR_GRAY2RGB)
        #img_test[h_resized*3:h_resized*4,w_resized*2:w_resized*3,:]=cv2.cvtColor(threshold(rgb2luv(img_blurred)[:,:,2],thresh_luv)*255,cv2.COLOR_GRAY2RGB)
        '''


        
        '''  
        #HLS, Lab, LUV analysis
        img_test[h_resized*1:h_resized*2,w_resized*0:w_resized*1,:]=cv2.cvtColor(threshold(rgb2hls(img_blurred)[:,:,0],thresh_hls)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*0:w_resized*1,:]=cv2.cvtColor(threshold(rgb2hls(img_blurred)[:,:,1],thresh_hls)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*3:h_resized*4,w_resized*0:w_resized*1,:]=cv2.cvtColor(threshold(rgb2hls(img_blurred)[:,:,2],thresh_hls)*255,cv2.COLOR_GRAY2RGB)

        img_test[h_resized*1:h_resized*2,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(rgb2lab(img_blurred)[:,:,0],thresh_lab)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(rgb2lab(img_blurred)[:,:,1],thresh_lab)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*3:h_resized*4,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(rgb2lab(img_blurred)[:,:,2],thresh_lab)*255,cv2.COLOR_GRAY2RGB)

        img_test[h_resized*1:h_resized*2,w_resized*2:w_resized*3,:]=cv2.cvtColor(threshold(rgb2luv(img_blurred)[:,:,0],thresh_luv)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*2:w_resized*3,:]=cv2.cvtColor(threshold(rgb2luv(img_blurred)[:,:,1],thresh_luv)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*3:h_resized*4,w_resized*2:w_resized*3,:]=cv2.cvtColor(threshold(rgb2luv(img_blurred)[:,:,2],thresh_luv)*255,cv2.COLOR_GRAY2RGB)
        
        
        #Red&White corner line analysis
        img_test[h_resized*0:h_resized*1,w_resized*2:w_resized*3,:] = cv2.cvtColor(line_wr_bin(img_blurred)*255, cv2.COLOR_GRAY2RGB)
        img_test[h_resized*1:h_resized*2,w_resized*0:w_resized*1,:]=cv2.cvtColor(threshold(rgb2hls(img_blurred)[:,:,2],thresh_hls)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*1:h_resized*2,w_resized*1:w_resized*2,:]=cv2.cvtColor((1-threshold(rgb2lab(img_blurred)[:,:,2],thresh_lab))*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*1:h_resized*2,w_resized*2:w_resized*3,:]=cv2.cvtColor((threshold(rgb2luv(img_blurred)[:,:,1],thresh_luv))*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*0:w_resized*1,:]=cv2.cvtColor(threshold(rgb2gray(img_blurred),thresh)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*1:w_resized*2,:]=cv2.cvtColor(threshold(img_blurred[:,:,0],thresh)*255,cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*2:w_resized*3,:]=cv2.cvtColor((1-threshold(rgb2luv(img_blurred)[:,:,1],thresh_luv))*255,cv2.COLOR_GRAY2RGB)
        '''




        
        #Gradient analysis
        img_test[h_resized*1:h_resized*2,w_resized*0:w_resized*1,:] = cv2.cvtColor(abs_sobel_th(img_color, orient='x', ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)
        img_test[h_resized*1:h_resized*2,w_resized*1:w_resized*2,:] = cv2.cvtColor(abs_sobel_th(img_color, orient='y', ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)
        img_test[h_resized*1:h_resized*2,w_resized*2:w_resized*3,:] = cv2.cvtColor(mag_sobel_th(img_color, ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)

        img_test[h_resized*2:h_resized*3,w_resized*0:w_resized*1,:] = cv2.cvtColor(abs_sobel_th(img_hls, orient='x', ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*1:w_resized*2,:] = cv2.cvtColor(abs_sobel_th(img_hls, orient='y', ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)
        img_test[h_resized*2:h_resized*3,w_resized*2:w_resized*3,:] = cv2.cvtColor(mag_sobel_th(img_hls, ksize=kernel_abx, thresh=thresh_abx)*255, cv2.COLOR_GRAY2RGB)

        img_test[h_resized*3:h_resized*4,w_resized*0:w_resized*1,:] = cv2.cvtColor(img_bin*255, cv2.COLOR_GRAY2RGB)


        #Perspective Transform analysis
        #img_copy = cv2.cvtColor(np.copy(img_bin)*255, cv2.COLOR_GRAY2RGB)
        img_copy = cv2.cvtColor(img_grad_mag*255, cv2.COLOR_GRAY2RGB)
        (bottom_px, right_px) = (img_copy.shape[0] - 1, img_copy.shape[1] - 1) 
        #(bottom_px, right_px) = (h_resized*2 - 1, w_resized*2 - 1)
        #print(bottom_px, right_px)
        print(img_copy.shape)
        #pts = np.array([[0,bottom_px-120],[280,10],[w_resized-280,10], [w_resized, bottom_px-120]], np.int32)#LB, LT, RT, RB, [x,y]
        #pts = np.array([[0,bottom_px],[160,30],[w_resized-160,30], [w_resized, bottom_px]], np.int32)
        pts = np.array([[0,bottom_px],[w_src,h_src],[w_resized-w_src,h_src], [w_resized, bottom_px]], np.int32)#LB, LT, RT, RB, [x,y]
        cv2.polylines(img_copy,[pts],True,(255,0,0), 3)
        
        img_test[h_resized*3:h_resized*4,w_resized*1:w_resized*2,:] = img_copy

        
        src_pts = pts.astype(np.float32)
        #dst_pts = np.array([[50,bottom_px-10], [50,10], [590,10], [590, bottom_px-10]], np.float32)
        #dst_pts = np.array([[80,bottom_px-20], [80,20], [340,20], [340, bottom_px-20]], np.float32)
        dst_pts = np.array([[w_dst,bottom_px-h_dst], [w_dst,h_dst], [w_resized-w_dst,20], [w_resized-w_dst, bottom_px-h_dst]], np.float32)
        #img_pers, Mpers, Minvs = corners_unwarp(np.copy(img_bin)*255, src_pts, dst_pts)
        img_pers, Mpers, Minvs = corners_unwarp(np.copy(img_grad_mag)*255, src_pts, dst_pts)
        #print(img_pers)
        img_test[h_resized*3:h_resized*4,w_resized*2:w_resized*3,:] = cv2.cvtColor(img_pers, cv2.COLOR_GRAY2RGB)


        img_test[h_resized*4:h_resized*5,w_resized*0:w_resized*1,:] = draw_hist(img_pers)


        '''
        hist_pers = np.sum(img_pers[img_pers.shape[0]//2:,:], axis=0)
        fig, ax = plt.subplots(1,2, figsize=(15,4))
        
        ax[0].imshow(img_pers, cmap = 'gray')
        ax[0].axis("off")
        ax[1].plot(hist_pers)
        
        plt.imshow(ax)
        '''
        
        

        



        if (mode == 'adjust'):
            cv2.putText(img_test, (guide1), (50, h_resized*6+0), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide2), (50, h_resized*6+30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide3), (50, h_resized*6+60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide4) + "threshold =" + str(thresh) + ", direction =" + str(direction) + ", kernel =" + str(kernel), (50, h_resized*6+90), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img_test, (guide1), (50, h_resized*6+0), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide2 + "th_hls= " + str(thresh_hls )+ ", th_lab= " + str(thresh_lab) + ", th_luv= " + str(thresh_luv)), (50, h_resized*6+30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide3 + "th_abx= " + str(thresh_abx) + ", th_aby= " + str(thresh_aby) + ", th_mag= " + str(thresh_mag)), (50, h_resized*6+60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_test, (guide4 + "th_dir= " + str(direction_dir)), (50, h_resized*6+90), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("test", img_test)
        

        ##Note##
        #kernel = 11, thresh = 64 good for S in HLS --> Grass/road separation
        #kernel = 11, thresh = 94 good for b in Lab --> RedWhite separation
        #kernel = 11, thresh = 92 good for U in LUV --> RedWhite separation
        #kernel = 11, thresh = 190 good for W in GRAY --> White separation



        
        


    else:
        break

cap.release()
cv2.destroyAllWindows()