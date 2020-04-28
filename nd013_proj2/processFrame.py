import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import pickle


from calibrateCam import findPts, calibrate, calibrateCam
from transformImage import undistort, corners_unwarp
from thresholdColorGrad import threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset
from drawLane import draw_lane
import classLine




    
nx = 9
ny = 6
ks = 17
th = (20,120)
w_win = 10
h_win = 92
margin = 25
font = cv2.FONT_HERSHEY_DUPLEX

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
#path_imgs = "../camera_cal/calibration*.jpg"
#objpoints, imgpoints = findPts(path_imgs)
#ret, mtx, dist, rvecs, tvecs = calibrate(objpoints, imgpoints)

#img_sample = cv2.imread('../test_images/straight_lines1.jpg')

#h_img = img_sample.shape[0]
#w_img = img_sample.shape[1]
src_x1 = 544
src_y1 = 470
src_x2 = 128
src_y2 = 700
dst_x  = 256
dst_y  = 128
'''
src = np.float32([[src_x1, src_y1], [w_img-src_x1, src_y1],
                  [src_x2, src_y2], [w_img-src_x2, src_y2]])
dst = np.float32([[dst_x, dst_y], [w_img-dst_x, dst_y],
                  [dst_x, h_img], [w_img-dst_x, h_img]])
'''
#Minvs = cv2.getPerspectiveTransform(dst, src)
#ploty_img = np.linspace(0, h_img-1, h_img)

dist_diff_squa_stack = []


#org
def process_frame(self, img_org, flag=False):
    
 #   img_undist = undistort(img_org, mtx, dist)
 #   img_pers, Mpers = corners_unwarp(img_undist, src, dst)
    img_pers = np.copy(img_org)
    
    
    h_img = img_org.shape[0]
    w_img = img_org.shape[1]
    ploty_img = np.linspace(0, h_img-1, h_img)
    src = np.float32([[src_x1, src_y1], [w_img-src_x1, src_y1], [src_x2, src_y2], [w_img-src_x2, src_y2]])
    dst = np.float32([[dst_x, dst_y], [w_img-dst_x, dst_y], [dst_x, h_img], [w_img-dst_x, h_img]])
    Minvs = cv2.getPerspectiveTransform(dst, src)
        


    img_hls_ch_s = hls_select(img_pers, ch='s', thresh=(125,190))

    img_grad_abx = abs_sobel_th(img_pers, orient='x', ksize=ks, thresh=(35,115))
    img_grad_mag = mag_sobel_th(img_pers, ksize=ks, thresh=(35,115))
    img_grad_dir = dir_sobel_th(img_pers, ksize=7, thresh=(0.8,1.4))

    img_wlane_th = threshold(cv2.cvtColor(img_pers, cv2.COLOR_RGB2GRAY), thresh=(225,255))
    img_roi_wlane = mask_roi(img_wlane_th)

    img_ylane_th = threshold(img_pers[:,:,0], thresh=(235,255))
    img_roi_ylane = mask_roi(img_ylane_th)

    img_roi_abx = mask_roi(img_grad_abx)
    img_roi_mag = mask_roi(img_grad_mag)
    img_roi_hls = mask_roi(img_hls_ch_s)

    img_combined = np.zeros_like(img_grad_dir)
    img_combined[(img_roi_abx==1) | (img_roi_hls==1) | (img_roi_ylane==1) | ((img_roi_mag==1) & (img_grad_dir==0)) | (img_roi_wlane==1)]=1

    win_c, lc_pts, rc_pts = find_window_centroids(img_combined, w_win, h_win, margin)
    
    left_fitx, right_fitx, ploty_win = polyfit_window(h_img, lc_pts, rc_pts, h_win)
    avg_curve_r = measure_curve_r(ploty_img, lc_pts, rc_pts, ploty_win, h_img)
    offset_m = get_offset(lc_pts, rc_pts, w_img)
    
    dist_bw_lines = abs(np.average(left_fitx) - np.average(right_fitx))
    self.dist_bw_lines.append(dist_bw_lines)
    avg_dist_bw_lines = np.average(self.dist_bw_lines[-60:])
    dist_diff_squa = (abs(avg_dist_bw_lines - dist_bw_lines))**2
    
    dist_diff_squa_stack.append(dist_diff_squa)
    dist_mse = np.average(dist_diff_squa_stack) # Mean Square Error
    
    if dist_diff_squa <= 750:
        self.l_curr_fitx.append([left_fitx])
        self.r_curr_fitx.append([right_fitx])
        
    l_avg_fitx = np.average(self.l_curr_fitx[-5:], axis=0)
    r_avg_fitx = np.average(self.r_curr_fitx[-5:], axis=0)
    
  
    img_fin = draw_lane(img_combined, l_avg_fitx, r_avg_fitx, ploty_img, img_org, Minvs)

    #cv2.imshow('TEST_WINDOW', img_fin)

    
    if flag is False:
        return img_fin
    else:
        show_window_centroids(img_combined, w_win, h_win, win_c)
        return img_fin, img_grad_abx, img_grad_mag, img_grad_dir, img_combined, show_window_centroids, img_wline_th, img_yline_th


'''
cap = cv2.VideoCapture("../../ac_laguna_mx5_2.mp4")

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        cv2.imshow('Output', process_frame(img_org = img))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''