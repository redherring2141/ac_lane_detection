import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import pickle


from transformImage import undistort, corners_unwarp
from thresholdColorGrad import threshold, hls_select, lab_select, luv_select, abs_sobel_th, mag_sobel_th, dir_sobel_th
from findLane import mask_roi, mask_window, find_window_centroids, show_window_centroids, polyfit_window, measure_curve_r, get_offset
from drawLane import draw_lane
import classLine

    
nx = 9
ny = 6
ks = 17
th = (35,115)
w_win = 10
h_win = 90
margin = 25
font = cv2.FONT_HERSHEY_SIMPLEX

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "./wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img_sample = cv2.imread('../test_images/straight_lines1.jpg')

h_img = img_sample.shape[0]
w_img = img_sample.shape[1]
src = np.float32([[540, 480], [w_img-540, 480], [120, h_img], [w_img-120, h_img]])
dst = np.float32([[240, 120], [w_img-240, 120], [240, h_img], [w_img-240, h_img]])
Minv = cv2.getPerspectiveTransform(dst, src)
ploty_img = np.linspace(0, h_img-1, h_img)

dist_diff_squa_stack = []

def process_frame(self, img_org, flag=False):    
    
    img_undist = undistort(img_org, mtx, dist)
    img_undist_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

    img_wlane_th = lab_select(img_undist, thresh=(150, 200));
    img_ylane_th = luv_select(img_undist, thresh=(225, 255));
#    img_grad_abx = abs_sobel_th(img_undist, orient='x', ksize=ks, thresh=th)
#    img_grad_mag = mag_sobel_th(img_undist, ks, th)
#    img_grad_dir = dir_sobel_th(img_undist, ksize=7, thresh=(0.9, 1.3))

    img_combined = np.zeros_like(img_undist_gray)
    img_combined[ (img_wlane_th == 1) | (img_ylane_th == 1) ] = 1

#    img_combined[ (img_wlane_th == 1) | (img_ylane_th == 1) |
#                  (img_grad_abx == 1) | (img_grad_mag == 1) | (img_grad_dir == 1) ] = 1 
    img_pers, persM = corners_unwarp(img_combined, nx, ny, mtx, dist, src, dst)
    
    img_combined_pers = np.copy(img_pers)
    roi_combined = mask_roi(img_combined_pers)

    win_c, lc_pts, rc_pts = find_window_centroids(roi_combined, w_win, h_win, margin)
    left_fitx, right_fitx, ploty_win = polyfit_window(h_img, lc_pts, rc_pts, h_win)
    avg_curve_r = measure_curve_r(ploty_img, lc_pts, rc_pts, ploty_win, h_img)
    offset_m = get_offset(lc_pts, rc_pts, w_img)
    
    dist_bw_lines = np.average(left_fitx) - np.average(right_fitx)
    self.dist_bw_lines.append(dist_bw_lines)
    avg_dist_bw_lines = np.average(self.dist_bw_lines[-60:])
    dist_diff_squa = (avg_dist_bw_lines-dist_bw_lines)**2
    
    dist_diff_squa_stack.append(dist_diff_squa)
    dist_mse = np.average(dist_diff_squa_stack) # Mean Square Error
    print("dist_bw_lines", dist_bw_lines)
    print("dist_mse", dist_mse)
    
    
    if dist_mse <= 750:
        self.l_curr_fitx.append([left_fitx])
        self.r_curr_fitx.append([right_fitx])
        
    l_avg_fitx = np.average(self.l_curr_fitx[-5:], axis=0)
    r_avg_fitx = np.average(self.r_curr_fitx[-5:], axis=0)
    
  
    img_fin = draw_lane(roi_combined, l_avg_fitx, r_avg_fitx, ploty_img, img_undist, Minv)

    cv2.putText(img_fin, ("Curve Radius: " + str(avg_curve_r) + "m"), (256, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_fin, ("Center Offset: " + str(offset_m) + "m"), (896, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite("../output_images/output_img_undist.jpg", img_undist)
    cv2.imwrite("../output_images/output_img_wlane_th.jpg", img_wlane_th*255)
    cv2.imwrite("../output_images/output_img_ylane_th.jpg", img_ylane_th*255)
    """
    cv2.imwrite("../output_images/output_img_grad_abx.jpg", img_grad_abx*255)
    cv2.imwrite("../output_images/output_img_grad_mag.jpg", img_grad_mag*255)
    cv2.imwrite("../output_images/output_img_grad_dir.jpg", img_grad_dir*255)
    """
    cv2.imwrite("../output_images/output_img_combined.jpg", img_combined*255)
    cv2.imwrite("../output_images/output_img_combined_pers.jpg", img_combined_pers*255)
    cv2.imwrite("../output_images/output_img_fin.jpg", img_fin)
    
    
    if flag is False:
        return img_fin
    else:
        show_window_centroids(img_combined, w_win, h_win, win_c)
        return img_fin, roi_grad_abx, roi_grad_mag, roi_grad_dir, roi_hls_ch_s, img_combined, show_window_centroids, roi_wline_th, roi_yline_th
    
    
frame = classLine.Line()
img_sample = cv2.imread('../test_images/straight_lines2.jpg')
process_frame(frame, img_sample)
    