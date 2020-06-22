import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from classLaneLine import LaneLine
from classLaneLineHistory import LaneLineHistory


def mask_roi(img):
    unit = 128
    h=img.shape[0]
    w=img.shape[1]
    
    #left side
    l_bot_l = (unit*0, h*0.45)
    l_bot_r = (w, h*0.45)
    l_top_l = (unit*0, h*0.75)
    l_top_r = (w, h*0.75)
    
    #right side
    r_bot_l = (unit*0, h*0)
    r_bot_r = (w, h*0)
    r_top_l = (unit*0, h*0)
    r_top_r = (w, h*0)
    
    v = np.array([[l_bot_l, l_top_l, l_top_r, l_bot_r], [r_bot_l, r_top_l, r_top_r, r_bot_r]], dtype=np.int32)
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, v, (255,255,255))
    
    img_masked = cv2.bitwise_and(img, mask)
    
    binary_output = np.copy(img_masked)
    
    #return binary_output
    return img_masked


def mask_window(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    
    # Store the (left,right) window centroid positions per level
    window_centroids = []
    lc_pts = []
    rc_pts = []
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    #lc_pts.append(l_center)
    #rc_pts.append(r_center)
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))
        lc_pts.append(l_center)
        rc_pts.append(r_center)
        
    return window_centroids, lc_pts, rc_pts


def show_window_centroids(img_warp, window_width, window_height, window_centroids):
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img_warp)
        r_points = np.zeros_like(img_warp)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # mask_window is a function to draw window areas
            l_mask = mask_window(window_width,window_height,img_warp,window_centroids[level][0],level)
            r_mask = mask_window(window_width,window_height,img_warp,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
#        warpage= np.dstack((img_warp, img_warp, img_warp))*255 # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((img_warp, img_warp, img_warp)), np.uint8)
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((img_warp,img_warp,img_warp)),np.uint8)
        
    return output


def polyfit_window(height_img, left_center_points, right_center_points, window_height):
    
    ploty_img = np.linspace(0, height_img-1, height_img)
    windows = int((height_img - window_height) / window_height)
    
    ploty_win = []
    for k in range(1, windows+1):
        ploty_win.append(window_height * k)
        
    ploty_win.reverse()

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty_win, left_center_points, 2)
    left_fitx = left_fit[0]*ploty_img**2 + left_fit[1]*ploty_img + left_fit[2]
    
    right_fit = np.polyfit(ploty_win, right_center_points, 2)
    right_fitx = right_fit[0]*ploty_img**2 + right_fit[1]*ploty_img + right_fit[2]
    
    return left_fitx, right_fitx, ploty_win


'''
Calculates the curvature of polynomial functions in pixels.
'''
def measure_curve_r(ploty_img, left_center_points, right_center_points, ploty_win, h_img):

    
    l_xpts = np.array(left_center_points)
    r_xpts = np.array(right_center_points)
    ypts = np.array(ploty_win)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ypts)
    

    # Make sure to feed in your real data instead in your project!    
    left_fit = np.polyfit(ypts, left_center_points, 2)
    right_fit = np.polyfit(ypts, left_center_points, 2)

    left_curverad = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(1.5))//(np.absolute(2*left_fit[0]))
    right_curverad = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(1.5))//(np.absolute(2*right_fit[0]))
    
    avg_curve = round(((left_curverad + right_curverad) / 2), 0)
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/768 # meters per pixel in x dimension
    
    # Make sure to feed in your real data instead in your project!    
    left_fit_cr = np.polyfit(ypts*ym_per_pix, l_xpts*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ypts*ym_per_pix, r_xpts*xm_per_pix, 2)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad_cr = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(1.5))//(np.absolute(2*left_fit_cr[0]))
    right_curverad_cr = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(1.5))//(np.absolute(2*right_fit_cr[0]))
    
    avg_curve_cr = round(((left_curverad_cr + right_curverad_cr) / 2), 0)
    
    return avg_curve_cr


def get_offset(left_center_points, right_center_points, w_img):
    xm_per_pix = 3.7/700

    lx = left_center_points[0]
    rx = right_center_points[0]
    
    mid_img = w_img/2
    mid_car = (lx+rx)/2
    
    offset_p = mid_img - mid_car
    offset_m = round(offset_p * xm_per_pix, 2)
    
    return offset_m



def draw_hist(img):
    col_intgr_proj_hist = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    h_bins = np.sum(img[img.shape[0]//2:,:]//255, axis=0)
    #h_bins = np.sum(img//255, axis=0)
    #print(len(h_bins))

    for col in range(img.shape[1]):
        cv2.line(col_intgr_proj_hist, (col,0), (col,h_bins[col]), (0,255,0), 1)

    result = np.flipud(col_intgr_proj_hist)
    
    return result



def compute_lane_lines(img_pers):
    """
    Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LaneLine instances for
    the computed left and right lanes, for the supplied binary warped image
    """

    # Take a histogram of the bottom half of the image, summing pixel values column wise 
    histogram = np.sum(img_pers[img_pers.shape[0]//2:,:]//255, axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines 
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # don't forget to offset by midpoint!
    

    # Set height of windows
    window_height = np.int(img_pers.shape[0]//sliding_windows_per_line)
    # Identify the x and y positions of all nonzero pixels in the image
    # NOTE: nonzero returns a tuple of arrays in y and x directions
    nonzero = img_pers.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    total_non_zeros = len(nonzeroy)
    non_zero_found_pct = 0.0
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base    


    # Set the width of the windows +/- margin
    margin = sliding_window_half_width
    # Set minimum number of pixels found to recenter window
    minpix = sliding_window_recenter_thres
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Our lane line objects we store the result of this computation
    left_line = LaneLine()
    right_line = LaneLine()
                    
    if previous_left_lane_line is not None and previous_right_lane_line is not None:
        # We have already computed the lane lines polynomials from a previous image
        left_lane_inds = ((nonzerox > (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                        + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                        + self.previous_left_lane_line.polynomial_coeff[2] - margin)) 
                            & (nonzerox < (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                        + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                        + self.previous_left_lane_line.polynomial_coeff[2] + margin))) 

        right_lane_inds = ((nonzerox > (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                        + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                        + self.previous_right_lane_line.polynomial_coeff[2] - margin)) 
                            & (nonzerox < (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                        + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                        + self.previous_right_lane_line.polynomial_coeff[2] + margin))) 
        
        non_zero_found_left = np.sum(left_lane_inds)
        non_zero_found_right = np.sum(right_lane_inds)
        non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
        
        print("[Previous lane] Found pct={0}".format(non_zero_found_pct))
        #print(left_lane_inds)
    
    if non_zero_found_pct < 0.85:
        print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.sliding_windows_per_line):
            # Identify window boundaries in x and y (and right and left)
            # We are moving our windows from the bottom to the top of the screen (highest to lowest y value)
            win_y_low = img_pers.shape[0] - (window + 1)* window_height
            win_y_high = img_pers.shape[0] - window * window_height

            # Defining our window's coverage in the horizontal (i.e. x) direction 
            # Notice that the window's width is twice the margin
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            left_line.windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
            right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

            # Super crytic and hard to understand...
            # Basically nonzerox and nonzeroy have the same size and any nonzero pixel is identified by
            # (nonzeroy[i],nonzerox[i]), therefore we just return the i indices within the window that are nonzero
            # and can then index into nonzeroy and nonzerox to find the ACTUAL pixel coordinates that are not zero
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices since we now have a list of multiple arrays (e.g. ([1,3,6],[8,5,2]))
        # We want to create a single array with elements from all those lists (e.g. [1,3,6,8,5,2])
        # These are the indices that are non zero in our sliding windows
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        non_zero_found_left = np.sum(left_lane_inds)
        non_zero_found_right = np.sum(right_lane_inds)
        non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
        
        print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))
        

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    #print("[LEFT] Number of hot pixels={0}".format(len(leftx)))
    #print("[RIGHT] Number of hot pixels={0}".format(len(rightx)))
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #print("Poly left {0}".format(left_fit))
    #print("Poly right {0}".format(right_fit))
    left_line.polynomial_coeff = left_fit
    right_line.polynomial_coeff = right_fit
    
    if not self.previous_left_lane_lines.append(left_line):
        left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
        left_line.polynomial_coeff = left_fit
        self.previous_left_lane_lines.append(left_line, force=True)
        print("**** REVISED Poly left {0}".format(left_fit))            
    #else:
        #left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
        #left_line.polynomial_coeff = left_fit


    if not self.previous_right_lane_lines.append(right_line):
        right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
        right_line.polynomial_coeff = right_fit
        self.previous_right_lane_lines.append(right_line, force=True)
        print("**** REVISED Poly right {0}".format(right_fit))
    #else:
        #right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
        #right_line.polynomial_coeff = right_fit



    # Generate x and y values for plotting
    ploty = np.linspace(0, img_pers.shape[0] - 1, img_pers.shape[0] )
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    
    left_line.polynomial_coeff = left_fit
    left_line.line_fit_x = left_fitx
    left_line.non_zero_x = leftx  
    left_line.non_zero_y = lefty

    right_line.polynomial_coeff = right_fit
    right_line.line_fit_x = right_fitx
    right_line.non_zero_x = rightx
    right_line.non_zero_y = righty

    
    return (left_line, right_line)

