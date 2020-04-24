import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
