#!/usr/bin/env python

import numpy as np
import pyscreenshot as ImageGrab
import cv2
import time
import matplotlib as plt

X1 = 90
Y1 = 60
X2 = 990
Y2 = 900


def equalizeIntensity(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])

    # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    return img


def apply_thresholds(image, show=True):
    img = np.copy(image)
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,2]   

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    #color_binary = np.dstack((u_binary, s_binary, l_binary))
    
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    if show == True:
        # Plotting thresholded images
        f, ((ax1, ax2, ax3), (ax4,ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(10,4))
        f.tight_layout()
        
        ax1.set_title('Original Image', fontsize=16)
        ax1.imshow(cv2.cvtColor(undistort(image, show=False),cv2.COLOR_BGR2RGB))
        
        ax2.set_title('Warped Image', fontsize=16)
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'))
        
        ax3.set_title('s binary threshold', fontsize=16)
        ax3.imshow(s_binary, cmap='gray')
        
        ax4.set_title('b binary threshold', fontsize=16)
        ax4.imshow(b_binary, cmap='gray')
        
        ax5.set_title('l binary threshold', fontsize=16)
        ax5.imshow(l_binary, cmap='gray')

        ax6.set_title('Combined color thresholds', fontsize=16)
        ax6.imshow(combined_binary, cmap='gray')
        
        
    else: 
        return combined_binary

def canny_edge_thresh(img, thresh_min=10, thresh_max = 40):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_size = 15
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    masked_edges = cv2.Canny(blur_gray, thresh_min, thresh_max)

    return cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)

def abs_sobel_thresh(img, orient='xy', thresh_min=20, thresh_max=100):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
    if orient == 'x':
        sobel = sobelx
    elif orient == 'y':
        sobel = sobely
    else:
        sobel = (sobelxy+sobelx+sobely)/3.0
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(thresh_min <= scaled_sobel) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return 255*cv2.cvtColor(binary_sobel, cv2.COLOR_GRAY2BGR), binary_sobel

def sliding_window_polyfit(img, prevpoint, forget_factor = 0.95):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//4:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered

    if (prevpoint[0]==0) and (prevpoint[1]==0):
        leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint
    else:
        leftx_base = prevpoint[0]*forget_factor + (1-forget_factor)*(np.argmax(histogram[quarter_point:midpoint]) + quarter_point)
        rightx_base = prevpoint[1]*forget_factor + (1-forget_factor)*(np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint)
    
    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int(img.shape[0]/(2*nwindows))
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 20
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, [leftx_base, rightx_base]

def hsv_detect_gray(image, s_thresh=30,v_thresh=150):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = img[:,:,1] < s_thresh
    v1 = img[:,:,2] > v_thresh
    av_thresh = np.percentile(img[:,:,2],90)
    s = img[:,:,1]< s_thresh
    v2 = img[:,:,2] > av_thresh
    v = (v1==1) | (v2==1)
    bin_img = np.bitwise_and(s,v, dtype = np.uint8)
    # print bin_img
    return 255*cv2.cvtColor(bin_img,cv2.COLOR_GRAY2BGR)

def draw_lines_extrapolate(img, lines, color=[255, 0, 0], thickness=2):
    # Assume lines on left and right have opposite signed slopes
    lines_left = []
    lines_right = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0: continue; # Infinite slope
            slope = float(y2-y1)/float(x2-x1)
            if abs(slope) < .5 or abs(slope) > .9: continue # Discard unlikely slopes
            if slope > 0: 
                lines_left += [(x1,y1),(x2,y2)]
            else: 
                lines_right += [(x1,y1),(x2,y2)]

    left_xs = map(lambda x: x[0], lines_left)
    left_ys = map(lambda x: x[1], lines_left)
    right_xs = map(lambda x: x[0], lines_right)
    right_ys = map(lambda x: x[1], lines_right)
    
    left_fit = np.polyfit(left_xs, left_ys, 1)
    right_fit = np.polyfit(right_xs, right_ys, 1)
    
    y1 = img.shape[0] # Bottom of image
    y2 = img.shape[0] / 2+ 50 # Middle of view
    x1_left = (y1 - left_fit[1]) / left_fit[0]
    x2_left = (y2 - left_fit[1]) / left_fit[0]
    x1_right = (y1 - right_fit[1]) / right_fit[0]
    x2_right = (y2 - right_fit[1]) / right_fit[0]    
    y1 = int(y1); y2 = int(y2);
    x1_left = int(x1_left); x2_left = int(x2_left);
    x1_right = int(x1_right); x2_right = int(x2_right);

    cv2.line(img, (x1_left, y1), (x2_left, y2), color, thickness)
    cv2.line(img, (x1_right, y1), (x2_right, y2), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

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
        origin_img = cv2.resize(frame,(0,0), fx=0.5, fy=0.5)
        warped = cv2.warpPerspective(frame, src2dstM, warp_size, flags=cv2.INTER_LINEAR, borderValue = (128,128,128))
        # warped = equalizeIntensity(warped)
        warped_mask = np.zeros((int(warp_y/2), int(warp_x), 3))
        warped_mask = warped[int(warp_y*4/10):int(warp_y*9/10),:,:]
        edge = canny_edge_thresh(warped_mask)
        test = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2HSV)
        # edge = cv2.merge([test[:,:,1],test[:,:,1],test[:,:,1]])
        edge = hsv_detect_gray(warped_mask)
        sobel, sobel_bin = abs_sobel_thresh(warped_mask,'x',70,170)
        # sobel = cv2.cvtColor(test[:,:,1], cv2.COLOR_GRAY2BGR)
        # sobel,_ = abs_sobel_thresh(sobel,'x',40,100)
        output = np.zeros((int(warp_y),int(screen_x/2+warp_x),3), dtype = "uint8")
        #--------------------------------------------------------------test
        # left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, prev_point = sliding_window_polyfit(sobel_bin, prev_point)
        # h = sobel_bin.shape[0]-1
        # ploty = np.linspace(0, sobel_bin.shape[0]-1, sobel_bin.shape[0]).astype(int)
        # left_fit_points = 0
        # right_fit_points = sobel_bin.shape[1]-1
        # try:
        #     left_fit_points = np.round(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]).astype(int)
        #     right_fit_points = np.round(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]).astype(int)
        #     ploty_left = ploty[(left_fit_points<sobel_bin.shape[1]-1) & (left_fit_points>0)]
        #     ploty_right = ploty[(right_fit_points<sobel_bin.shape[1]-1) & (right_fit_points>0)]
        #     left_fit_points = left_fit_points[(left_fit_points<sobel_bin.shape[1]-1) & (left_fit_points>0)]
        #     right_fit_points = right_fit_points[(right_fit_points<sobel_bin.shape[1]-1) & (right_fit_points>0)]
        # except:
        #     pass
        # nonzero = sobel_bin.nonzero()
        # nonzeroy = np.array(nonzero[0])
        # nonzerox = np.array(nonzero[1])
        # sobel[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
        # sobel[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
        # sobel[ploty_left,left_fit_points]=[0,255,255]
        # sobel[ploty_right,right_fit_points]=[0,255,255]
    
        #--------------------------------------------------------------test end
        # 
        #----------------------------------------------------------------test2
        output = np.zeros((int(screen_y), int(screen_x/2),3),dtype='uint8')

        test2 = hsv_detect_gray(origin_img)
        test2 = canny_edge_thresh(origin_img)
        
        lines = cv2.HoughLinesP(test2[:,:,1], 1, np.pi/180.0, 150,np.array([]), 200, 100)
        # lines = cv2.HoughLines(test2[:,:,1], 1, np.pi/180.0, 200)
        print(lines)
        # line_img = np.zeros((test2.shape[0], test2.shape[1],3),dtype='uint8')
        draw_lines(test2, lines)

        # for rho,theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)

        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 + test2.shape[1]*(-b))
        #     y1 = int(y0 + test2.shape[1]*(a))
        #     x2 = int(x0 - test2.shape[1]*(-b))
        #     y2 = int(y0 - test2.shape[1]*(a))

        #     cv2.line(test2,(x1,y1),(x2,y2),(0,0,255),2)


        
        
        output[0:int(screen_y/2), 0:int(screen_x/2)] = origin_img
        output[int(screen_y/2):int(screen_y), 0:int(screen_x/2)] = test2
        cv2.line(output, (0,int(screen_y/2)),(int(screen_x/2), int(screen_y/2)),(0,255,255),5)
        # # output[0:warp_y, 0:warp_x] = warped
        # output[0:warp_y/2, warp_x:2*warp_x] = sobel
        # output[warp_y/2:warp_y, warp_x:2*warp_x] = edge
        # cv2.line(output, (warp_x,0), (warp_x, warp_y), (0, 255, 255), 5)
        # cv2.line(output, (warp_x,warp_y/2), (2*warp_x, warp_y/2), (0, 255, 255), 5)
        cv2.imshow('TEST_WINDOW', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


