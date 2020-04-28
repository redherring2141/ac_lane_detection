import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def threshold(img, thresh=(0,255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    thresholded = np.zeros_like(img)
    thresholded[(img >= thresh_min) & (img <= thresh_max)] = 1
    img_bin = np.copy(thresholded)
    
    return img_bin
    
    
#fefaeaf:qqfqwefqwipjfqweipjfipqwejfi Define a function that thresholds the selected channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, ch='s', thresh=(115, 255)):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Select a channel
    if ch == 'h':
        img_ch = img_hls[:, :, 0] #0 for H, 1 for L, 2 for S
    elif ch == 'l':
        img_ch = img_hls[:, :, 1]
    else:
        img_ch = img_hls[:, :, 2]
    # 3) Return a binary image of threshold result
    img_bin = threshold(img_ch, thresh)
    return img_bin


# Define a function that thresholds the selected channel of Lab
# Use exclusive lower bound (>) and inclusive upper (<=)
def lab_select(img, ch='b', thresh=(0, 255)):
    # 1) Convert to Lab color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    # 2) Select a channel
    if ch == 'l':
        img_ch = img_lab[:, :, 0] #0 for L, 1 for a, 2 for b
    elif ch == 'a':
        img_ch = img_lab[:, :, 1]
    else:
        img_ch = img_lab[:, :, 2]
    # 3) Return a binary image of threshold result
    img_bin = threshold(img_ch, thresh)
    return img_bin


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def luv_select(img, ch='l', thresh=(0, 255)):
    # 1) Convert to HLS color space
    img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    # 2) Select a channel
    if ch == 'l':
        img_ch = img_luv[:, :, 0] #0 for L, 1 for u, 2 for v
    elif ch == 'u':
        img_ch = img_luv[:, :, 1]
    else:
        img_ch = img_luv[:, :, 2]
    # 3) Return a binary image of threshold result
    img_bin = threshold(img_ch, thresh)
    return img_bin


# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_th(img, orient='x', ksize=3, thresh=(30,120)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your img_bin image
    
    # Read in an image and grayscale it
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobel = np.zeros_like(gray)

    if orient == 'x' :
        # Sobel x
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    if orient == 'y' :
        # Sobel y
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y
        
    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#   scaled_sobel = np.uint8(abs_sobel/np.max(abs_sobel))
    
    # Threshold gradient
    img_bin = threshold(scaled_sobel, thresh)
    return img_bin


# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_sobel_th(img, ksize=31, thresh=(30,100)):
    # Apply the following units to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your img_bin image

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)

    scaled_sobel = (255*mag_sobel/np.max(mag_sobel)).astype(np.uint8)
#    scaled_sobel = (mag_sobel/np.max(mag_sobel)).astype(np.uint8)

    # Thresholding
    img_bin = threshold(scaled_sobel, thresh)
    return img_bin 


# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_sobel_th(img, ksize=31, thresh=(0.5,1.5)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your img_bin image

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) # Take the derivative in x
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1)) # Take the derivative in y
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    
    # Thresholding
    img_bin = threshold(dir_sobel, thresh)
    return img_bin


#def combine_imgbin(img, ks=3, orient='x', th_sb=(0,255), th_r=(0,255), th_g=(0,255), th_b=(0,255)):
    
