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
def hls_select(img, ch='s', thresh=(64, 255)):
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
def lab_select(img, ch='b', thresh=(94, 255)):
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
def luv_select(img, ch='u', thresh=(92, 255)):
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
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobel = np.zeros_like(img)

    if orient == 'x' :
        # Sobel x
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    if orient == 'y' :
        # Sobel y
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1) # Take the derivative in y
        
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
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1) # Take the derivative in y
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
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0)) # Take the derivative in x
    abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1)) # Take the derivative in y
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    
    # Thresholding
    img_bin = threshold(dir_sobel, thresh)
    return img_bin


def hls_wy_bin(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Compute a binary thresholded image where yellow is isolated from HLS components
    img_hls_yellow_bin = np.zeros_like(img_hls[:,:,0])
    img_hls_yellow_bin[((img_hls[:,:,0] >= 15) & (img_hls[:,:,0] <= 35))
                 & ((img_hls[:,:,1] >= 30) & (img_hls[:,:,1] <= 204))
                 & ((img_hls[:,:,2] >= 115) & (img_hls[:,:,2] <= 255))                
                ] = 1
    
    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(img_hls[:,:,0])
    img_hls_white_bin[((img_hls[:,:,0] >= 0) & (img_hls[:,:,0] <= 255))
                 & ((img_hls[:,:,1] >= 200) & (img_hls[:,:,1] <= 255))
                 & ((img_hls[:,:,2] >= 0) & (img_hls[:,:,2] <= 255))                
                ] = 1
    
    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(img_hls[:,:,0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

    return img_hls_white_yellow_bin


def line_wr_bin(img):
    #img_b = 1-threshold(img=rgb2lab(img)[:,:,2], thresh=(94,255))
    img_b = 1-lab_select(img)
    #img_u = threshold(img=rgb2luv(img)[:,:,1], thresh=(92,255))
    img_u = luv_select(img)
    img_bin = np.zeros_like(img_b)
    img_bin[(img_b == 1) | (img_u == 1)] = 1
    return img_bin


def line_rgw_bin(img):
    img_r = threshold(img=img[:,:,0], thresh=(125,255))
    img_g = threshold(img=img[:,:,1], thresh=(140,255))
    img_w = 1-threshold(img=rgb2gray(img), thresh=(0,124))

    img_bin = np.zeros_like(img_r)
    img_bin[(img_r == 1) | (img_g == 1) | (img_w == 1)] = 1
    return img_bin


def combined_color(img):
    img_s = hls_select(img)
    img_bin = np.zeros_like(img_s)
    #img_bin[(img_s == 1) | (line_wr_bin(img) == 1)] = 1
    img_bin[(img_s == 1) | (line_wr_bin(img) == 1) | (line_rgw_bin(img) == 1)] = 1
    return img_bin




#def combine_imgbin(img, ks=3, orient='x', th_sb=(0,255), th_r=(0,255), th_g=(0,255), th_b=(0,255)):

def rgb2gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

def rgb2lab(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    return img_lab

def rgb2luv(img):
    img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    return img_luv

def rgb2hls(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return img_hls

def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):
    #sxy_direction_binary = dir_sobel_th(gray_img, ksize=kernel_size, thresh=angle_thres)
    #combined = np.zeros_like(sxy_direction_binary)
    combined = np.zeros_like(sxy_magnitude_binary)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
    #combined[(sx_binary == 1) | ((sy_binary == 1) & (sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1
    combined[(sx_binary == 1) | ((sy_binary == 1) | (sxy_magnitude_binary == 1) )] = 1
    
    return combined


def compute_hls_white_yellow_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    hls_img = rgb2hls(rgb_img)
    
    # Compute a binary thresholded image where yellow is isolated from HLS components
    img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                 & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                 & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

    return img_hls_white_yellow_bin



def get_combined_binary_thresholded_img(undist_img):
    """
    Applies a combination of binary Sobel and color thresholding to an undistorted image
    Those binary images are then combined to produce the returned binary image
    """
    undist_img_gray = rgb2lab(undist_img)[:,:,0]
    sx = abs_sobel_th(undist_img_gray, orient='x', ksize=15, thresh=(20, 120))
    sy = abs_sobel_th(undist_img_gray, orient='y', ksize=15, thresh=(20, 120))
    sxy = mag_sobel_th(undist_img_gray, ksize=15, thresh=(80, 200))
    sxy_combined_dir = combined_sobels(sx, sy, sxy, undist_img_gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))   
    
    hls_w_y_thres = compute_hls_white_yellow_binary(undist_img)
    
    combined_binary = np.zeros_like(hls_w_y_thres)
    combined_binary[(sxy_combined_dir == 1) | (hls_w_y_thres == 1)] = 1
        
    return combined_binary



def colorGradThreshProcess(img_resized):
    img_blurred = cv2.GaussianBlur(img_resized,(11,11),25)
    img_color = combined_color(img_blurred)
    img_grad_mag = mag_sobel_th(img_color, ksize=7, thresh=(64,255))
    img_result = np.copy(img_grad_mag)*255
    return img_grad_mag