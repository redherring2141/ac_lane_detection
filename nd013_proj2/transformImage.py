import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, src, dst):
    # Use the OpenCV undistort() function to remove distortion
    # Convert undistorted image to grayscale
    # Search for corners in the grayscaled image

    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    #src = np.float32([[540, 480], [1280-540, 480], [120, 720], [1280-120, 720]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    #dst = np.float32([[240, 120], [1280-240, 120], [240, 720], [1280-240, 720]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    img_warp = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return img_warp, M
