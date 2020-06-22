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
    Mpers = cv2.getPerspectiveTransform(src, dst)
    Minvs = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    img_warp = cv2.warpPerspective(img, Mpers, img_size)

    # Return the resulting image and matrix
    return img_warp, Mpers, Minvs


def compute_perspective_transform_matrices(src, dst):
    """
    Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
    and M_inv is the matrix used to revert the transformed image back to the original one
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return (M, M_inv)


def perspective_transform(img, src, dst):   
    """
    Applies a perspective 
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped
