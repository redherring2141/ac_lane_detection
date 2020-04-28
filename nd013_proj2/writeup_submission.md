## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration/corners_found10.jpg "Corners found chessboard image"
[image2]: ./output_images/output_straight_line1.jpg "Undistorted test image"
[image3]: ./output_images/camera_calibration/calibrated10.jpg "Calibrated chessboard image"
[image4]: ./output_images/output_img_undist.jpg "Undistorted test image"
[image5]: ./output_images/output_img_hls_ch_s.jpg "S channel in HLS color space thresholded image"
[image6]: ./output_images/output_img_lanes_th.jpg "Grayscale lane color thresholded image"
[image7]: ./output_images/output_img_grad_abx.jpg "Gradient in x thresholded image"
[image8]: ./output_images/output_img_grad_mag.jpg "Gradient magnitude thresholded image"
[image9]: ./output_images/output_img_grad_dir.jpg "Gradient direction thresholded image"
[image10]: ./output_images/output_img_combined.jpg "Combined thresholded image"
[image11]: ./output_images/output_img_combined_pers.jpg "Perspective-transformed image"
[image12]: ./output_images/output_img_fin.jpg "Final sample image with lane detection"

[video1]: ./output_videos/output_project_video.mp4 "Final output video with lane detection"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration section is located in “./files_for_submission/calibrate_camera.ipynb”.
At the beginning of the code, the required libraries are imported firstly, and the parameters for camera calibration, the number of horizontal and vertical corners in chessboard images `nx` and `ny` are set to 9 and 6, respectively. Based on the parameters set, the coordinates of object corners are set in the matrix `objp`, and also the two arrays to store the corners, `objpoints` and `imgpoints` are declared. The variable `images` stores the directory path containing uncalibrated chessboard images.

Once the variable setting procedure is done above, it enters the iteration process to collect corners from the object into `objpoints` and from the chessboard images into `imgpoints`. Firstly, the uncalibrated chessboard image is read and stored to the variable `img_cal_org` by `cv2.imread(fname)`, and then they are converted to grayscale image and stored to the variable `img_cal_gray`. After that, the function `cv2.findChessboardCorners(img_cal_gray, (nx,ny), None)` returns detected corners, and if the return value is valid, the found corner coordinates are appended to the two arrays of `objpoints` and `imgpoints`,respectively. The detected corners are denoted on the original image by the function `cv2.drawChessboardCorners(img_cal_org, (nx,ny), corners, ret)`. The output images are recorded by the function `cv2.imwrite(write_name1, img_cal_corner)` where `write_name1` designates the directory path to save the images. One of the output images are shown below at [image1].
![alt text][image1]

After finishing the iteration over the all chessboard images, with the two arrays `objpoints` and `imgpoints` containing appended corners, the function `cv2.calibrateCamera(objpoints, imgpoints, size_img, None, None)`  performs camera calibration and returns required parameters of `ret`, `mtx`, `dist`, `rvecs`, `tvecs`. Using `pickle` library, the outputs are exported to “./files_for_submission/wide_dist_pickle.p”.

Then, the calibrated result is applied to a test image at “./test_images/straight_lines1.jpg” by the function `cv2.imread` to check whether the calibration was performed correctly. After converting to grayscale image by the function `cv2.cvtColor(img_test_org, cv2.COLOR_BGR2GRAY)`, the converted grayscale version of test image `img_test_gray` is undistorted by the function `cv2.undistort(img_test_org, mtx, dist, None, mtx)`. The output result is shown at [image2].
![alt text][image2]

Finally, in the same way, the uncalibrated chessboard images with highlighted corners are undistorted through iteration using `cv2.undistort(img_cal_corner, mtx, dist, None, mtx)` and saved at `write_name2` containing the path “’./output_images/camera_calibration/calibrated’+str(idx)+’.jpg’”. One of them is shown at [image3].
![alt text][image3]




### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
The distortion correction procedure was already done in the camera calibration step described above, using the function `cv2.undistort(img. Mtx, dist, None, mtx)` which is wrapped by the function `undistort(img, mtx, dist)` contained in “./files_for_submission/transformImage.py”, so the output distortion-correction image is [image4] and located at “./output_images/output_img_undist.jpg”.
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The code section for this procedure is shown below, which is at “./files_for_submission/processFrame.py”, where all the called functions are defined at “./files_for_submission/thresholdColorGrad.py”.

    img_undist = undistort(img_org, mtx, dist)
    img_undist_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

    img_hls_ch_s = hls_select(img_undist)
    img_lanes_th = threshold(img_undist_gray, thresh=(225,255))
    img_grad_abx = abs_sobel_th(img_undist, orient='x', ksize=ks, thresh=th)
    img_grad_mag = mag_sobel_th(img_undist, ks, th)
    img_grad_dir = dir_sobel_th(img_undist, ksize=7, thresh=(0.9, 1.3))
    img_combined = np.zeros_like(img_grad_dir)
    img_combined[ (img_hls_ch_s == 1) | (img_lanes_th == 1) |
                  (img_grad_abx == 1) | (img_grad_mag == 1) | (img_grad_dir == 1) ] = 1

1.  Color Threshold
The types of color thresholding used here are S channel selection and thresholding after HLS color space conversion, gray-scale image thresholding customized for white lane and yellow lane. Each of them are shown in [image5], and [image6], respectively, which are thresholded with empirical threshold values after color space conversion using functions including `cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)` for grayscale conversion and `cv2.cvtColor(img, cv2.COLOR_BGR2HLS)` for HLS color space conversion inside `hls_select(img, thresh)` in “./files_for_submission/thresholdColorGrad.py”.
![alt text][image5]
![alt text][image6]

2. Gradient Threshold
The types of gradient thresholding used at this stage includes gradient value in x direction, gradient magnitude, and gradient direction. Each of them are depicted in [image7], [image8], and [image9], respectively.
![alt text][image7]
![alt text][image8]
![alt text][image9]

3. Combined Threshold
The color thresholded and gradient thresholded results are all combined through OR operator, and stored in `img_combined`, shown at [image10] below.
![alt text][image10]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
After performing color-based and gradient-based thresholding, the resulted combined-threshold image is perspective-transformed by the function `corners_unwarp(img_combined, nx, ny, mtx, dist, src, dst)` which is a combination of `cv2.getPerspectiveTransform(src, dst)` and `cv2.warpPerspective(nudist, M, img_size)` and defined at “./files_for_submission/transformImage.py”. The function returns a transformed image and perspective transform matrix, and the output image is shown at [image11].
![alt text][image11]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The functions used in this procedure are all defined in “./files_for_submission/findLane.py”.
    roi_combined = mask_roi(img_combined_pers)
    winc_pts, lc_pts, rc_pts = find_window_centroids(roi_combined, w_win, h_win, margin)
    left_fitx, right_fitx, ploty_win = polyfit_window(h_img, lc_pts, rc_pts, h_win)
The lane-line detection process firstly starts by masking the ROI(region-of-interest), which is performed by the function `mask_roi(img)`. The function simply determines the region of interest consisting of four vertices for each side of left and right.
Then, the ROI-masked is processed by the function `find_window_centroids(image, window_width, window_height, margin)`, which returns the coordinates of window centers at the left and right sides. 
The found window center positions are used in the function `polyfit_window(height_img, left_center_points, right_center_points, window_height)` which returns the x and y coordinate values of the window fit to polynomial. 


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The functions used in this procedure are all defined in “./files_for_submission/findLane.py”.
    avg_curve_r = measure_curve_r(ploty_img, lc_pts, rc_pts, ploty_win, h_img)
    offset_m = get_offset(lc_pts, rc_pts, w_img)
The function `measure_curve_r(ploty_img, left_center_points, right_center_points, ploty_win, h_img)` return average curvature radius calculated from the equation learned learned during the class to calculate curvature radius. It also includes the conversion from pixel to real world metric scale.
The function `get_offset(left_center_points, right_center_points, w_img)` returns simply how much difference there is between the center of the car and the center of the image in metric scale.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
After measuring the radius of curvature and offset, the mean square error of lines at both sides is calculated, and the detected lane-line region is denoted by the function `draw_lane(warped, left_fix, right_fitx, ploty, image, Minv)` defined in “./files_for_submission/draw_lane.py”. The result output image is shown at [image12].
![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
The final video output is at [video1].
![alt text][video1]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. The order of pipelined procedures
Whether the order of procedures can affect the performance, accuracy, or the result of lane-detection should be researched more. In other words, making perspective transform or ROI masking to precede thresholding section can result in different output. This was tried in the middle of development, and it seemd that it did not affect the output result crucially, so in terms of fine tuning, some changes in orders can possibly improve the final output. 
2. Fine tuning of parameters
There are many user-defined parameters which are determined empirically, for example, the vertices determining ROI mask, or many thresholding values. By tuning with delicately controlled parameters, the results could be improved.
3. Using ‘Search from Prior’ method
In this submission version, the approach of finding lane from previous result is not used, under the assumption that if it does not affect the throughput very much, doing search every frame is preferred for better accuracy. However, this is not proven assumption though it sounds reasonable.

