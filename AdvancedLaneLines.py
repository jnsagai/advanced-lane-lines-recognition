# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:03:57 2020

@author: jnnascimento
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import AdvancedComputerVision as acv

sx_thresh = (60 , 100)
mag_thresh = (60, 200)
dir_thresh = (0.5, 1.1)

# HLS Channels threshold
h_thresh = (5, 100)
l_thresh = (215, 255)
s_thresh = (100, 255)

# Bird-eye view offsets
w_offset = 350
h_offset = 0

# Define an offset for the ROI relative to the transformation source polygon
roi_offset = 30


# Get the set of chessboard images for the calibration process
images = glob.glob('camera_cal/calibration*.jpg')

# Road test image
test_image = mpimg.imread('test_images/straight_lines1.jpg')
#test_image = mpimg.imread('test_images/straight_lines2.jpg')
#test_image = mpimg.imread('test_images/test1.jpg')
#test_image = mpimg.imread('test_images/test2.jpg')
#test_image = mpimg.imread('test_images/test3.jpg')
#test_image = mpimg.imread('test_images/test4.jpg')
#test_image = mpimg.imread('test_images/test5.jpg')
#test_image = mpimg.imread('test_images/test6.jpg')


# Get the distortion coefficients
mtx, dist = acv.compute_dist_coeff(images)

# Hardcoded values for the matrix transform and distortion coefficients.
# mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
#         [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# dist = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])

# Undistort the current frame according to the calibration parameters
undistorted = acv.cal_undistort(test_image, mtx, dist)

# Safety copy of the image
img = np.copy(undistorted)

# Apply threshold for each component of the HLS color space image
combined_hls = acv.hls_threshold(img, h_thresh, l_thresh, s_thresh)

# Apply Sobel threshold in the x direction
sxbinary = acv.abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = sx_thresh)

# Apply the magnitude threshold
mag_binary = acv.mag_thresh(img, sobel_kernel = 3, thresh = mag_thresh)

# Apply the direction threshold
dir_binary = acv.dir_threshold(img, sobel_kernel = 3, thresh = dir_thresh)

# Combine Sobel, Magnitude and Direction threshold
combined_thres = np.zeros_like(dir_binary, dtype=np.uint8)
combined_thres[(sxbinary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Stack each channel
color_binary = np.dstack(( np.zeros_like(combined_thres), combined_thres, combined_hls)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(combined_hls == 1) | (combined_thres == 1)] = 1

# Get the warped image
# For source points I'm grabbing the outer four detected corners
img_h, img_w = combined_binary.shape[:2]

# Define the source points for the projection
src = np.float32([[200,720],
                  [588,450],
                  [693,450],
                  [1120,720]])

# Define the destination points for the projection
dst = np.float32([[w_offset, img_h - h_offset],
                  [w_offset, h_offset],
                  [img_w - w_offset, h_offset],
                  [img_w - w_offset, img_h - h_offset]])

# Create a mask edge for a region of interest
vertices = np.array([[(src[0][0] - roi_offset, src[0][1]),
                      (src[1][0] - roi_offset, src[1][1] - roi_offset),
                      (src[2][0] + roi_offset, src[2][1] - roi_offset),
                      (src[3][0] + roi_offset, src[3][1])]], dtype=np.int32)

# Create a mask edge for a region of interest based on the projection source
masked_edges = acv.region_of_interest(combined_binary, vertices)

# Warp the image using the projection source and destination
warped_img, M, Minv = acv.warp_image(masked_edges, src, dst)

#warped_color, _, _ = acv.warp_image(undistorted, src, dst)

poly_img, left_fit, left_fitx, right_fit, right_fitx, ploty, roi_pixels = acv.fit_polynomial(warped_img)

# Use the left and right lanes pixes to calculate the curvature of the road
left_curverad, right_curverad = acv.measure_curvature_real(ploty, roi_pixels)

# Calculate the car center position relatively to the lanes
car_center_dev = acv.calc_car_rel_position(warped_img.shape, ploty, left_fit, right_fit)

# Draw the lane lines back to the original image.
unwarped_image = acv.draw_lines_original_image(undistorted, warped_img, roi_pixels, left_fitx, right_fitx, ploty, Minv)

# Write the metrics on the image
final_image = acv.write_metrics(unwarped_image, left_curverad, right_curverad, car_center_dev)

print(left_curverad, 'm', right_curverad, 'm')
print('Car center: ', car_center_dev, 'm')

# Plotting thresholded images
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))
ax1.set_title('Undistorted image')
ax1.imshow(undistorted)

ax2.set_title('Masked Edges')
ax2.imshow(masked_edges, cmap='gray')

ax3.set_title('Warped Image')
ax3.imshow(warped_img, cmap='gray')

ax4.set_title('Final Image')
ax4.imshow(final_image)