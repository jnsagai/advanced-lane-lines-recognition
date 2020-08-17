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

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

##############################################################################
#########################   GLOBAL PARAMETERS    #############################
##############################################################################

sx_thresh = (60 , 100)
mag_thresh = (60, 200)
dir_thresh = (0.5, 1.1)

# HLS Channels threshold
h_thresh = (5, 100)
l_thresh = (210, 255)
s_thresh = (100, 255)

# Bird-eye view offsets
w_offset = 350
h_offset = 0

# Define an offset for the ROI relative to the transformation source polygon
roi_offset = 30

# Get the set of chessboard images for the calibration process
images = glob.glob('camera_cal/calibration*.jpg')

# Instances of Circular Buffers to apply the low-pass filter
left_line_cur_buf = acv.RingBuffer(20)
right_line_cur_buf = acv.RingBuffer(20)
car_center_buf = acv.RingBuffer(20)

# Last lanes fit values
prev_left_fit = np.array([])
prev_right_fit = np.array([])

##############################################################################
##############################   MAIN CODE   #################################
##############################################################################

# Get the distortion coefficients
mtx, dist = acv.compute_dist_coeff(images)

# Hardcoded values for the matrix transform and distortion coefficients.
# mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
#         [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
#         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# dist = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])

def process_image(image):    
    global prev_left_fit
    global prev_right_fit
    
    # Undistort the current frame according to the calibration parameters
    undistorted = acv.cal_undistort(image, mtx, dist)   
 
    # Apply threshold for each component of the HLS color space image
    combined_hls = acv.hls_threshold(undistorted, h_thresh, l_thresh, s_thresh)
    
    # Apply Sobel threshold in the x direction
    sxbinary = acv.abs_sobel_thresh(undistorted, orient='x', sobel_kernel = 3, thresh = sx_thresh)
    
    # Apply the magnitude threshold
    mag_binary = acv.mag_thresh(undistorted, sobel_kernel = 3, thresh = mag_thresh)
    
    # Apply the direction threshold
    dir_binary = acv.dir_threshold(undistorted, sobel_kernel = 3, thresh = dir_thresh)
    
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
    
    # Create a mask edge for a region of interest based on the projection source
    vertices = np.array([[(src[0][0] - roi_offset, src[0][1]),
                          (src[1][0] - roi_offset, src[1][1] - roi_offset),
                          (src[2][0] + roi_offset, src[2][1] - roi_offset),
                          (src[3][0] + roi_offset, src[3][1])]], dtype=np.int32)
    
    # Based on the ROI, apply a mask on the combined binary image
    masked_edges = acv.region_of_interest(combined_binary, vertices)
    
    # Warp the image using the projection source and destination
    warped_img, M, Minv = acv.warp_image(masked_edges, src, dst)    
    
    # Try to get the new line values from the previous one
    # If an error occurs, then search for a new line using whole pipeline
    if prev_left_fit.size != 0 and prev_right_fit.size != 0:
        error, left_fit, left_fitx, right_fit, right_fitx, ploty, lane_pixels = acv.search_around_poly(warped_img, prev_left_fit, prev_right_fit)
        if (error == True):
            _, left_fit, left_fitx, right_fit, right_fitx, ploty, lane_pixels = acv.fit_polynomial(warped_img)
    else:
        _, left_fit, left_fitx, right_fit, right_fitx, ploty, lane_pixels = acv.fit_polynomial(warped_img)
    
    # Use the left and right lanes pixes to calculate the curvature of the road
    left_curverad, right_curverad = acv.measure_curvature_real(ploty, lane_pixels)
    
    # Apply a low-pass filter to the lane curvature by buffering the last n reading and taking the average
    left_line_cur_buf.append(left_curverad)
    right_line_cur_buf.append(right_curverad)    
    avg_left_line_cur = np.average(left_line_cur_buf.get())
    avg_right_line_cur = np.average(right_line_cur_buf.get())
    
    # Calculate the car center position relatively to the lanes
    car_center_dev = acv.calc_car_rel_position(warped_img.shape, ploty, left_fit, right_fit)
    
    # Apply low-pass filter
    car_center_buf.append(car_center_dev)
    avg_car_center = np.average(car_center_buf.get())
    
    # Draw the lane lines back to the original image.
    unwarped_image = acv.draw_lines_original_image(undistorted, warped_img, lane_pixels, left_fitx, right_fitx, ploty, Minv)
    
    # Write the metrics on the image
    final_image = acv.write_metrics(unwarped_image, avg_left_line_cur, avg_right_line_cur, avg_car_center)
    
    # Save the current lane line fit values
    prev_left_fit = left_fit
    prev_right_fit = right_fit
    
    return final_image
    
clip = VideoFileClip('project_video.mp4')
white_clip = clip.fl_image(process_image)
white_clip.write_videofile('output_videos/project_video.mp4', audio=False)