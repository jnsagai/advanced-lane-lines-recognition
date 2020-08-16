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

# src = np.float32([[200,720], [588,450], [693,450], [1120,720]])
# dst = np.float32([[316, 720], [316, 0], [959, 0],  [959, 720]])

# src = np.float32([[180,720], [590,450], [680,450], [1120,720]])
# dst = np.float32([[310, 720], [310, 0], [960, 0],  [960, 720]])

# Get the set of chessboard images for the calibration process
images = glob.glob('camera_cal/calibration*.jpg')

# Road test image
#test_image = mpimg.imread('test_images/straight_lines1.jpg')
#test_image = mpimg.imread('test_images/straight_lines2.jpg')
#test_image = mpimg.imread('test_images/test1.jpg')
#test_image = mpimg.imread('test_images/test2.jpg')
#test_image = mpimg.imread('test_images/test3.jpg')
#test_image = mpimg.imread('test_images/test4.jpg')
#test_image = mpimg.imread('test_images/test5.jpg')
test_image = mpimg.imread('test_images/test6.jpg')


##############################################################################
##############################   MAIN CODE   #################################
##############################################################################

# Get the distortion coefficients
# mtx, dist = acv.compute_dist_coeff(images)

# TODO REMOVE THIS CODE AFTER THE TESTS AND UNCOMMENT THE PREVIOUS LINE
mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
        [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])

curr_left_line = acv.Line()
prev_left_line = acv.Line()
curr_right_line = acv.Line()
prev_right_line = acv.Line()

line_buf = acv.RingBuffer(10)

def process_image(image):
    undistorted = acv.cal_undistort(image, mtx, dist)   
 
    combined_hls = acv.hls_threshold(undistorted, h_thresh, l_thresh, s_thresh)
    
    sxbinary = acv.abs_sobel_thresh(undistorted, orient='x', sobel_kernel = 3, thresh = sx_thresh)
    mag_binary = acv.mag_thresh(undistorted, sobel_kernel = 3, thresh = mag_thresh)
    dir_binary = acv.dir_threshold(undistorted, sobel_kernel = 3, thresh = dir_thresh)
    
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
    
    src = np.float32([[200,720],
                      [588,450],
                      [693,450],
                      [1120,720]])
    
    dst = np.float32([[w_offset, img_h - h_offset],
                      [w_offset, h_offset],
                      [img_w - w_offset, h_offset],
                      [img_w - w_offset, img_h - h_offset]])
    
    # Create a mask edge for a region of interest
    vertices = np.array([[(src[0][0] - roi_offset, src[0][1]),
                          (src[1][0] - roi_offset, src[1][1] - roi_offset),
                          (src[2][0] + roi_offset, src[2][1] - roi_offset),
                          (src[3][0] + roi_offset, src[3][1])]], dtype=np.int32)
    
    masked_edges = acv.region_of_interest(combined_binary, vertices)
    
    warped_img, M, Minv = acv.warp_image(masked_edges, src, dst)  
   
    _, left_fit, left_fitx, right_fit, right_fitx, ploty, roi_pixels = acv.fit_polynomial(warped_img)
    
    left_curverad, right_curverad = acv.measure_curvature_real(ploty, left_fit, right_fit)
    
    car_center_dev = acv.calc_car_rel_position(warped_img.shape, ploty, left_fit, right_fit)
    
    unwarped_image = acv.draw_lines_original_image(undistorted, warped_img, roi_pixels, left_fitx, right_fitx, ploty, Minv)
    
    final_image = acv.write_metrics(unwarped_image, left_curverad, right_curverad, car_center_dev)
    
    return final_image
    
clip = VideoFileClip('challenge_video.mp4')
#clip = VideoFileClip('project_video.mp4').subclip(0,5)
white_clip = clip.fl_image(process_image)
white_clip.write_videofile('output_videos/challenge_video.mp4', audio=False)