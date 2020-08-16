import numpy as np
import cv2
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh = (0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)    
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
        
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(abs_sobel * 255 / np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sobel_bin = np.zeros_like(scaled_sobel)
    
    sobel_bin[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1    
          
    # 6) Return this mask as your binary_output image
    return sobel_bin

def mag_thresh(img, orient = 'xy', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    if orient == 'xy':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
    # 3) Calculate the magnitude 
    if orient == 'x':
        abs_sobel = np.sqrt(np.power(sobelx, 2))
    if orient == 'y':
        abs_sobel = np.sqrt(np.power(sobely, 2))
    if orient == 'xy':
        abs_sobel = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))        
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(abs_sobel * 255 / np.max(abs_sobel))
        
    # 5) Create a binary mask where mag thresholds are met
    sobel_bin = np.zeros_like(scaled_sobel)
    
    sobel_bin[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image

    return sobel_bin

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_threshold(img, thresh_h = (0,255), thresh_l = (0,255), thresh_s = (0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    bin_h = np.zeros_like(h_channel)
    bin_h[(h_channel > thresh_h[0]) & (h_channel < thresh_h[1])] = 1
    
    bin_s = np.zeros_like(s_channel)
    bin_s[(s_channel > thresh_s[0]) & (s_channel < thresh_s[1])] = 1
    
    bin_l = np.zeros_like(l_channel)
    bin_l[(l_channel > thresh_l[0]) & (l_channel < thresh_l[1])] = 1
    
    combined_hls = np.zeros_like(bin_h)
    combined_hls[((bin_h == 1) & (bin_s == 1)) | (bin_l == 1)] = 1
    
    return combined_hls

def warp_image(img, src, dst):
    
    img_size = (img.shape[1], img.shape[0])
   
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    # Calculate the Inverse Perspective Matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return warped, M, Minv

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def compute_dist_coeff(image_set):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.    
  
    test_image = cv2.imread(image_set[0])
    
    # Step through the list and search for chessboard corners
    for fname in image_set:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, test_image.shape[1::-1], None, None)

    return mtx, dist

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 12
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Create a dictionary to store all the nonzero pixels in the ROI
    roi_pixels = {
        'left_x' : leftx,
        'left_y' : lefty,
        'right_x' : rightx,
        'right_y' : righty
        }

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    
    draw_points = (np.asarray([left_fitx, ploty]).T).astype(np.int32) 
    cv2.polylines(out_img, [draw_points], False, (255,255,60), thickness = 5)
    draw_points = (np.asarray([right_fitx, ploty]).T).astype(np.int32) 
    cv2.polylines(out_img, [draw_points], False, (255,255,60), thickness = 5)
    
    return out_img, left_fit, left_fitx, right_fit, right_fitx, ploty, roi_pixels

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def calc_car_rel_position(img_shape, ploty, left_fit, right_fit):
    '''
    Calculates the relative position of the car with respect to the center
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Width of the image
    image_width = img_shape[1]
    
    # Calculate the middle point of the image ( reference point )
    mid_point = image_width / 2   
    
    # Get the starting point of the left and right lanes
    left_lane_sp = left_fit[0]*ploty[-1]**2 + left_fit[1]*ploty[-1] + left_fit[2]
    right_lane_sp = right_fit[0]*ploty[-1]**2 + right_fit[1]*ploty[-1] + right_fit[2]
    
    # Calculate the middle point between the left and right lane
    mid_lanes = left_lane_sp + (right_lane_sp - left_lane_sp) / 2
    
    # Calculate the relative deviation from the center
    # If the value is positive means that the car is to the right of the center
    # Otherwise it is to the left
    car_dev = ( mid_lanes - mid_point ) * xm_per_pix
    
    return car_dev

def draw_lines_original_image(original_img, warped, roi_pixels, left_fitx, right_fitx, ploty, Minv):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    color_lane_lines = np.zeros_like(color_warp)
    
    # Unpack the ROI pixels
    leftx = roi_pixels['left_x']
    lefty = roi_pixels['left_y']
    rightx = roi_pixels['right_x']
    righty = roi_pixels['right_y']
    
    color_lane_lines[lefty, leftx] = [255, 0, 0]
    color_lane_lines[righty, rightx] = [0, 0, 255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
    warped_lines = cv2.warpPerspective(color_lane_lines, Minv, (original_img.shape[1], original_img.shape[0])) 
    
    # Combine the result with the original image and the detected lane lines
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result, 0.8, warped_lines, 1, 0)
    
    return result

def write_metrics(image, left_curverad, right_curverad, car_center_dev):
    
    # Let's take the average of the curvature
    avg_curv = (left_curverad + right_curverad) / 2
    
    font_scale = 2
    
    thickness = 2
    
    img_w_metrics = image.copy()
    
    cv2.putText(img_w_metrics, 'Radius of Curvature: {0:d} m'.format(int(avg_curv)), (70, 70), 
                cv2.FONT_HERSHEY_SIMPLEX , font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if car_center_dev <= 0:        
        cv2.putText(img_w_metrics, 'Vehicle is: {0:2.2f} m left of center'.format(car_center_dev), (70, 130), 
                cv2.FONT_HERSHEY_SIMPLEX , font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    else:
        cv2.putText(img_w_metrics, 'Vehicle is: {0:2.2f} m right of center'.format(car_center_dev), (70, 130), 
                cv2.FONT_HERSHEY_SIMPLEX , font_scale, (255, 255, 255), thickness, cv2.LINE_AA)       
    
    return img_w_metrics