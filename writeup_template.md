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

[image1]: ./output_images/Undistorted_Chess_Image.png "Undistorted"
[image2]: ./output_images/Undistorted_Car_Image.png "Road Transformed"
[image3]: ./output_images/ThresholdedBinaryImage.png "Binary Example"
[image4]: ./output_images/WarpedImage.png "Warp Example"
[image5]: ./output_images/LaneLinePixels.png "Fit Visual"
[image6]: ./output_images/Final_Image.png "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The first step for the Camera Calibration was preparing the object points using x, y and z coordinates, where the chessboard corners will be stored. I used the given set of calibration image to doing so, where each image contains 6 x 9 corners.

After that, I've created two arrays to store the object points (3D points in real world space) and the image points (2D points in image plane). Afterwards the algorithm iterate over each image in the image set where, for each iteration, the image was converted to gray scale and the OpenCV function `cv2.findChessboardCorners` was used to find out the chessboard corners, then the amount was appended in the image points array.

When the iterations are finally completed, the camera matrix and the distortion coefficient is calculated using OpenCV function `cv2.calibrateCamera`.

I applied the camera matrix and the distortion coefficient to generate a test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Frame distortion correction

The first step in the pipeline is to correct the frame distortion by applying the Camera Matrix and the distortion coefficient calculated previously in the Camera Calibration step. I've used the OpenCV `cv2.undistort()` function to undistort the frame. An example of frame before and after being undistorted follows below:

![alt text][image2]

#### 2. Create a thresholded binary image

In order to identify the relevant pixels in the image which represent the left and right lane-lines I've used some threshold methods for filter the original frame and extract only the interested pixels.

 - First I converted the frame from RGB to HLS since this color space shown to be more relevant for image filtering. Then I applied some threshold on each HLS channel:
 
 ```python
h_thresh = (5, 100)
l_thresh = (210, 255)
s_thresh = (100, 255)
```

- After that I combined the individual thresholded binary image. The function can be found at lines 128 through 146 in `AdvancedComputerVision.py'.
- The next step was applying the Sobel ( x oriented ), Magnitude and Direction thresholds in the undistorted frame, then combined them in an individual binary image using the following combination ( the threshold functions can be found at lines 48 through 146 in `AdvancedComputerVision.py' ):

 ```python
combined_thres[(sxbinary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

- Then I joined this binary image and the hls thresholded image into a final thresholded combination.

 ```python
combined_binary[(combined_hls == 1) | (combined_thres == 1)] = 1
```

- Finally, using a set of vertices which surround the lane-lines, I applied a mask to thresholded image to get only the pixels in a region of interest.

Here's an example of an output for this step:

![alt text][image3]

#### 3. Perspective Transform

The perspective transform is applied by the function `warp_image()`, which is defined in lines 173 through 185 in the file `AdvancedComputerVision.py`. This function takes an input image (in this case the masked_edges images from the previous step), the source and the destination points. The source points were hardcoded based on an image with straight lane-lines, where the edges of lanes were used as vertices for the source. The destination points are based on the image width and height, considering a couple of offsets (`w_offset` and `h_offset`). The values follow below:
```python
src = np.float32([[200,720],
                  [588,450],
                  [693,450],
                  [1120,720]])
dst = np.float32([[w_offset, img_h - h_offset],
                  [w_offset, h_offset],
                  [img_w - w_offset, h_offset],
                  [img_w - w_offset, img_h - h_offset]])
```

Considering `w_offset = 350` and `h_offset = 0` the source and destination points are defined as:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 350, 720      | 
| 588, 450      | 350, 0        |
| 693, 450      | 930, 0        |
| 1120, 720     | 930, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane-line pixels identification and polynomial fit

In this step I checked whether a previous fit for the left and right lanes already exist, in order to optimize my search for the current line, which is defined in lines 126 through 131 in the file `AdvancedLaneLinesVideo.py`

If the pipeline was running upon the first frame, or the current lane-line couldn't be found within the margin of the previous line fit, then I used the function `fit_polynomial()` (defined in lines 355 through 392 in the file `AdvancedComputerVision.py`) to detect the lane-line pixels and calculate the new polynomial fit.
In order to find the lane pixels, the first act of `fit_polynomial()` function consists in calling `find_lane_pixels()` (lines 220 through 299 in the file `AdvancedComputerVision.py`), which steps are described as follow:
 - First, it takes a histogram of the bottom half of the image;
 - The peak of the left and right halves of the histogram is used to identify the lane-lines starting points;
 - Using the "Slicing Windows" method, a series of consecutive rectangular regions are defined in order to identify the relevant lane-line pixels for each side. The first window is center in the starting point defined in the previous step, then the next windows are center in the mean point of the found pixels inside the current window. Each set of pixels from each window are appended in a list.
 - After that, all the relevant lane-line pixels are returned to the `fit_polynomial()` function.

Using the left and right lane-line pixels, the function `np.polyfit()` is used to calculate the 2nd order polynomial coefficients for each curve. Finally, the polynomial curve, the sliding windows, and the left and right lane-line pixels are drawn in a composed image, as can be in the next image.
For the next next frames, first, it is checked whether the new lane can be found inside a region near to the previous one. The function `search_around_poly()` is used for this purpose (defined in lines 318 through 353 in the file `AdvancedComputerVision.py`). This function uses a margin around the previous polynomial and tries to reach the lane curve inside it. In this way, the pipeline performance can be improved, since part of the lane finding can be skipped and the pixel searching process is optimized.

![alt text][image5]

#### 5. The Radius of Curvature and Vehicle position

In this step, the radius of the curvature of the lane can be calculated based on the following equation [Radius of Curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).
Since we are dealing with the image in pixels space, it is needed to convert them to real-world metrics. In this case, meters.
As we are using US road examples, we can use US regulations that require a minimum lane width of 3.7 meters, and the dashed lane lines are 3 meters long each. We can capture the distance of the pixel between the lane and the dashed lane length from the warped frame in order to get a conversion factor. In this particular case I'm using the following factors:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
By using the lane-lines polynomial fit values and the conversion factors, and applying the Radius of Curvature formula, we can reach the approximate curvature for each lane. The  code is implemented in lines 393 through 417 in the file `AdvancedComputerVision.py`

The position of the car is calculated relative to the center point between the lanes. Considering the assumption that the camera is positioned in the center of the vehicle, the car position can be estimated by calculating the deviation of the frame's center point and the lanes' center point. Afterward, the value is converted to meters by using the `xm_per_pix` factor. The  code is implemented in lines 419 through 445 in the file `AdvancedComputerVision.py`

A low-pass filter was created based on both lane curvature and the car position values, in order to reduce the bouncing between the frame. I created a circular buffer class where I can store the last "n" measurements, and then I just take the average of the measures to be displayed in the final frame.

#### 6. Final image result

Finally, the warped image can be unwarped to the original perspective, by using the inversion M matrix. The function `draw_lines_original_image()` (defined in lines 447 through 480 in the file `AdvancedComputerVision.py`) is used for this purpose. Also, the pixels for each lane side are highlighted in different colors ( red for left lane and blue for right lane ), and a filled polygon is also drawn in the image ( green background ). Then the lane curvature and the car position are printed at the top of the frame.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Video result using my pipeline

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

I'm still facing some problems in trying to find the best set of parameters for the threshold methods. I spent a great amount of time in this task and I think that the final result could be even better if I was able to optimize the parameters and also the combination of the thresholds. One strategy that I'd like to test in the near future is to create a genetic algorithm for this purpose. I'll research also other methods related to lane detection.
My current algorithm is still using a mask to isolate only an ROI in the frame, whoever this approach is not really generic, and only gets the good result for lanes with fewer curvatures. 
I need to implement also some low-filters in the lane pixels themselves, in order to reduce the frame bouncing and increase the robustness.
