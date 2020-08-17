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

Here's an example of a output for this step:

![alt text][image3]

#### 3. Perspective Transform

The perpective transform is applied by the function `warp_image()`, which is defined in lines 173 through 185 in the file `AdvancedComputerVision.py`. This function takes a input image (in this case the masked_edges images from the previous step), the source and the destination points. The source points were hardcoded based on an image with straight lane-lines, where the edges of lanes were used as vertices for the source. The destination points are based on the image width and height, considering a couple of offsets (`w_offset` and `h_offset`). The values follow below:
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
In order to find the lane pixels, the first action of `fit_polynomial()` function consist in calling `find_lane_pixels()` (lines 220 through 299 in the file `AdvancedComputerVision.py`), which steps are described as follow:
 - First it take a histogram of the bottom half of the image;
 - The peak of the left and right halves of the histogram are used to identify the lane-lines starting points;
 





Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
