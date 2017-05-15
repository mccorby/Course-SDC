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

[image1]: ./output_images/chessboard_corners.png "Chessboard corners"
[image2]: ./test_images/test2.jpg "Original"
[image3]: ./output_images/undistorted.png "Undistorted"
[image4]: ./output_images/original_binary.png "Binary Example"
[image5]: ./output_images/warped_image.png "Warp Example"
[image6]: ./output_images/window_search_histogram.png "Window search histogram"
[image7]: ./output_images/window_search.png "Fit Visual"
[image8]: ./output_images/final_result.png "Output"
[video1]: ./video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is defined in the function `calibrate_camera` in `calibration.py` file.
The main function checks if the calibration has been previously done and saved in a file to avoid computing it for every execution.
If there is no file, then the calibration process in invoked

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0,
such that the object points are the same for each calibration image. 
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time
`findChessboardCorners` successfully detects all chessboard corners in a test image. 
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function. The matrix and coefficients are saved
 
The following image show how the corners are detected in a chessboard

![Chessboard corners][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
|![Original][image2]|![Undistorted][image3]| 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS and Lab color spaces to generate a binary image. Channel L in HSL and b in Lab provided the best
results to isolate the white and yellow lines.

Initially I was using also gradient thresholding but the results were not improving.

The file `thresholds.py` contains the different functions used by the pipeline to generate the binary image of each frame


![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`
This function uses the computed matrix and distortion coefficients 

I chose the hardcode the source and destination points by direct observation of images

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the lane-line pixels I used an average approach by keeping a list of values in two different objects, each for each line.
Attributes of these objects are updated when certain conditions apply. If these conditions do not apply then a window search is performed
Note that window search will always be done with the first frame (as the line objects do not contain any information)
##### Window Search
Window search is performed by using the peaks in the histogram of the image. Those peaks indicate the presence of a line.
Sliding windows are then applied as can be seen in the image below
 Window search and sliding windows are implemented in its own file `window_search.py`.
 The function used by the pipeline is `find_window_by_histogram` that returns the polynomial for each line
 
##### Using previous results
Once there is a polynomial, the values generated for each line are used to update the line objects. If these new values are outside the limits
defined to determine if the fit is correct (method `sanity_check`), the line object will set its `detected` field as False indicating that a new window search must be performed
If the values are accepted, they are added to the list of fits (up until a maximum defined in the constructor of the class) and the average is used to draw the line for the current frame

The `Line` class is defined in the `pipeline.py` file


![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated by the function `measure_curvature` in `window_search.py`
Note that the radius is calculate using hard-coded values for US. This function would require a good refactoring


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally all the elements are put together in the method `draw_final_image` of the `LaneFinderPipeline` class
Note that the method uses the warped image and the inverse perspective matrix used for the distortion to create a new warp in the original image space.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This assignment forced me to work with different color spaces until I could obtain a good representation of both yellow and white lines.
Using the Line class helped a lot to obtain a smooth video.
To summarize, the pipeline for each image is as follows:
```python
        undistort(image)
        threshold_image()
        warp()
        get_line_fit()
        measure_curvature()
        draw_final_image()

```

There are a few things I would like to improve
* Curvature is calculated using too many hard-coded values. The results are in the range required by the assigment but it could be improve
* A better tuning of thresholds would be required for the pipeline to work with other videos
* Values for the corners of the distoreted/undistorted images were obtain manually. There are automatic ways of doing it with OpenCV
