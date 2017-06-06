**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_feature.png
[image2]: ./output_images/combined.png
[image3]: ./output_images/multiple-windows-detection-1.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/boxes-detection.png
[video1]: ./test_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The features are extracted in the `feature_extraction.py` file (see the different functions to do it). In particular, HOG features are extracted 
by the function `get_hog_features`. 
The combined feature vector for a single image is obtained by using the `extract_features` function

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.
I tried different color spaces but `YCrCb` seemed to be giving the best results. This was in consonance with the recommendations I took from some people.
 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features in the three channels, color features and spatially binned color. This is done in the `feature_extraction.py` 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions using several scale values. Eventually the ones that returned better results were 1.25 and 1.5.
The window search is done only in a defined ROI which roughly removes the image above the horizon

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (1.25 and 1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame with its corresponding heatmaps:

![alt text][image4]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In my opinion, the pipeline is not very robust in the sense that the model imposses too many restrictions to the input data. Once a selection of features has been done it has to be replicated when predicting.
Though this is a common issue in any ML problem (the data must follow the input format used during the training) I found that the preprocessing was excessive and that a pure ML approach (CNN for instance) could do the work better and faster

I tuned the parameters of the model using `GridSearchCV` but this proved to be unnecessary: the default values of Linear SVC were doing the job quite well (the model has a .9804 of accuracy)

Tuning the threshold in the heatmap was also challenging but eventually helped reducing the number of false positives.


