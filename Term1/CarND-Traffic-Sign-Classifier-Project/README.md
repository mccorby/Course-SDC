#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./unseen_images/front_right.png "Traffic Sign 1"
[image5]: ./unseen_images/no_entry.png "Traffic Sign 2"
[image6]: ./unseen_images/no_entry_bricks.png "Traffic Sign 3"
[image7]: ./unseen_images/no_passing.png "Traffic Sign 4"
[image8]: ./unseen_images/no_vehicles.png "Traffic Sign 5"
[image9]: ./unseen_images/roundabout_posters.png "Traffic Sign 6"
[image10]: ./unseen_images/speed_limit_40.png "Traffic Sign 7"
[image11]: ./unseen_images/speed_limit_50.png "Traffic Sign 8"
[image12]: ./unseen_images/stop.png "Traffic Sign 9"
[image13]: ./unseen_images/working.png "Traffic Sign 10"
[image14]: ./unseen_images/yellow_diamond.png "Traffic Sign 11"
[image15]: ./unseen_images/yield.png "Traffic Sign 12"

[support_img1]: ./supporting_images/samples_count.png "Samples per class"
[support_img2]: ./supporting_images/grayscale.png "Grayscaling"
[support_img3]: ./supporting_images/rotated.png "Rotated"
[support_img4]: ./supporting_images/grayscale.png "Blurred"


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
TODO Update link to project. Maybe not necessary. This is the README
You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard Python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34779
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.
The graph shows the number of samples per class (a class being a type of traffic sign)

![Samples per class][support_img1]


###Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is demonstrated the color feature does not improve the performance of image recognition. Furthermore a grayscale image only needs a channel which makes the dataset lighter to process

As a last step, I normalized the image data because data should be in a similar range when processed by the network
This  helps avoiding the gradients running wild

I decided to generate additional data because having more samples helps the model to be more accurate. Note that by using regularization techniques overfitting will be reduced

To add more data to the the data set, I used the following techniques:
 * Rotation. I applied a random rotation of less than 10 degrees
 * Gaussian smoothing

These techniques make available more data to the model and it's easily labeled while being slightly different

Here is an example of an original image and its augmented versions:

![alt text][support_img2] ![alt text][support_img3] ![alt text][support_img4]

 The final dataset used to train has around 100k samples


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model I chose was a LeNet architecture (as proposed in the previous lesson) with variations on the feature kernels and depths. 
Though I tried adding dropout I could not see much differences in the performance yet the execution time was sensible increased

Note that the input is 32x32x1 because the network works with grayscale images. Here is where we can see now the convenience of using this transformation to the image

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 8x8    	| 1x1 stride, valid padding, outputs 25x25x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x20   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 8x8x16     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x16   				|
| Fully connected		| 256x120 with RELU activation                  |
| Fully connected		| 120x84 with RELU activation                   |
| Fully connected       | 84x43        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer to minimize the cross-entropy (cost) of the network.
Classes are represented by a one-hot vector
Hyper-parameters:
 * Epochs: 10
 * Batch size: 128
 * Learning rate: 0.001
 * When using dropout, the probability of keeping a connection was set to 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.968 
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? I started with LeNet architecture, a well-know solution for imaging processing.
LeNet is based on convolutional layers. The initial architecture consisted of two of these layers plus two full connected layers
* Why did you believe it would be relevant to the traffic sign application? As said, convolutional networks are a good fit for image processing.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The performance of the model is good though it can be improved. Training and validation are quite high with training being superior to that of a human being.
More epochs and more data would increase the model's performance


Without further changes, LeNet was giving a performance of 0.993 for the training set. The validation and test accuracies were giving lower values (0.911 and 0.894)
From this point, I started modifying the network:
* Model #2: I first wanted to see if by regularizing the network the test accuracy would improve. This is because the previous value was too low
  * Dropout increased the test accuracy to 0.911
* Model #3: I decided to modify the convolutional layers to provide more features. I was expecting to increase the accuracy in all the sets
  * The result was an increase of accuracy in training, validation and test. For this last one, the value was at 0.931, this is, above the minimum requirement for this assignment!
* Model #4: By adding more data I was expecting to keep increasing the performance of the model. I augmented the dataset as previously discussed obtaining the following results:
  * Training: 0.993
  * Validation: 0.968
  * Test: 0.940
 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13] ![alt text][image14] ![alt text][image15]

Some of the images are difficult to predict because they are either not present in the original dataset
 or because they look too similar to other signs or are too distorted for the model to know them.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or right  | Go straight or right							| 
| No entry     			| No entry 										|
| No entry (diff bg)	| No entry 										|
| Yield					| Yield											|
| No passing      		| No passing					 				|
| No entry      		| No entry  					 				|
| Roundabout mandatory  | Roundabout mandatory                          |
| Road work             | Road work                                     |
| Priority road         | Priority road                                 |
| Stop      			| Speed limit (60 km/h)							|
| Speed limit (40 km/h)	| Turn right ahead  							|
| Speed limit (50 km/h)	| Turn right ahead  							|


The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The next image shows the top-5 probabilities for each image



For the first image, the model is sure that this is a "Go straight or right" sign
(probability of 0.98). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Go straight or right							| 
| .01     				| Keep right									|
| ~ .00					| Turn left ahead								|
| ~ .00	      			| End of all speed and passing limits			|
| ~ .00				    | Priority road      							|


For the second and thirs images it had no doubts and predicted a "No entry" with a 100% of accuracy

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry          							| 

Fourth image. It predicted with a 100% that it was a "No passing" sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing          							|
 
 Fifth image: It was a "No entry" image. It predicted it with a 82.5% of probability.
 
 | Probability         	|     Prediction	        					| 
 |:--------------------:|:---------------------------------------------:| 
 | 0.825         		| No entry          							|
 | 0.1           		| End of speed limit (80 km/h)					|
 | 0.048           		| End of all speed and passing limits			|
 | 0.006           		| Speed limit (20 km/h)             			|
 | 0.002           		| Dangerous curve to the right         			|
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


