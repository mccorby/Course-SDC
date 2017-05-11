# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/ProcessedImages.png "Processed Image"
[image2]: ./writeup_images/histogram.png "Histogram"
[image3]: ./writeup_images/loss_epochs.png "Loss per epochs"

## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* models.py: Contains the definition of several models
* data_explorer.py: Script to do an exploration of the data
* data_processor.py: Contains a class that provides a Python generator and different functions to process the data
* trainer.py: Entry point of the application. It wires up the different components and do the training of the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2 . Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The models.py file contains the code for training the convolution neural network.
The file shows different implementations I've used during this assignment.
Each model describes a pipeline for training and validating the model, and it contains comments to explain how the code works.

The final model used is the simplified model (nvidia_simplified)


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64.
The architecture also includes three full-connected layers of 500, 100 and 20 nodes

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after the full connected layers in order to reduce overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I decided to use the data already provided by Udacity as that should be enough to train the model, the objective being for the car to be kept driving on the road
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different architectures and processing of the data.

I started by implementing a very simple architecture (simple_nn) to check that all the components were working.

I then implemented NVidia's convolution neural network as described in their paper https://arxiv.org/pdf/1604.07316.pdf
The improvement was sensible as expected. I suspected however that in this assignment the network would have a lesser impact than the processing of the data.

For that reason, I decided to implement a simplified version of NVidia's CNN and ALVINN.
Results were similar for both of them in terms of how the car is driven autonomously with ALVINN requiring more epochs to get good evaluation figures

Models are using dropout after each fully connected layer to prevent overfitting

The final step was to run the simulator to see how well the car was driving around track one.
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I decided to augment the dataset by flipping all images.
This resulted in the desired improvement and the car could autonomously complete a lap without leaving the road

I also tried other augmentations as adding shadows, change brightness without improvements worth the addition of these processes.
However these augmentations could help the model to generalise and make it useful to drive in other tracks (i.e. Track 2 of the demo)


#### 2. Final Model Architecture

The final model architecture (nvidia_simplified) consisted of a convolution neural network with the following layers

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66x200x3 image. Using a Keras Lambda layer to normalise the input data                 |
| Convolution 3x3    	| 1x1 stride, valid padding, 16 filters         |
| RELU					|												|
| Max pooling	      	| 2x2 stride                     				|
| Convolution 3x3    	| 1x1 stride, valid padding, 32 filters      	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                     				|
| Convolution 3x3    	| 1x1 stride, valid padding, 64 filters      	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                     				|
| Fully connected		| 500x100 with RELU activation                  |
| Dropout               | 0.5 keep probability                          |
| Fully connected		| 100x20 with RELU activation                   |
| Dropout               | 0.5 keep probability                          |
| Fully connected       | 20x1        									|


#### 3. Creation of the Training Set & Training Process

I relied on the data provided to the assignment and decided to go with them without further manual driving

The data is skewed (Skew is: -0.130313571372) that could make the model be biased to movements to the left.
The distribution of the steering angles can be seen in the following histogram

![alt text][image2]


To augment the data sat, I also flipped images and angles. This would give more data to the model and will provide samples of turning to the right 

![alt text][image1]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 10 as evidenced by the fact that the error loss was not being improved 

![alt text][image3]


I used an adam optimizer so that manually training the learning rate wasn't necessary.

