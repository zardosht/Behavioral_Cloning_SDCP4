#### Udacity Self-driving Car Nanodegree
# Porject 4: Behavioral Cloning

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./writeup_files/model.png "Model Visualization"
[dataset_info]: ./writeup_files/dataset_info.png "Dataset Info"
[preprocessing]: ./writeup_files/preprocessing.png "Preprocessing"
[learning_curve]: ./writeup_files/learning_curve.png "Learning Curve"


## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


#### Files Submitted & Code Quality

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

**2. Submission includes functional code**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Model Architecture and Training Strategy

**1. An appropriate model architecture has been employed**

The model is a CNN based on LeNet with 3 preprocesssing layers, 2 convolutional layers, 2 fully connected layers. Layers all use ReLU as activation function, except the output layer that has linear activation. (see `model.py:72-97`). 

**2. Attempts to reduce overfitting in the model**

The model contains dropout layers in order to reduce overfitting (`model.py` lines 88, 91, 94). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

**3. Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

**4. Appropriate training data**

Training data was chosen to keep the vehicle driving on the road. I used the simulator to drive on Track 1 clockwise and counter-clockwise. I did extra recording of smooth curve driving. I also recorded a lap in Track 2. 

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

#### 1. Dataset and Training Process

Training data was chosen to keep the vehicle driving on the road. I used the simulator to drive on Track 1 clockwise and counter-clockwise. I did extra recording of smooth curve driving. I also recorded a lap in Track 2. 

Figure below shows a summary of the dataset and a smaple image for center, left, and right cameras. 

The driving data was collected mostly with center driving. In order to train for recovering from off-the-center deviations, I used the images for left and right camera with a steering correction of `0.3`. 

The recorded driving data resulted in 6773 recorded samples each with 3 images (center, left, right).  I also augemented the data by flipping the center image and inverting the corresponding steering value. With the total number of samples equal `6773 * 4 = 27092`. These samples were shufflede and splitted into training and validation sets with `20%` for validation data. 

![alt_text][dataset_info]

I used a generator (`model.py:37-69`) to feed the data into optimizer for training. Each batch was created by loading the center image with the steering value, the left and right image with the corrected steering value, and the augmented image by flipping the center image and inverting the steering value. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

I trained the model for 10 epochs. and stoped the training when the improvement in validation loss was not very large. I then tested this model in the simulator and it passed the requirement test for Track1 (the car does not leave the drivable area of the road; see the [vidoe]("./output_video.mp4"))


#### 2. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
