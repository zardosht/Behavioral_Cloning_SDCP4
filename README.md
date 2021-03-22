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

I trained the model for 10 epochs and stoped the training when the improvement in validation loss was not very large. I then tested this model in the simulator and it passed the requirement test for Track1 (the car does not leave the drivable area of the road). Figure below shows the learning curve. As can be seen, the validation loss has not plateued yet and the rate of decrease in validation loss is still high. This model however performs well enough for the requirements of this project. 

![alt_text][learning_curve]

#### 2. Solution Design Approach

I first created a dummy model to test the pipeline for training and testing in the simulator. 

I then decided to create a model based on LeNet. I started with the LeNet architecture and adopted it to my image size and output for MSE loss. I also added Lambda layers for preprocessing: one layer for nomralizing the data to zero mean with values between `[-0.5 , 0.5]`, and a layer for cropping the upper and lower part of the image so that only the road portion remains for training. 

However training of this model with the original image size was too slow and lead to out-of-memory errors on my machine. So I added a layer to resize the images. 

I then collected about 3000 samples and trained the model. The validation loss showed overfitting. In order to make training faster and reduce the overfitting, I decided to reduce the parameters of the model: I reduced the filter size for the conv layers and removed, reduced the number of nodes in fully connected layers, and removed one of the fully connected layers. The training with this model was fast with 3000 samples and passed the requirement for track 1, with a validation loss of 0.03.

I then tested this model on track 2, which performed poorly. So I collected more data from track 2, and trained the model. This time model was underfitting. So I added the discarded FC layer back, and added dropout to prevent overfitting. I trained with this architecture for only 10 epochs with about 6700 samples from both tracks. The resulting model again passes the requirement of the track 1. I also perfomrs surprisingly good for track 2. However drives off the road for very sharp turns. This can be be fixed by collecting more data for sharp curves in track2 and training the model for more epochs. 

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

#### 3. Final Model Architecture

The final model architecture (`model.py` lines 72-97) consisted of a convolution neural network with the following layers and layer sizes: 

The first three layers do preprocessing. The input image has shape `(160, 320, 3)`. The first `Lambda` layer resizes the image to `(40, 80, 3)`. The next layer crops out the upper and lower portion of the image by factor `0.375 * height` for the top cropping and `0.125 * image_height` for bottom cropping. I defined these cropping factors manully by examining multiple images. Third preprocessing layer normalizes the input to have zero mean and values between  `[-0.5 , 0.5]`.  Figure below shows the result of resizing and cropping the input: 

![alt_text][preprocessing]

After preprocessing follow two Conv and MaxPooling layer: first conv layer with 5 filter of size 3x3 and second one with 10 filters of size 3x3. Both layers use ReLU activation. Other parameters (stride, padding) are left as Keras defaults. 

Follwing the conv layers is a flatten layer and then two fully connected layers. The out put layer is 1 node output the steering value. 


Here is a visualization of the architecture: 

![alt text][model]
