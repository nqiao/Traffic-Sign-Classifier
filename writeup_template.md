# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image4]: ./examples/test1.jpg "Traffic Sign 1"
[image5]: ./examples/test2.jpg "Traffic Sign 2"
[image6]: ./examples/test3.jpg "Traffic Sign 3"
[image7]: ./examples/test4.jpg "Traffic Sign 4"
[image8]: ./examples/test5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale image has only one layer which could decrease the number of parameters in the neural network and save training time. 


As the last step, I normalized the image data. In my opinion, convolutional neural networks don't rely on normalization much. I did because the course suggested to do so and it might increase the efficiency of the CNN. Since the result of normalization is in the range of [-1,1], it's unable to plot the result.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                               | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x6                      |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x16                   |
| Fully connected        | inputs 400, outputs 120                        |
| RELU                    |                                                |
| Fully connected        | inputs 120, outputs 84                        |
| RELU                    |                                                |
| Fully connected        | inputs 84, outputs 43                         |
| Softmax                |                                               |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The parameters of my final model:  
batch size: 256: 
learning rate: 0.001;  
epoch: 70
optimizer: adam;  
loss function: **weighted cross entropy** (the training set is unbalanced, use this weighted cross-entropy to make minor class weigh greater than the major class, preventing the model tending to predict major classes)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results of the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.946
* test set accuracy of 0.931
* 
I tried:  
1. Grayscale or not. RGB inputs always have a better result in my case. This makes sense because RGB inputs carry more information than grayscale image. Grayscale saves training time but I treat accuracy as most important. 
2. Filter size. I tried filter size of 3x3 and 5x5. There is no big difference between this two option, 5x5 filters have slightly better performance in my case.
3. Drop out layer (or not) after the first two dense layer(keep_prob=0.75). Since there are already max-pooling layers right after two convolutional layers, I tried drop out technique after two dense layers. I printed out the training set accuracy, as well as the validation, set accuracy after every epoch and it turns out my model seems underfit since the validation set accuracy is always increasing. So I decided not to use drop out layers, and validation accuracy increased after throwing away the drop out layer.
4. Learning rate (0.001 or 0.01): learning rate of 0.01 makes the neural network learns faster but not as good as 0.001 in the end. The original 0.001 learning rate is pretty appropriate especially after a normalization is done. No need to try more values.
5. Batch size: I set it 256 since my GPU can handle this batch size. No more tries because I couldn't see much benefit when changing batch size.
6. Loss function. I used weighted cross-entropy as my loss function. As I mentioned before, the training set is not balanced: 10x difference between the maximum and minimum class. I weighed each class according to the times it appears in the training set. With weighed cross entropy, I make the neural network treats mistakes in minor class more importantly. The training set accuracy increased much and validation set accuracy increased slightly as well while other conditions keep the same.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because it's hard for the model to distinguish different speed limit signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign              | Stop sign                                       | 
| Speed limit (30km/h)                 | Speed limit (50km/h)                                         |
| Turn right ahead                    | Turn right ahead                                            |
| Right-of-way at the next intersection              | Right-of-way at the next intersection                                     |
| No entry        | No entry                               |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares lower to the accuracy on the test set of 93.1%. It made a mistake when predicting speed limit (30km/h). The model is not good enough to distinguish different speed limit signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 38th and 39th cell of the Ipython notebook.

For all five test examples, the model is pretty confidence to make the prediction although it made a mistake. The probabilities to the final predictions are almost 1.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


