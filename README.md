# CarND-BehavioralCloning
Project #3 for Udacity Self-Driving Car Nanodegree: Behavioral Cloning

## Model Architecture
My model uses 3 Convolutional Layers, 2 Hidden Layers, and 1 output node. I also make use of RELUs, MaxPooling and Dropout.

### Preprocessing
I have a Lambda layer that normalizes the data from -1<>+1

### Picking out features
* Convoluational Layers give non-linearity
* RELUs also give non-linearity

### Data size
MaxPooling is used to bring down the number of variables

### Classification and Solution
* The 2 hidden layers are used to pick useful features to make decisions on
* The last output neuron uses the features to decide a steering angle

### Preventing overfitting
Dropout is used to prevent overfitting


## Training

### Data
* I gathered ~10k images, almost all of which were of recovery. To do this I:
  * Drove up to edges of the track
  * Started recording
  * Drove car back to center of track
  * Stopped recording
* I only used center images, as that was what would be fed to the car during autonomous driving
* All 10k images are from the 'Left' track, as that is what we would be tested on

### Data Feedthrough
I use a data generator to feed in small batches of images, so that I don't use up too much memory.

### Optimization & Learning
I use Adam optimizer to get automatic learning rate with momentum

### Hardware
I trained the net on AWS EC2 instance of g2.2xlarge
