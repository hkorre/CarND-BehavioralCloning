# CarND-BehavioralCloning
Project #3 for Udacity Self-Driving Car Nanodegree: Behavioral Cloning

## Data

### get_data.sh
I access the data provided by Udacity: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

get_data.sh downloads from this link.

### data_parser.py
I define a DataParser class that:
* Access the CSV file
* Stores the suffix of filename (e.g. 2016_12_11_22_52_25_418.jpg) in an array
* Stores the recorded steering angles in an array
* Provides left, center, and right images in batches (i.e. a few images at a time) as arrays

### model.py
Defines a generator that is called by model’s learning function to prepare chunks of data for learning. It grabs a random chunk of left, center, and right images and the corresponding steering angles. For each grouping of steering angle, 1 left, center, and right images, the generator does the following:
* Picks left, center, or right images with equal probability (33%)
* If the left or right images are picked, the steering angle is modified accordingly
* Image/steering angles are sometimes ignored - i.e. the higher the absolute value of the steering angle, the more likely we are to use the image for training
* We flip an image and steering angle with 50% probability
We then add the image and steering angle to the training batch

### Example Images
Example images are in example_images/


## Model Derivation
I began with the “Nvidia Model” from the paper - End to End Learning for Self-Driving Cars, April 25, 2016 (the paper is included in the repo as pdf). I then began adjusting the model and adding new features.

Features added include:
* Normalization
* Resizing of image
* Dropout
* Color Transform

Modifications include:
* Change number of ConvNet layers
* Changing ConvNet kernel sizes
* Changing boarder mode
* Changing RELU to ELU
* Adjusting ConvNet depths


## Simulator
Simulator Available at:
* Linux - https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip
* MacOS - https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip
* Windows 32-bit - https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip
* Windows 64-bit - https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip


## Model Architecture

### Normalization
I use a Keras lambda function to normalize the data between -1 to 1. Putting normalization in Keras allows the operation to be parallelized in GPUs and I don’t have to normalize manually when running the model during testing in the simulator in autonomous mode

### Color Transform
There is a 1x1,depth3 convolutional layer. It’s job is color space transformation. We could use OpenCV to do a color space transform, but it’s not clear what color space or spaces are most useful. Adding color transformation as a convolutional layer allows back-propagation to surmise the most useful color channels. Also, again since it’s in Keras, it is more efficient.

### Feature Extraction
There are 4 ConvNet layers. Each has:
* 2D Convolution
* ELU activation function
* Max Pooling
* Dropout

For the first two 2D Convolutions, we first do 5x5 to extract large features. Then the later two convolutions, we do 3x3 to extract groupings of features.

For the activation we use ELU instead of RELU, which was talked about in the lectures. With RELU some neurons can become inactive because the negative half of the RELU sends them to 0. ELU uses both positive and negative values and can train systems more consistently:
http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/

We use max pooling to bring down the dimensionality of the training and yield less data.

We use dropout to prevent overfitting to the specific images that the system is trained on. 


### Decision Making
First the data is flattened. Then 3 hidden layers are used, of sizes 100, 50, and 10 neurons. Each of these has a ELU activation function. Lastly, there is 1 output neuron.

## Validation
I validate the model by:
* Create a generator that only returns back center images and steering angles
* Run evaluate_generator() which runs feedforward on the images and compares them the steering angle, resulting in a loss value
