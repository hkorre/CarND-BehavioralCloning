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
Example images are in XX


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
