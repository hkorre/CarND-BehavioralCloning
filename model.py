#!/usr/bin/env python

import numpy as np
import json
import traceback

from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.layers import Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ELU
from keras.regularizers import l2

from data_parser import DataParser



class BehaviorCloner:
  """Create Keras model to train learn from driving images in simulator and 
     learn to control the car on it's own"""

  def __init__(self):
    self._data_parser = DataParser()


  def _combine_LCR(self, labels_):
    left_imgs = self._data_parser.left_imgs
    center_imgs = self._data_parser.center_imgs
    right_imgs = self._data_parser.right_imgs
    total_imgs = np.concatenate((left_imgs, center_imgs, right_imgs))

    left_labels = np.copy(labels_) + 0.1
    center_labels = np.copy(labels_) + 0.1
    right_labels = np.copy(labels_) - 0.1
    total_labels = np.concatenate((left_labels, center_labels, right_labels))

    return total_imgs, total_labels


  def _generator_creator(self, labels_, batch_size_, xDiv_, yDiv_):
      def _f():
          start = 0
          end = start + batch_size_
          num_imgs = labels_.shape[0]
  
          while True:
              self._data_parser.combine_batch(start, end, xDiv_, yDiv_) #setup data
              X_batch, y_batch = self._combine_LCR(labels_[start:end])  #get data
              start += batch_size_
              end += batch_size_
              if start >= num_imgs:
                start = 0
                end = batch_size_
              if end >= num_imgs:
                end = num_imgs
  
              yield (X_batch, y_batch)
  
      return _f


  '''
  External API
  '''
  def setup_data(self):
    self._data_parser.parse_data()

  # Build model based on
  # Nvidia "End to End Learning for Self-Driving Cars"
  def build_model(self, xDiv_, yDiv_):

    input_height = int(self._data_parser.img_height/yDiv_)
    input_width = int(self._data_parser.img_width/xDiv_)
    input_channels = self._data_parser.img_channels

    self._model = Sequential()

    # normalize -1<>+1
    self._model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(input_height, input_width, input_channels),
              output_shape=(input_height, input_width, input_channels)))


    # Conv Layer #1 (depth=24, kernel=5x5, stride=2x2)
    self._model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',
                    W_regularizer=l2(0.01)))
    self._model.add(ELU())

    # Conv Layer #2 (depth=36, kernel=5x5, stride=2x2)
    self._model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same',
                    W_regularizer=l2(0.01)))
    self._model.add(ELU())

    # Conv Layer #3 (depth=48, kernel=5x5, stride=2x2)
    self._model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same',
                    W_regularizer=l2(0.01)))
    self._model.add(ELU())

    # Conv Layer #4 (depth=64, kernel=3x3, stride=1x1)
    self._model.add(Convolution2D(64, 3, 3, border_mode='same',
                    W_regularizer=l2(0.01)))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer #5 (depth=64, kernel=3x3, stride=1x1)
    self._model.add(Convolution2D(64, 3, 3, border_mode='same',
                    W_regularizer=l2(0.01)))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))

    self._model.add(Flatten())

    # Hidden Layer #1
    self._model.add(Dense(100, W_regularizer=l2(0.01)))
    self._model.add(ELU())
    #self._model.add(Dropout(0.25))

    # Hidden Layer #2
    self._model.add(Dense(50, W_regularizer=l2(0.01)))
    self._model.add(ELU())
    #self._model.add(Dropout(0.25))

    # Hidden Layer #3
    self._model.add(Dense(10, W_regularizer=l2(0.01)))
    self._model.add(ELU())
    #self._model.add(Dropout(0.25))

    # Answer
    self._model.add(Dense(1))
    self._model.summary()



  def train_model(self, num_epochs_, batch_size_, xDiv_, yDiv_):
    print('BehaviorCloner: train_model()...')

    # setup for training
    self._model.compile(optimizer="adam", loss="mse")

    # train the model
    train_gen = self._generator_creator(self._data_parser.steering_angles,
                                        batch_size_, xDiv_, yDiv_)
    num_imgs = self._data_parser.steering_angles.shape[0]*3   #3x for left, center, right
    history = self._model.fit_generator(train_gen(), num_imgs, num_epochs_)

    print('... train_model() done')


  def save_model(self):
    model_json = self._model.to_json()
    with open('model.json', 'w') as outfile:
      json.dump(model_json, outfile)

    self._model.save_weights('model.h5')


if __name__ == '__main__':
  print('Running main in model.py')

  try:
    behavior_cloner = BehaviorCloner()
    behavior_cloner.setup_data()

    x_down_sample = 4
    y_down_sample = 4
    behavior_cloner.build_model(x_down_sample, y_down_sample)

    test_num_epochs = 3
    test_batch_size = 8
    behavior_cloner.train_model(test_num_epochs, test_batch_size, 
                                x_down_sample, y_down_sample)

    behavior_cloner.save_model()

    print('... main done')
  except:
    print(traceback.format_exc())

