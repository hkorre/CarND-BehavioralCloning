#!/usr/bin/env python

import inspect
import json
import traceback

from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Merge, Dense, Lambda
from keras.layers import Flatten, Activation, Dropout
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D
from keras.layers import ELU
from keras.layers import MaxPooling2D

from data_parser import DataParser



class BehaviorCloner:
  """Create Keras model to train learn from driving images in simulator and 
     learn to control the car on it's own"""

  def __init__(self):
    self._data_parser = DataParser()


  def _print_trainable_layers(self):
    print('Trainable layer?')
    for layer in self._model.layers:
      print(layer.get_config()['name'])
      if 'trainable' in layer.get_config():
        print('\t{}'.format(layer.get_config()['trainable']))
    
  def _generator_creator(self, labels_, batch_size_):
      def _f():
          start = 0
          end = start + batch_size_
          num_imgs = labels_.shape[0]
  
          while True:
              self._data_parser.combine_batch(start, end)
              #self._data_parser.preprocess_data()
              X_batch = self._data_parser.center_imgs
              y_batch = labels_[start:end]
              start += batch_size_
              end += batch_size_
              if start >= num_imgs:
                start = 0
                end = batch_size_
              if end >= num_imgs:
                end = num_imgs
  
              #print(start, end)
              yield (X_batch, y_batch)
  
      return _f


  '''
  External API
  '''
  def setup_data(self):
    self._data_parser.parse_data()
    #self._data_parser.preprocess_data()

  def build_model(self, n_hidden1_=512, n_hidden2_=512, pct_drop_=0.5):

    input_height = self._data_parser.img_height
    input_width = self._data_parser.img_width
    input_channels = self._data_parser.img_channels

    input_imgs = Input(shape=(input_height, input_width, input_channels), name='input_tensor')

    '''
    # comma.ai model
    self._model = Sequential()
    self._model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(input_height, input_width, input_channels),
              output_shape=(input_height, input_width, input_channels)))
    self._model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    self._model.add(ELU())
    self._model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    self._model.add(ELU())
    self._model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    self._model.add(Flatten())
    self._model.add(Dropout(.2))
    self._model.add(ELU())
    self._model.add(Dense(512))
    self._model.add(Dropout(.5))
    self._model.add(ELU())
    self._model.add(Dense(1))
    self._model.summary()
    '''

    self._model = Sequential()
    # normalize -1<>+1
    self._model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(input_height, input_width, input_channels),
              output_shape=(input_height, input_width, input_channels)))
    # Conv Layer #1
    self._model.add(Convolution2D(16, 5, 5, border_mode='same'))
    self._model.add(Activation('relu'))
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    # Conv Layer #2
    self._model.add(Convolution2D(32, 5, 5, border_mode='same'))
    self._model.add(Activation('relu'))
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    # Conv Layer #3
    self._model.add(Convolution2D(64, 5, 5, border_mode='same'))
    self._model.add(Activation('relu'))
    self._model.add(MaxPooling2D(pool_size=(2,2)))

    self._model.add(Flatten())

    # Hidden Layer #1
    self._model.add(Dense(128))
    self._model.add(Activation('relu'))
    self._model.add(Dropout(0.25))

    # Hidden Layer #2
    self._model.add(Dense(128))
    self._model.add(Activation('relu'))
    self._model.add(Dropout(0.25))

    # Answer
    self._model.add(Dense(1))
    self._model.summary()



  def train_model(self, num_epochs_, batch_size_):
    print('BehaviorCloner: train_model()...')

    # setup for training
    #self._model.compile(optimizer='adam', loss='mean_squared_error')
    self._model.compile(optimizer="adam", loss="mse")

    # train the model
    train_gen = self._generator_creator(self._data_parser.steering_angles,
                                        batch_size_)
    num_imgs = self._data_parser.steering_angles.shape[0]
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
    behavior_cloner.build_model()

    test_num_epochs = 5
    test_batch_size = 16
    behavior_cloner.train_model(test_num_epochs, test_batch_size)

    behavior_cloner.save_model()

    print('... main done')
  except:
    print(traceback.format_exc())

