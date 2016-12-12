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
              self._data_parser.preprocess_data()
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

    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_imgs)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # a few fully connected layers
    x = Dense(n_hidden1_, activation='relu', name='fully_connect1')(x)
    x = Dropout(pct_drop_)(x)
    x = Dense(n_hidden2_, activation='relu', name='fully_connect2')(x)
    x = Dropout(pct_drop_)(x)
    predictions = Dense(1, name='final')(x)

    # this is the model we will train
    self._model = Model(input=base_model. input, output=predictions)
    self._model.summary()

    set_trainable = False
    for layer in self._model.layers:
      if 'trainable' in layer.get_config():
        layer.trainable = set_trainable
      if layer.get_config()['name']=='block5_pool': #end of VGG16
        break
      
    #self._print_trainable_layers()


  def train_model(self, num_epochs_, batch_size_):
    print('BehaviorCloner: train_model()...')

    # setup for training
    self._model.compile(optimizer='adam', loss='mean_squared_error')

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

