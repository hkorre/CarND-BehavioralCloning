#!/usr/bin/env python

import inspect
import traceback

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Input, Merge, Dense, Lambda
from keras.layers import Flatten, Activation, Dropout
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam, RMSprop

from data_parser import DataParser



class BehaviorCloner:
  """Create Keras model to train learn from driving images in simulator and 
     learn to control the car on it's own"""

  def __init__(self):
    self._data_parser = DataParser()


  '''
  External API
  '''
  def setup_data(self):
    self._data_parser.parse_data()
    self._data_parser.preprocess_data()

  def build_model(self, n_hidden1_=512, n_hidden2_=512, pct_drop_=0.5):
    #pool_size_ = 7

    input_height = self._data_parser.img_height
    input_width = self._data_parser.img_width
    input_channels = self._data_parser.img_channels

    input_imgs = Input(shape=(input_height, input_width, input_channels), name='input_tensor')

    self._model = Sequential()
    self._model.add(VGG16(include_top=False, weights='imagenet', input_tensor=input_imgs))
    self._model.add(Flatten())
    self._model.add(Dense(n_hidden1_, activation='relu', name='fully_connect1'))
    #self._model.add(Dropout(pct_drop_))
    self._model.add(Dense(n_hidden2_, activation='relu', name='fully_connect2'))
    #self._model.add(Dropout(pct_drop_))
    self._model.add(Dense(1, name='final'))

    self._model.summary()

    for layer in self._model.layers:
      print('layer...')
      print(layer.get_config()['name'])
      print(layer.get_config()['trainable'])
      print('')


  def train_model(self, num_epochs_, batch_size_):

    # setup for training
    self._model.compile(optimizer=Adam(),
                  loss='mean_squared_error')

    # train the model
    print(self._data_parser.left_imgs.shape)
    print(self._data_parser.center_imgs.shape)
    print(self._data_parser.right_imgs.shape)
    print(self._data_parser.steering_angles.shape)
    history = self._model.fit([self._data_parser.left_imgs, 
                               self._data_parser.center_imgs, 
                               self._data_parser.right_imgs], 
                               self._data_parser.steering_angles, 
                               nb_epoch=num_epochs_, 
                               batch_size=batch_size_)



if __name__ == '__main__':
  print('Running main in model.py')

  try:
    behavior_cloner = BehaviorCloner()
    #behavior_cloner.setup_data()
    behavior_cloner.build_model()
    #behavior_cloner.train_model(10, 32)
  except:
    print(traceback.format_exc())
    print('---')
    #print(inspect.getargvalues(traceback.tb_frame))
    '''
    for frame, filename, line_num, func, source_code, source_index in inspect.stack():
      print( '{}[{}]\n  -> {}'.format(filename, line_num, source_code[source_index].strip()) )
      print(inspect.getargvalues(frame))
      print('')
    '''

