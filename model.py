#!/usr/bin/env python

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Input, Merge, Dense, Lambda
from keras.layers import Flatten, Activation, Dropout
from keras import backend as K
from keras.engine.topology import Layer


INPUT_HEIGHT = 160
INPUT_WIDTH = 320 #160
INPUT_CHANNELS = 3

'''
VGG_HEIGHT = 224
VGG_WIDTH = 224
VGG_CHANNELS = 3
'''



class BehaviorCloner:
  """Create Keras model to train learn from driving images in simulator and 
     learn to control the car on it's own"""

  def __init__(self):
    #TODO
    return


  def _grab_data(self):
    #TODO
    return

  def _preprocess_data(self):
    #TODO
    # normalize
    #
    return

    ## Split to training and validation sets?

    '''
    # run session to resize the data
    img_placeholder = tf.placeholder("uint8", (None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    resize_op = tf.image.resize_images(img_placeholder, (VGG_HEIGHT, VGG_WIDTH), method=0)
    '''


  def build_model(self, n_hidden1_=512, n_hidden2_=512, pct_drop_=0.5):
    pool_size_ = 7


    left_input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    left_branch = Sequential()
    left_branch.add(VGG16(include_top=False, weights='imagenet', input_tensor=left_input,
      input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    #left_branch.add(AveragePooling2D(pool_size=(pool_size_, pool_size_))
    left_branch.add(Flatten())

    center_input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    center_branch = Sequential()
    center_branch.add(VGG16(include_top=False, weights='imagenet', input_tensor=center_input,
      input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    #center_branch.add(AveragePooling2D(pool_size=(pool_size_, pool_size_))
    center_branch.add(Flatten())

    right_input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    right_branch = Sequential()
    right_branch.add(VGG16(include_top=False, weights='imagenet', input_tensor=right_input,
      input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    #right_branch.add(AveragePooling2D(pool_size=(pool_size_, pool_size_))
    right_branch.add(Flatten())

    merged = Merge([left_branch, center_branch, right_branch], mode='concat')
    #merged = K.stop_gradient(merged)

    self._model = Sequential()
    self._model.add(merged)
    self._model.add(Lambda(lambda x: K.stop_gradient(x)))
    self._model.add(Dense(n_hidden1_, activation='relu', name='fully_connect1'))
    self._model.add(Dropout(pct_drop_))
    self._model.add(Dense(n_hidden2_, activation='relu', name='fully_connect2'))
    self._model.add(Dropout(pct_drop_))
    self._model.add(Dense(1, activation='sigmoid', name='final'))

    self._model.summary()



  def train_model(self, num_epochs_, batch_size_):

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # train the model
    model.fit([left_imgs, canter_imgs, right_imgs], labels, nb_epoch=num_epochs_, batch_size=batch_size_)



if __name__ == '__main__':
  print('Running main in model.py')

  behavior_cloner = BehaviorCloner()
  behavior_cloner.build_model()

