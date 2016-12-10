#!/usr/bin/env python

import csv
import matplotlib.image as mpimg
import numpy as np
import sys
import traceback



class DataParser:
  """Prepare the data from image files and csv file"""

  def __init__(self):
    self._img_height = 160
    self._img_width_original = 320
    self._img_channels = 3

    self._filename = 'driving_log.csv'
    self._steering_angles = []

  def _grab_data(self):
    self._file_IDs = []

    with open('driving_log.csv', 'r') as f:
      try:
        reader = csv.reader(f)
        for row in reader:
          self._file_IDs.append(row[0].split("center_",1)[1])
          self._steering_angles.append(float(row[3]))
      finally:
        f.close()


  def _grab_left_img(self, file_ID_):
    return mpimg.imread('IMG/left_' + file_ID_)

  def _grab_center_img(self, file_ID_):
    return mpimg.imread('IMG/center_' + file_ID_)

  def _grab_right_img(self, file_ID_):
    return mpimg.imread('IMG/right_' + file_ID_)


  def _crop_left_img(self, file_ID_):
    image = self._grab_left_img(file_ID_)
    start = 0
    end   = self._img_height
    image = image[:, start:end, :]
    return image

  def _crop_center_img(self, file_ID_):
    image = self._grab_center_img(file_ID_)
    start = int(self._img_height/2)
    end   = self._img_width_original - start
    image = image[:, start:end, :]
    return image

  def _crop_right_img(self, file_ID_):
    image = self._grab_right_img(file_ID_) 
    start = self._img_width_original - self._img_height
    end   = self._img_width_original
    image = image[:, start:end, :]
    return image



  def _combine_imgs(self):
    print('DataParser: _combining_imgs()...')

    num_imgs = len(self._steering_angles)

    self._left_imgs = np.zeros((num_imgs, self._img_height, self._img_width_original, 3))
    self._center_imgs = np.zeros_like(self._left_imgs)
    self._right_imgs = np.zeros_like(self._left_imgs)

    for index in range(num_imgs):
      if (index % 100 == 0):
        print('\tparsed {}/{}'.format(index, num_imgs))

      self._left_imgs[index]   = self._grab_left_img(self._file_IDs[index])
      self._center_imgs[index] = self._grab_center_img(self._file_IDs[index])
      self._right_imgs[index]  = self._grab_right_img(self._file_IDs[index])
  
    print('... combining imgs done')


  def _normalize_img(self, img_):
      # data from 0-255 -> -0.5-0.5
      #change type to np.float32 to accomodate negative numbers
      #  and get ready for further math
      return (img_.astype(np.float32)/255) - 0.5
 

  '''
  External API
  '''
  def parse_data(self):
    self._grab_data()
    self._combine_imgs()

  @property
  def img_height(self):
    return self._img_height

  @property
  def img_width(self):
    return self._img_width_original

  @property
  def img_channels(self):
    return self._img_channels

  @property
  def steering_angles(self):
    return self._steering_angles

  @property
  def left_imgs(self):
    return self._left_imgs

  '''
  @left_imgs.setter
  def left_imgs(self, imgs_):
    self._left_imgs = imgs_
  '''

  @property
  def center_imgs(self):
    return self._center_imgs

  '''
  @center_imgs.setter
  def center_imgs(self, imgs_):
    self._center_imgs = imgs_
  '''

  @property
  def right_imgs(self):
    return self._left_imgs

  '''
  @right_imgs.setter
  def right_imgs(self, imgs_):
    self._left_imgs = imgs_
  '''

  def preprocess_data(self):
    self._left_imgs = self._normalize_img(self._left_imgs)
    self._center_imgs = self._normalize_img(self._center_imgs)
    self._right_imgs = self._normalize_img(self._right_imgs)




if __name__ == '__main__':
  print('Running main in data_parser.py')

  try:
    data_parser = DataParser()
    data_parser.parse_data()
    data_parser.preprocess_data()
  except:
    print(traceback.format_exc())

