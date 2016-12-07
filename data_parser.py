#!/usr/bin/env python

import csv
import matplotlib.image as mpimg
import numpy as np
import sys



class DataParser:
  """Prepare the data from image files and csv file"""

  def __init__(self):
    self._img_height = 160
    self._img_width_original = 320

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

  def _format_left_img(self, file_ID):
    image = mpimg.imread('IMG/left_' + file_ID)
    start = 0
    end   = self._img_height
    image = image[:, start:end, :]
    return image

  def _format_center_img(self, file_ID):
    image = mpimg.imread('IMG/center_' + file_ID)
    start = int(self._img_height/2)
    end   = self._img_width_original - start
    image = image[:, start:end, :]
    return image

  def _format_right_img(self, file_ID):
    image = mpimg.imread('IMG/right_' + file_ID)
    start = self._img_width_original - self._img_height
    end   = self._img_width_original
    image = image[:, start:end, :]
    return image



  def _combine_imgs(self):
    print('DataParser: _combining_imgs()...')

    num_imgs = len(self._steering_angles)

    self._left_imgs = np.zeros((num_imgs, self._img_height, self._img_height, 3))
    self._center_imgs = np.zeros_like(self._left_imgs)
    self._right_imgs = np.zeros_like(self._left_imgs)

    for index in range(num_imgs):
      if (index % 100 == 0):
        print('\tparsed {}/{}'.format(index, num_imgs))

      self._left_imgs[index]   = self._format_left_img(self._file_IDs[index])
      self._center_imgs[index] = self._format_center_img(self._file_IDs[index])
      self._right_imgs[index]  = self._format_right_img(self._file_IDs[index])
  
 

  '''
  External API
  '''
  def parse_data(self):
    self._grab_data()
    self._combine_imgs()

  @property
  def steering_angles(self):
    return self._steering_angles

  @property
  def left_imgs(self):
    return self._left_imgs

  @property
  def center_imgs(self):
    return self._center_imgs

  @property
  def right_imgs(self):
    return self._left_imgs


if __name__ == '__main__':
  print('Running main in data_parser.py')

  data_parser = DataParser()
  data_parser.parse_data()

