#!/usr/bin/env python

import csv
import matplotlib.image as mpimg
import numpy as np
import sys



class DataParser:
  """Prepare the data from image files and csv file"""

  def __init__(self):
    self._filename = 'driving_log.csv'
    self._steering_angles = []

  def _grab_data(self):
    self._file_IDs = []

    with open('driving_log.csv', 'rb') as f:
      try:
        reader = csv.reader(f)
        for row in reader:
          self._file_IDs.append(row[0].split("center_",1)[1])
          self._steering_angles.append(float(row[3]))
      finally:
        f.close()

  def _combine_imgs(self):
    print('_combining_imgs')
    print(len(self._steering_angles))
  
    image = mpimg.imread('IMG/center_' + self._file_IDs[0])
    print('This image is:', type(image), 'with dimesions:', image.shape)
 

  '''
  External API
  '''
  def parse_data(self):
    self._grab_data()
    self._combine_imgs()

  def get_images(self):
    return (self._left_imgs, self._center_imgs, self._right_imgs)

  def get_angles(self):
    return self._steering_angles


if __name__ == '__main__':
  print('in data_parser.py')

  data_parser = DataParser()
  data_parser.parse_data()

  #print(data_parser.get_angles())

