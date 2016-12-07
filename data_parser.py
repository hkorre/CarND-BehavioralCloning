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

  def _format_center_img(self, file_ID):
    image = mpimg.imread('IMG/center_' + file_ID)
    start = int(self._img_height/2)
    end   = self._img_width_original - start
    image = image[:, start:end, :]
    return image



  def _combine_imgs(self):
    print('_combining_imgs')

    num_imgs = len(self._steering_angles)

    self._left_imgs = np.zeros((num_imgs, self._img_height, self._img_height, 3))
    self._center_imgs = np.zeros_like(self._left_imgs)
    self._right_imgs = np.zeros_like(self._left_imgs)

    for index in range(num_imgs):
      print(index)
      self._center_imgs[index] = self._format_center_img(self._file_IDs[index])
      
    '''
    print(self._right_imgs.shape)
    image_center = self._format_center_img(self._file_IDs[0])
    print('This image is:', type(image_center), 'with dimesions:', image_center.shape)
    '''
  
 

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

