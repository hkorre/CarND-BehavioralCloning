#!/usr/bin/env python

import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
import traceback



class DataParser:
  """Prepare the data from image files and csv file"""

  def __init__(self):
    self._img_height = 160
    self._img_width_original = 320
    self._img_channels = 3
    self._filename = 'data/driving_log.csv'

  # grabs steering angles and filenames
  def _grab_data(self):
    self._file_IDs = []
    steering_angles_list = []
    with open(self._filename, 'r') as f:
      try:
        reader = csv.reader(f)
        firstline = True
        for row in reader:
          if firstline:    #skip first line
            firstline = False
            continue
          self._file_IDs.append(row[0].split("center_",1)[1])
          steering_angles_list.append(float(row[3]))
      finally:
        f.close()
    self._steering_angles = np.asarray(steering_angles_list)

  # Returns YUV version of an image
  #  xDiv_ and yDiv_ should be 2^#
  def _grab_img_YUV(self, filename_, xDiv_, yDiv_):
    img_BGR = cv2.imread(filename_)                  #gives BGR
    img_BGR = cv2.resize(img_BGR, None, fx=1/xDiv_, fy=1/yDiv_, 
                         interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)  #gives YUV
    #TODO: resize input (maybe in data_parser.py)

  def _grab_left_img(self, file_ID_, xDiv_, yDiv_):
    return self._grab_img_YUV('data/IMG/left_' + file_ID_, xDiv_, yDiv_)

  def _grab_center_img(self, file_ID_, xDiv_, yDiv_):
    return self._grab_img_YUV('data/IMG/center_' + file_ID_, xDiv_, yDiv_)

  def _grab_right_img(self, file_ID_, xDiv_, yDiv_):
    return self._grab_img_YUV('data/IMG/right_' + file_ID_, xDiv_, yDiv_)

  def _combine_batch(self, start_, stop_, xDiv_, yDiv_):
    num_imgs = stop_-start_
    self._left_imgs = np.zeros((num_imgs, int(self._img_height/yDiv_), int(self._img_width_original/xDiv_), 3))
    self._center_imgs = np.zeros((num_imgs, int(self._img_height/yDiv_), int(self._img_width_original/xDiv_), 3))
    self._right_imgs = np.zeros((num_imgs, int(self._img_height/yDiv_), int(self._img_width_original/xDiv_), 3))
    index = 0
    for img_num in range(start_, stop_):
      self._left_imgs[index] = self._grab_left_img(self._file_IDs[img_num], xDiv_, yDiv_)
      self._center_imgs[index] = self._grab_center_img(self._file_IDs[img_num], xDiv_, yDiv_)
      self._right_imgs[index] = self._grab_right_img(self._file_IDs[img_num], xDiv_, yDiv_)
      index += 1
 

  '''
  External API
  '''
  def grab_data_info(self):
    self._grab_data()

  def combine_batch(self, start, stop, xDiv, yDiv):
    self._combine_batch(start, stop, xDiv, yDiv)

  def parse_data(self):
    self._grab_data()

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

  @property
  def center_imgs(self):
    return self._center_imgs

  @property
  def right_imgs(self):
    return self._right_imgs



if __name__ == '__main__':
  print('Running main in data_parser.py')

  try:
    data_parser = DataParser()
    data_parser.parse_data()
    data_parser.combine_batch(0, 3, 2, 2)
    print(data_parser.center_imgs.shape)
  except:
    print(traceback.format_exc())

