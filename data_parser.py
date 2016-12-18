#!/usr/bin/env python

import csv
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
        reader.next()   # skip the first line
        for row in reader:
          self._file_IDs.append(row[0].split("center_",1)[1])
          steering_angles_list.append(float(row[3]))
      finally:
        f.close()
    self._steering_angles = np.asarray(steering_angles_list)

  # Returns YUV version of an image
  def _grab_img_YUV(self, filename_):
    img_BGR = cv2.imread(filename_)                  #gives BGR
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)  #gives YUV
    #TODO: resize input (maybe in data_parser.py)

  def _grab_left_img(self, file_ID_):
    return _grab_img_YUV('data/IMG/left_' + file_ID_)

  def _grab_center_img(self, file_ID_):
    return _grab_img_YUV('data/IMG/center_' + file_ID_)

  def _grab_right_img(self, file_ID_):
    return _grab_img_YUV('data/IMG/right_' + file_ID_)

  def _combine_batch(self, start_, stop_):
    num_imgs = stop_-start_
    self._left_imgs = np.zeros((num_imgs, self._img_height, self._img_width_original, 3))
    self._center_imgs = np.zeros((num_imgs, self._img_height, self._img_width_original, 3))
    self._right_imgs = np.zeros((num_imgs, self._img_height, self._img_width_original, 3))
    index = 0
    for img_num in range(start_, stop_):
      self._left_imgs[index] = self._grab_left_img(self._file_IDs[img_num])
      self._center_imgs[index] = self._grab_center_img(self._file_IDs[img_num])
      self._right_imgs[index] = self._grab_right_img(self._file_IDs[img_num])
      index += 1
 

  '''
  External API
  '''
  def grab_data_info(self):
    self._grab_data()

  def combine_batch(self, start, stop):
    self._combine_batch(start, stop)

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
  except:
    print(traceback.format_exc())

