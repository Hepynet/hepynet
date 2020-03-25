# -*- coding: utf-8 -*-
"""Functions used for pDNN training.

This module is a collection of functions used for pDNN training. Include: array
manipulation, making plots, evaluation functions and so on.

"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import classification_report, accuracy_score, auc

from lfv_pdnn.common import array_utils


def get_mass_range(mass_array, weights, nsig=1):
  """Gives a range of mean +- sigma
  
  Note:
    Only use for single peak distribution

  """
  average = np.average(mass_array, weights=weights)
  variance = np.average((mass_array-average)**2, weights=weights)
  lower_limit = average - np.sqrt(variance) * nsig
  upper_limit = average + np.sqrt(variance) * nsig
  return lower_limit, upper_limit


def get_valid_feature(xtrain):
  """Gets valid inputs.
  
  Note:
    indice -2 is for channel
    indice -1 is for weight
  
  """
  xtrain = xtrain[:,:-2]
  return xtrain


def generate_shuffle_index(array_len, shuffle_seed=None):
  """Generates array shuffle index.
  
  To use a consist shuffle index to have different arrays shuffle in same way.

  """
  shuffle_index = np.array(range(array_len))
  if shuffle_seed is not None:
    np.random.seed(shuffle_seed)
  np.random.shuffle(shuffle_index)
  return shuffle_index

def get_mean_var(array, axis=None, weights=None):
  """Calculate average and variance of an array."""
  average = np.average(array, axis=axis, weights=weights)
  variance = np.average((array-average)**2, axis=axis, weights=weights)
  return average, variance + 0.000001


def norarray(array, average=None, variance=None, axis=None, weights=None):
  """Normalizes input array for each feature.
  
  Note:
    Do not normalize bkg and sig separately, bkg and sig should be normalized
    in the same way. (i.e. use same average and variance for normalization.)

  """
  if (average is None) or (variance is None):
    print("Warning! unspecified average or variance.")
    average, variance = get_mean_var(array, axis=axis, weights=weights)
  output_array = (array.copy() - average) / np.sqrt(variance)
  return output_array


def norarray_min_max(array, min, max, axis=None):
  """Normalizes input array to (-1, +1)"""
  middle = (min + max) / 2.0
  output_array = array.copy() - middle
  if max < min:
    print("ERROR: max shouldn't be smaller than min.")
    return None
  ratio = (max - min) / 2.0
  output_array = output_array / ratio


def split_and_combine(xs, xb, test_rate=0.2, shuffle_combined_array=True, shuffle_seed=None):
  """Prepares array for training & validation

  Args:
    xs: numpy array
      Siganl array for training.
    xb: numpy array
      Background array for training.
    test_rate: float, optional (default = 0.2)
      Portion of samples (array rows) to be used as independant test samples.
    shuffle_combined_array: bool, optional (default=True)
      Whether to shuffle outputs arrays before return.
    shuffle_seed: int or None, optional (default=None)
      Seed for randomization process.
      Set to None to use current time as seed.
      Set to a specific value to get an unchanged shuffle result.
  
  Returns:
    x_train/x_test/y_train/y_test: numpy array
      Array for training/testing.
      Contain mixed signal and background. 
    xs_test/xb_test: numpy array
      Array for scores plottind.
      Signal/bakcground separated.
      
  """
  ys = np.ones(len(xs))
  yb = np.zeros(len(xb))

  xs_train, xs_test, ys_train, ys_test = array_utils.shuffle_and_split(
    xs, ys, split_ratio=test_rate, shuffle_seed=shuffle_seed)
  xb_train, xb_test, yb_train, yb_test = array_utils.shuffle_and_split(
    xb, yb, split_ratio=test_rate, shuffle_seed=shuffle_seed)

  x_train = np.concatenate((xs_train, xb_train))
  y_train = np.concatenate((ys_train, yb_train))
  x_test = np.concatenate((xs_test, xb_test))
  y_test = np.concatenate((ys_test, yb_test))

  if shuffle_combined_array:
    # shuffle train dataset
    shuffle_index = generate_shuffle_index(
      len(y_train),
      shuffle_seed=shuffle_seed
      )
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]
    # shuffle test dataset
    shuffle_index = generate_shuffle_index(
      len(y_test),
      shuffle_seed=shuffle_seed
      )
    x_test = x_test[shuffle_index]
    y_test = y_test[shuffle_index]
  
  return x_train, x_test, y_train, y_test, xs_train, xs_test, xb_train, xb_test


def unison_shuffled_copies(*arr):
  """
  TODO: Old function, need to be removed later

  """
  assert all(len(a) for a in arr)
  p = np.random.permutation(len(arr[0]))
  return (a[p] for a in arr)
