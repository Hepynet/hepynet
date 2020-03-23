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


def plot_different_mass(mass_scan_map, input_path, para_index, model = "zprime", bins = 50, range = (-10000, 10000), 
                        density = True, xlabel="x axis", ylabel="y axis"):
  """
  TODO: function should be removed since it's too dedicated.

  """
  for i, mass in enumerate(mass_scan_map):
    # load signal
    if model == "zprime":
      xs_add = np.load(input_path + '/data_npy/emu/tree_{}00GeV.npy'.format(mass))
      """ # only use emu channel currently
      xs_temp = np.load(input_path + '/data_npy/etau/tree_{}00GeV.npy'.format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      xs_temp = np.load(input_path + '/data_npy/mutau/tree_{}00GeV.npy'.format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      """
    elif model == "rpv":
      xs_add = np.load(input_path + '/data_npy/emu/rpv_{}00GeV.npy'.format(mass))
      """ # only use emu channel currently
      xs_temp = np.load(input_path + '/data_npy/etau/rpv_{}00GeV.npy'.format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      xs_temp = np.load(input_path + '/data_npy/mutau/rpv_{}00GeV.npy'.format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      """
    xs_emu = xs_add.copy()
    # select emu channel and shuffle
    xs_emu = array_utils.modify_array(xs_emu, select_channel = True, norm = True,
      shuffle = True, shuffle_seed = 486)

    # make plots
    plt.hist(xs_emu[:, para_index], bins = bins, weights = xs_emu[:,-1], 
             histtype='step', label='signal {}00GeV'.format(mass), range = range, density = density)
  plt.legend(prop={'size': 10})
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()


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


def calculate_significance(xs, xb, mass_point, mass_min, mass_max, model=None, 
                           xs_model_input=None, xb_model_input=None, use_model_cut=False):
  """
  TODO: Old function, need to be removed later

  """
  # 1st value of each xs/xb's entry must be the mass
  signal_quantity = 0.
  background_quantity = 0.
  if (model != None) and (len(xs_model_input) != 0) and (len(xb_model_input) != 0) and (use_model_cut == True):
    signal_predict_result = model.predict(xs_model_input)
    background_predict_result = model.predict(xb_model_input)
    for n, (entry, predict_result) in enumerate(zip(xs, signal_predict_result)):
      if entry[0] > mass_min and entry[0] < mass_max and predict_result > 0.5:
        signal_quantity += entry[-1]
    for n, (entry, predict_result) in enumerate(zip(xb, background_predict_result)):
      if entry[0] > mass_min and entry[0] < mass_max and predict_result > 0.5:
        background_quantity += entry[-1]
  else:
    for entry in xs:
      if entry[0] > mass_min and entry[0] < mass_max:
        signal_quantity += entry[-1]
    for entry in xb:
      if entry[0] > mass_min and entry[0] < mass_max:
        background_quantity += entry[-1]
          
  print("for mass =", mass_point, "range = (", mass_min, mass_max, "):")
  print("  signal quantity =", signal_quantity, "background quantity =", background_quantity)
  print("  significance =", signal_quantity / sqrt(background_quantity))
