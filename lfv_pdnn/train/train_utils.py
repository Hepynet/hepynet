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
#from sklearn.model_selection import train_test_split

from ..common.common_utils import *


def get_input_array(sig_dict, sig_key, bkg_dict, bkg_key, channel_id,
                    rm_neg=True):
  """Get training array from dictionary and select channel.
  
  Args:
      rm_neg: bool, optional (default = True)
          whether to remove negtive weight events
  
  """
  xs = modify_array(sig_dict[sig_key], weight_id=-1, select_channel=True,
                    channel_id=channel_id, remove_negative_weight=rm_neg)
  xb = modify_array(bkg_dict[bkg_key], weight_id=-1, select_channel=True,
                    channel_id=channel_id, remove_negative_weight=rm_neg)
  return xs, xb


def get_part_feature(xtrain, feature_id_list):
  """Gets sub numpy array using given column index"""
  xtrain = xtrain[:,feature_id_list]
  return xtrain


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


def modify_array(input_array, weight_id=None, remove_negative_weight=False,
                 select_channel=False, channel_id=None, 
                 select_mass=False, mass_id=None, mass_min=None, mass_max=None,
                 reset_mass=False, reset_mass_array=None, reset_mass_id=None,
                 norm=False, sumofweight=1000,
                 shuffle=False, shuffle_seed=None):
  """Modifies numpy array with given setup.
  
  Args:
    input_array: numpy array
      Array to be modified.
    weight_id: int or None, optional (default=None)
      Column index of weight value.
    remove_negative_weight: bool, optional (default=False) 
      Whether to remove events with negative weight.
    select_channel: bool, optional (default=False) 
      Whether to select specific channel. 
      If True, channel_id should not be None.
    channel_id: int or None, optional (default=None) 
      Column index of channel name.
    select_mass: bool, optional (default=False)
      Whether to select elements within cerntain mass range.
      If True, mass_id/mass_min/mass_max shouldn't be None.
    mass_id: int or None, optional (default=None)
      Column index of mass id.
    mass_min: float or None, optional (default=None)
      Mass lower limit.
    mass_max: float or None, optional (default=None)
      Mass higher limit.
    reset_mass: bool, optional (default=None)
      Whether to reset mass with given array's value distribution.
      If set True, reset_mass_array/reset_mass_id shouldn't be None.
    reset_mass_array: numpy array or none, optional (default=None):
      Array used to reset input_array's mass distribution.
    reset_mass_id: int or None, optional (default=None)
      Column index of mass id to reset input_array.
    norm: bool, optional (default=False)
      Whether normalize array's weight to sumofweight.
    sumofweight: float or None, optional (default=None)
      Total normalized weight.
    shuffle: bool, optional (default=None)
      Whether to randomlize the output array.
    shuffle_seed: int or None, optional (default=None)
      Seed for randomization process.
      Set to None to use current time as seed.
      Set to a specific value to get an unchanged shuffle result.

    Returns:
      new: numpy array
        Modified numpy array.

  """
  # Modify
  new = input_array.copy() # copy data to avoid original data operation
  # select channel
  if select_channel == True:
    if not has_none([channel_id, weight_id]):
      for ele in new:
        if ele[channel_id] != 1.0:
          ele[weight_id] = 0
    else:
      print("missing parameters, skipping channel selection...")
  # select mass range
  if select_mass == True:
    if not has_none([mass_id, mass_min, mass_max]):
      for ele in new:
        if ele[mass_id] < mass_min or ele[mass_id] > mass_max:
          ele[weight_id] = 0
    else:
      print("missing parameters, skipping mass selection...")
  # clean array
  new = clean_array(new, -1, remove_negative=remove_negative_weight, 
                    verbose=False)
  # reset mass
  if reset_mass == True:
    if not has_none([reset_mass_array, reset_mass_id]):
      new = prep_mass_fast(new, reset_mass_array, mass_id=reset_mass_id)
    else:
      print("missing parameters, skipping mass reset...")
  # normalize weight
  if norm == True:
    if not has_none([weight_id]):
      new[:, weight_id] = norweight(new[:, weight_id], norm=sumofweight)
    else:
      print("missing parameters, skipping normalization...")
  # shuffle array
  if shuffle == True:
    # use time as random seed if not specified
    """
    if has_none([shuffle_seed]):
      shuffle_seed = int(time.time())
    new, x2, y1, y2 = train_test_split(new, np.zeros(len(new)), test_size=0.01, ########################
                                       random_state=shuffle_seed, shuffle=True)
    """
    new, _, _, _ = shuffle_and_split(
      new, np.zeros(len(new)), split_ratio=0.,
      shuffle_seed=shuffle_seed
      )

  # clean array
  new = clean_array(new, -1, remove_negative=remove_negative_weight, 
                    verbose=False)
  # return result
  return new


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


def norweight(weight_array, norm=1000):
  """Normalize given weight array to certain value

  Args:
    weight_array: numpy array
      Array to be normalized.
    norm: float (default=1000)
      Value to be normalized to.

  Returns:
    new: numpyt array
      normalized array.
  
  Example:
    arr has weight value saved in column -1.
    To normalize it's total weight to 1:
      arr[:, -1] = norweight(arr[:, -1], norm=1)

  """
  new = weight_array.copy() # copy data to avoid original data operation
  total_weight = sum(new)
  frac = norm/total_weight
  new = frac * new
  return new


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
    xs_emu = modify_array(xs_emu, weight_id = -1, 
                          select_channel = True, channel_id = -4,
                          norm = True, shuffle = True, shuffle_seed = 485)

    # make plots
    plt.hist(xs_emu[:, para_index], bins = bins, weights = xs_emu[:,-1], 
             histtype='step', label='signal {}00GeV'.format(mass), range = range, density = density)
  plt.legend(prop={'size': 10})
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()


def prep_mass_fast(xbtrain, xstrain, mass_id=0, shuffle_seed=None):
  """Resets background mass distribution according to signal distribution

  Args:
    xbtrain: numpy array
      Background array
    xstrain: numpy array
      Siganl array
    mass_id: int (default=0)
      Column index of mass.
    shuffle_seed: int or None, optional (default=None)
      Seed for randomization process.
      Set to None to use current time as seed.
      Set to a specific value to get an unchanged shuffle result.

  Returns:
    new: numpy array
      new background array with mass distribution reset

  """
  new =  reset_col(xbtrain, xstrain, col=mass_id, weight_id=-1, shuffle_seed=None)
  return new


def reset_col(reset_array, ref_array, col=0, weight_id=-1, shuffle_seed=None):
  """Resets one column in an array based on the distribution of refference."""
  if has_none([shuffle_seed]):
    shuffle_seed = int(time.time())
  np.random.seed(shuffle_seed)
  new = reset_array.copy()
  total_events = len(new)
  sump = sum(ref_array[:, weight_id])
  reset_list = np.random.choice(ref_array[:, col], size=total_events, p=1/sump*ref_array[:, -1])
  for count, entry in enumerate(new):
    entry[col] = reset_list[count]
  return new


def shuffle_and_split(x, y, split_ratio=0., shuffle_seed=None):
  """Self defined function to replace train_test_split in sklearn to allow
  more flexibility.
  """
  # Check consistance of length of x, y
  if len(x) != len(y):
    raise ValueError("Length of x and y is not same.")
  array_len = len(y)
  np.random.seed(shuffle_seed)
  # get index for the first part of the splited array
  first_part_index = np.random.choice(
    range(array_len),
    int(array_len * 1. * split_ratio),
    replace=False
  )
  # get index for last part of the splited array
  last_part_index = np.setdiff1d(np.array(range(array_len)), first_part_index)
  first_part_x = x[first_part_index]
  first_part_y = y[first_part_index]
  last_part_x = x[last_part_index]
  last_part_y = y[last_part_index]
  return first_part_x, last_part_x, first_part_y, last_part_y


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

  xs_train, xs_test, ys_train, ys_test = shuffle_and_split(
    xs, ys, split_ratio=test_rate,
    shuffle_seed=shuffle_seed
    )
  xb_train, xb_test, yb_train, yb_test = shuffle_and_split(
    xb, yb, split_ratio=test_rate,
    shuffle_seed=shuffle_seed
    )

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
