"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to on type of
  signal or background. For example: "top", "rpv_1000"
  2. Special key names that can be used for training:
    * all: all sig/bkg concatenated directly
    * all_norm: (sig only) with each mass point array's weight normed and then
    concatenated.

"""

import numpy as np

from lfv_pdnn.common import array_utils

def get_bkg(npy_path, campaign, channel, bkg_list, selected_features):
  print("Loading raw background array.")
  # Load individual bkg
  bkg_dict = get_npy_individuals(
    npy_path, campaign, channel, bkg_list, selected_features, "bkg")
  # Add all bkg together
  bkg_all_array = np.concatenate(list(bkg_dict.values()))
  bkg_dict['all'] = bkg_all_array
  return bkg_dict

def get_data(npy_path, campaign, channel, data_list, selected_features):
  print("Loading raw data array.")
  # Load data, only one tag "all"
  data_dict = get_npy_individuals(
    npy_path, campaign, channel, data_list, selected_features, "data")
  return data_dict

def get_sig(npy_path, campaign, channel, sig_list, selected_features):
  print("Loading raw signal array.")
  # Load individual sig
  sig_dict = get_npy_individuals(
    npy_path, campaign, channel, sig_list, selected_features, "sig")
  # Add all sig together
  sig_all_array = np.concatenate(list(sig_dict.values()))
  sig_dict['all'] = sig_all_array
  # Add all sig together with each mass point normalized
  sig_all_array_norm = None
  for sig in sig_list:
    temp_array = sig_dict[sig]
    temp_array = array_utils.modify_array(temp_array, norm=True)
    if sig_all_array_norm is None:
      sig_all_array_norm = temp_array
    else:
      sig_all_array_norm = np.concatenate((sig_all_array_norm, temp_array))
  sig_dict['all_norm'] = sig_all_array_norm
  return sig_dict

def get_npy_individuals(npy_path, campaign, channel, npy_list,
  selected_features, npy_prefix):
  """Gets individual npy arrays with given info.
  
  Return:
    A dict of individual npy arrays.
    Example: signal dict of rpv_500 rpv_1000 rpv_2000

  """
  return_dict = {}
  # Load individual npy array
  for npy in npy_list:
    if campaign in ["run2", "all"]:
      directory = npy_path
      npy_array = None
      for feature in (selected_features + [channel, "weight"]):
        temp_array1 = np.load(
          directory+"/mc16a" + "/" + npy_prefix + "_"+npy+"_"+feature+".npy")
        temp_array1 = np.reshape(temp_array1, (-1, 1))
        temp_array2 = np.load(
          directory+"/mc16d" + "/" + npy_prefix + "_"+npy+"_"+feature+".npy")
        temp_array2 = np.reshape(temp_array2, (-1, 1))
        temp_array3 = np.load(
          directory+"/mc16e" + "/" + npy_prefix + "_"+npy+"_"+feature+".npy")
        temp_array3 = np.reshape(temp_array3, (-1, 1))
        temp_array = np.concatenate(
          (temp_array1, temp_array2, temp_array3), axis=0)
        #### hot fix
        temp_array1 = np.load(
          directory+"/mc16a" + "/" + npy_prefix + "_"+npy+"_normsf.npy")
        temp_array1 = np.reshape(temp_array1, (-1, 1))
        temp_array2 = np.load(
          directory+"/mc16d" + "/" + npy_prefix + "_"+npy+"_normsf.npy")
        temp_array2 = np.reshape(temp_array2, (-1, 1))
        temp_array3 = np.load(
          directory+"/mc16e" + "/" + npy_prefix + "_"+npy+"_normsf.npy")
        temp_array3 = np.reshape(temp_array3, (-1, 1))
        temp_array_normsf = np.concatenate(
          (temp_array1, temp_array2, temp_array3), axis=0)
        if feature == "weight":
          temp_array = temp_array * temp_array_normsf
        ####
        if npy_array is None:
          npy_array = temp_array
        else:
          npy_array = np.concatenate((npy_array, temp_array), axis=1)
      print(npy, "shape:", npy_array.shape)
      return_dict[npy] = npy_array
    else:
      directory = npy_path + "/" + campaign
      npy_array = None
      for feature in (selected_features + [channel, "weight"]):
        temp_array = np.load(
          directory+"/" + npy_prefix + "_"+npy+"_"+feature+".npy")
        temp_array = np.reshape(temp_array, (-1, 1))
        #### hot fix
        temp_array_normsf = np.load(
          directory+"/" + npy_prefix + "_"+npy+"_normsf.npy")
        temp_array_normsf = np.reshape(temp_array_normsf, (-1, 1))
        if feature == "weight":
          temp_array = temp_array * temp_array_normsf
        ####
        if npy_array is None:
          npy_array = temp_array
        else:
          npy_array = np.concatenate((npy_array, temp_array), axis=1)
      print(npy, "shape:", npy_array.shape)
      return_dict[npy] = npy_array
  return return_dict
