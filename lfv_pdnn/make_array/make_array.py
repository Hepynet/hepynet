import io
import logging
import os

import numpy as np

import ROOT
import uproot


def dump_flat_ntuple_individual(
    root_path: str,
    ntuple_name: str,
    variable_list: list,
    save_dir: str,
    save_pre_fix: str,
    use_lower_var_name: bool = False,
) -> None:
    """Reads numpy array from ROOT ntuple and convert to numpy array.
   
  Note:
    Each branch will be saved as an individual file.

  """
    try:
        events = uproot.open(root_path)[ntuple_name]
    except:
        raise IOError("Can not get ntuple")
    print("Read arrays from:", root_path)
    for var in variable_list:
        if use_lower_var_name:
            file_name = save_pre_fix + "_" + var.lower()
        else:
            file_name = save_pre_fix + "_" + var
        print("Generating:", file_name)
        temp_arr = events.array(var)
        save_array(temp_arr, save_dir, file_name)


def save_array(array, directory_path, file_name, dump_empty=False):
    """Saves numpy data as .npy file

  Args:
    array: numpy array, array to be saved
    directory_path: str, directory path to save the file
    file_name: str, file name used by .npy file

  """
    save_path = directory_path + "/" + file_name + ".npy"
    if array.size == 0:
        if dump_empty:
            logging.warning("Empty array detected! Will save empty array as specified.")
        else:
            logging.warning(
                "Empty array detected! Skipping saving current array to: " + save_path
            )
            return
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with io.open(save_path, "wb") as f:
        np.save(f, array)
