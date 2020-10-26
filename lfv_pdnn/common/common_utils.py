import glob
import json
import math
import os

import numpy as np
from lfv_pdnn.common.logging_cfg import *


def create_folders(foldernames, parent_path="./"):
    """Checks existence of given folder names, creats if not exsits.
  
  Args:
    foldernames: list of str, folder names to be checked/created.
    parent_path: str, parent path where to create folders.
  """
    for foldername in foldernames:
        today_dir = os.path.join(parent_path, foldername)
        if not os.path.isdir(today_dir):
            os.makedirs(today_dir)


def dict_key_strtoint(json_data):
    """Cast string keys to int keys"""
    correctedDict = {}
    for key, value in json_data.items():
        if isinstance(value, list):
            value = [
                dict_key_strtoint(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            value = dict_key_strtoint(value)
        try:
            key = int(key)
        except:
            pass
        correctedDict[key] = value
    return correctedDict


def display_dict(input_dict):
    """Print dict in a readable way."""
    for key in list(input_dict.keys()):
        print("*", key, ":", input_dict[key])


def get_file_list(directory, search_pattern, out_name_identifier="None"):
    """Gets a full list of file under given directory with given name pattern

  To use:
  >>> get_file_list("path/to/directory", "*.root", "signal_emu_500_GeV{}")

  Args:
    directory: str, path to search files
    search_pattern: str, pattern of files to search
    out_name_identifier: patter to rename file_name_list with increased number

  Returns:
    A list of file absolute path & file name 
  """
    # Get absolute path
    absolute_file_list = glob.glob(directory + "/" + search_pattern, recursive=True)
    absolute_file_list.sort()
    if len(absolute_file_list) == 0:
        logging.warning("Empty file list, please check input.")
    # Get file name match the pattern
    file_name_list = [os.path.basename(path) for path in absolute_file_list]
    # Rename file_name_list if out_name_identifier is specified
    if out_name_identifier is not None:
        if len(file_name_list) == 1:
            file_name_list[0] = out_name_identifier
        else:  # add number for multiple files that match the pattern
            for id, ele in enumerate(file_name_list):
                file_name_list[id] = (out_name_identifier + ".{}").format(id)
    # check duplicated name in file_name_list
    for name in file_name_list:
        num_same_name = 0
        for name_check in file_name_list:
            if name == name_check:
                num_same_name += 1
        if num_same_name > 1:
            logging.warning("Same file name detected.")
    return absolute_file_list, file_name_list


def get_newest_file_version(path_pattern, n_digit=2, ver_num=None, use_existing=False):
    """Check existed file and return last available file path with version.

  Version range 00 -> 99 (or 999)
  If reach limit, last available version will be used. 99 (or 999)

  """
    # return file path if ver_num is given
    if ver_num is not None:
        return {
            "ver_num": ver_num,
            "path": path_pattern.format(str(ver_num).zfill(n_digit)),
        }
    # otherwise try to find ver_num
    max_version = int(math.pow(10, n_digit) - 1)
    ver_num = 0
    path = path_pattern.format(str(ver_num).zfill(n_digit))
    while os.path.exists(path):
        ver_num += 1
        path = path_pattern.format(str(ver_num).zfill(n_digit))
    if use_existing:
        if ver_num < 1:
            logging.warning("Non existing folder found! Using 0 for the counting.")
            path = path_pattern.format(str(0).zfill(n_digit))
        else:
            path = path_pattern.format(str(ver_num - 1).zfill(n_digit))
    if ver_num > max_version:
        logging.warning(
            "Too much model version detected at same date. \
      Will only keep maximum {} different versions.".format(
                max_version
            )
        )
        logging.warning("Version {} will be overwrite!".format(max_version))
        ver_num = max_version
    return {
        "ver_num": ver_num,
        "path": path,
    }


def get_significant_digits(number, n_digits):
    if round(number) == number:
        m = len(str(number)) - 1 - n_digits
        if number / (10 ** m) == 0.0:
            return number
        else:
            return float(int(number) / (10 ** m) * (10 ** m))
    if len(str(number)) > n_digits + 1:
        return round(number, n_digits - len(str(int(number))))
    else:
        return number


def has_none(list):
    """Checks whether list's element has "None" value element.
  
  Args:
    list: list, input list to be checked.

  Returns:
    True, if there IS "None" value element.
    False, if there ISN'T "None" value element.
  """
    for ele in list:
        if ele is None:
            return True
    return False


def read_dict_from_json(json_input):
    """Reads dict type data from json input

  Args:
    json_input: json, used to read dict from

  Returns:
    dict type data of json input
  """
    pass


def read_dict_from_txt(file_path, key_type="str", value_type="str"):
    """Reads dict type data from text file
  
  Args:
    file_path: str, path to the input text file
    key_type: str, specify type of key of dict
      use 'str' for string type
      use 'float' for float value
    value_type: str, specify type of value of dict
      available type same as key_type

  Returns:
    dict type data of text file input
  """
    dict_output = {}

    with open(file_path, "r") as lines:
        for line in lines:
            key_error = False
            value_error = False
            content1, content2 = line.strip().split(",", 1)
            # get key
            if key_type == "str":
                key = content1.strip()
            elif key_type == "float":
                try:
                    key = eval(content1)
                except ZeroDivisionError:
                    key_error = True
                    logging.warning("Float division by zero.")
                    continue  # skip invalid key
                except:
                    key_error = True
                    logging.warning("Unknown evaluation error.")
                    continue  # skip invalid key
            else:
                logging.warning("Unrecognized key type.")
            # get value
            if value_type == "str":
                value = content2.strip()
            elif value_type == "float":
                try:
                    value = eval(content2)
                except ZeroDivisionError:
                    value_error = True
                    value = 0  # set default value
                    logging.warning("Float division by zero.")
                except:
                    value_error = True
                    value = 0  # set default value
                    logging.warning("Unknown evaluation error.")
            else:
                logging.warning("Unrecognized value type.")
            # save dict item
            if key in dict_output:
                logging.warning("Key already exists, overwriting value...")
                if value_error == True:
                    continue  # skip invalid value if value of key already exists
            dict_output[key] = value
    return dict_output
