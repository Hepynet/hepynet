import glob
import os

import json
import numpy as np

from lfv_pdnn_code_v1.common import print_helper

def clean_array(data, weight_id, verbose = False, remove_negative = False):
  """Removes elements with 0 weight

  Args:
    data: numpy array, array to be cleaned
    weight_id: int, index of weight in each row
    verbose: bool, set True to give detailed infomation
    remove_negative: bool, set True to remove negative weight event
  """
  if verbose:
    print("cleaning array...")
  new = []
  if remove_negative == False:
    for d in data:
      if d[weight_id] == 0.:
        continue
      new.append(d)
    out = np.array(new)
  elif remove_negative == True:
    for d in data:
      if d[weight_id] <= 0.:
        continue
      new.append(d)
    out = np.array(new)
  if verbose:
    print("shape before", data.shape, 'shape after', out.shape)
  return out


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
      value = [dict_key_strtoint(item) if isinstance(item, dict) else item for item in value]
    elif isinstance(value, dict):
      value = dict_key_strtoint(value)
    try:
      key = int(key)
    except Exception as ex:
      pass
    correctedDict[key] = value
  return correctedDict


def get_file_list(directory, search_pattern, out_name_pattern = "None"):
  """Gets a full list of file under given directory with given name pattern

  To use:
  >>> get_file_list("path/to/directory", "*.root")

  Args:
    directory: str, path to search files
    search_pattern: str, pattern of files to search

  Returns:
    A list of file absolute path & file name 
  """
  # Get absolute path
  absolute_file_list = glob.glob(directory + "/" + search_pattern)
  if len(absolute_file_list) == 0:
    print_helper.print_warning("empty file list, please check input",
                               "(in get_file_list)")
  # Get file name match the pattern
  file_name_list = [os.path.basename(path) for path in absolute_file_list]
  # check duplicated name in file_name_list
  for name in file_name_list:
    num_same_name = 0
    for name_check in file_name_list:
      if name == name_check:
        num_same_name += 1
    if num_same_name > 1:    
      print_helper.print_warning("same file name detected",
                                 "(in get_file_list)")
  return absolute_file_list, file_name_list


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


def read_dict_from_txt(file_path, key_type='str', value_type='str'):
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
      content1, content2 = line.strip().split(',', 1)
      # get key
      if key_type == 'str':
        key = content1.strip()
      elif key_type == 'float':
        try:
          key = eval(content1)
        except ZeroDivisionError:
          key_error = True
          print_helper.print_warning("float division by zero",
                                     "in common_utils.read_dict_from_txt")
          continue  # skip invalid key
        except:
          key_error = True
          print_helper.print_error("unknown evaluation error",
                                   "in common_utils.read_dict_from_txt")
          continue  # skip invalid key
      else:
        print_helper.print_error("unrecognized key type",
                                 "in common_utils.read_dict_from_txt")
      # get value
      if value_type == 'str':
        value = content2.strip()
      elif value_type == 'float':
        try:
          value = eval(content2)
        except ZeroDivisionError:
          value_error = True
          value = 0  # set default value
          print_helper.print_warning("float division by zero",
                                     "in common_utils.read_dict_from_txt")
        except:
          value_error = True
          value = 0  # set default value
          print_helper.print_error("unknown evaluation error",
                                   "in common_utils.read_dict_from_txt")
      else:
        print_helper.print_error("unrecognized value type",
                                 "in common_utils.read_dict_from_txt")
      # save dict item
      if key in dict_output:
        print_helper.print_warning("key already exists, overwriting value...")
        if value_error == True:
          continue  # skip invalid value if value of key already exists
      dict_output[key] = value
  return dict_output