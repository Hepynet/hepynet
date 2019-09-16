import os

import json
import numpy as np

from lfv_pdnn_code_v1.common import print_helper

def create_folders(foldernames, parent_path="./"):
  """checks existence of given folder names, creats if not exsits.
  
  Args:
    foldernames: list of str, folder names to be checked/created.
    parent_path: str, parent path where to create folders.
  """
  for foldername in foldernames:
    today_dir = os.path.join(parent_path, foldername)
    if not os.path.isdir(today_dir):
      os.makedirs(today_dir)


def has_none(list):
  """checks whether list's element has "None" value element.
  
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
  """reads dict type data from json input

  Args:
    json_input: json, used to read dict from

  Returns:
    dict type data of json input
  """
  pass


def read_dict_from_txt(file_path, key_type='str', value_type='str'):
  """reads dict type data from text file
  
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