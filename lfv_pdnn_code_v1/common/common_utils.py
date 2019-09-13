import os

import numpy as np

def clean_array(array, weight_id, remove_negative=False, verbose=False):
  """removes elements with 0 weight.

  Args:
    array: numpy array, input array to be processed, must be numpy array
    weight_id: int, indicate which column is weight value.
    remove_negative: bool, optional, remove zero weight row if set True
    verbose: bool, optional, show more detailed message if set True.

  Returns:
    cleaned numpy array.

  """
  # Start
  if verbose:
    print "cleaning array..."
  
  # Clean
  # create new array for output to avoid direct operation on input array
  new = []
  if remove_negative == False:
    for d in array:
      if d[weight_id] == 0.:  # only remove zero weight row
        continue
      new.append(d)
  elif remove_negative == True:
    for d in array:
      if d[weight_id] <= 0.:  # remove zero or negative weight row
        continue
      new.append(d)

  # Output
  out = np.array(new)
  if verbose:
    print "shape before", array.shape, 'shape after', out.shape
  return out


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


def show_array_example(array, max_row = 5):
  """shows some rows of given array.

  Args:
    array: numpy array to be showed.
    max_row: int, number of rows to be shown.

  """
  print "show some array examples:"
  for i, ele in enumerate(array):
    if i < max_row:
      print ele
