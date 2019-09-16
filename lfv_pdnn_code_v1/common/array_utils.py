import numpy as np

def clean_array(array, weight_id, remove_negative=False, verbose=False):
  """removes elements with 0 weight.

  Args:
    array: numpy array, input array to be processed, must be numpy array
    weight_id: int, indicate which column is weight value.
    remove_negative: bool, optional, remove zero weight row if set True
    verbose: bool, optional, show more detailed message if set True.

  Returns:
    cleaned new numpy array.
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
