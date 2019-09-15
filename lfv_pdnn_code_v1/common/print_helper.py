def print_error(*inputs):
  """prints uniform error message output.

  Args:
    *inputs: variable number of str, error messages to be print
  
  """
  assert inputs is not None
  print "[Error]",
  for input in inputs:
    print input,  # use comma to aviod changing line
  print ''


def print_warning(*inputs):
  """prints uniform warning message output.

  Args:
    *inputs: variable number of str, warning messages to be print
  
  """
  assert inputs is not None
  print "[Warning]",
  for input in inputs:
    print input,  # use comma to aviod changing line
  print ''


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
