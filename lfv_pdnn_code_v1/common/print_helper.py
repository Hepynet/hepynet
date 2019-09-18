import warnings

def print_error(*inputs): # should use raise instead
  """Prints uniform error message output,
     only for debugging and non-important error.

  Args:
    *inputs: Variable number of str, error messages to be print.
  """
  assert inputs is not None
  print "Error:",
  for input in inputs:
    print input,  # use comma to aviod changing line
  print ''


def print_warning(*inputs): # should use warings instead
  """Prints uniform warning message output.

  Args:
    *inputs: Variable number of str, warning messages to be print.
  """
  assert inputs is not None
  content = ""
  for input in inputs:
    content = content + input
  warnings.warn(content)


def show_array_example(array, max_row = 5):
  """Shows some rows of given array.

  Args:
    array: numpy array to be showed.
    max_row: int, number of rows to be shown.
  """
  print "show some array examples:"
  for i, ele in enumerate(array):
    if i < max_row:
      print ele
