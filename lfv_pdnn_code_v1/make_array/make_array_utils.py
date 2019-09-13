import array
import datetime
import math

import numpy as np
import ROOT

import lfv_pdnn_code_v1.common.common_utils

# root_vector_double = vector('double')()
# _length_const = 24

def build_array_withcut(rootfile_path, clean_array=True):
  """builds array with cuts according to Marc's example code.

  Args:
    rootfile_path: str, path to the root file to be processed.
    clean_array: bool, flag to decide whether to remove zero weight events.

  Returns:
    numpy array built from ntuple with selection applied.
  """
  ARRAY_ELEMENT_LENTH = 24  # number of variables to stored for each event

  # Prepare
  print "\nbuilding array for: ", rootfile_path
  try:
    f = ROOT.TFile.Open(rootfile_path)
    tree = f.nominal  # nominal is the name of ntuple
  except:
    print "[ERROR] Can not get tree."
    return None
  events_num = tree.GetEntries()
  data = np.zeros((events_num, ARRAY_ELEMENT_LENTH))

  # Loop
  for n, event in enumerate(tree):
    # get parameters needed
    event_number = event.eventNumber

    if n < 10:
      print "event number:", event_number