from ROOT import (TCanvas, TPad, TFile, TPaveLabel, 
                  TPaveText, gROOT, TH1F, TH1D, TLegend, 
                  gStyle, TH2F, TChain, TGraphErrors, TText, gPad, gROOT, TTree, vector, TLorentzVector)

from array import array
import datetime
import math
import numpy as np

from common_utils import *

root_vector_double = vector('double')()
array_length_const = 24

def init_dirs():
  TODAY = datetime.datetime.now().strftime('%Y-%m-%d')
  import datetime, os
  for _dir in ["plots", "models"]:
    today_dir = os.path.join(_dir, TODAY)
    if not os.path.isdir(today_dir):
      os.makedirs(today_dir)

# make dedicated cut for lfv analysis
"""
def cut_array(data):
  print "Make cuts on array..."
  new = []
  for d in data:
"""

# function to calculate delta phi
def CalDeltaPhi(phi1, phi2):
  dphi = phi1 - phi2
  if abs(dphi) > math.pi:
    if dphi > 0:
      dphi = 2 * math.pi - dphi
    else:
      dphi = 2 * math.pi + dphi
  return dphi

# function to calculate delta R
def DeltaR(phi1, phi2, eta1, eta2):
  Dphi = CalDeltaPhi(phi1, phi2)
  Deta = eta1 - eta2
  DR = math.sqrt(Dphi * Dphi + Deta * Deta)
  return DR

def build_array(fname, should_clean_arrays=True, **kwargs):
  # used to build array with new sample
  print "\nbuilding array for: ", fname
  f = TFile.Open(fname)
  tree = f.nominal
  n_events = tree.GetEntries()
  data = np.zeros((n_events, array_length_const))
  example_numbers = 0 # used for debug
  #print "array example: "
  for n, event in enumerate(tree):
    # get parameters needed
    weight_mc = event.weight_mc
    weight_pileup = event.weight_pileup
    weight_leptonSF = event.weight_leptonSF
     = event.
     = event.
     = event.
     = event.
     = event.
     = event.
    # = event.
    # skip the events with 0 weight
    if should_clean_arrays == True:
      if weight == 0.:
        continue

    
    
      """
      if example_numbers < 10:
        if emuChannel:
          print "emu channel"
        elif etauChannel:
          print "etau channel"
        elif mutauChannel:
          print "mutao channel"
        example_numbers += 1
        for i in range(0, 10):
          print "{:.2}".format(data[n][i]), "\t",
        print ""
        for i in range(10, array_length_const):
          print "{:.2}".format(data[n][i]), "\t",
        print ""
      """
    
  return clean_array(data) if should_clean_arrays else data

def unison_shuffled_copies(*arr):
  assert all(len(a) for a in arr)
  p = np.random.permutation(len(arr[0]))
  return (a[p] for a in arr)
