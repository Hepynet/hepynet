import array
import datetime
import math

import numpy as np
import ROOT

from lfv_pdnn_code_v1.common import common_utils, print_helper
from lfv_pdnn_code_v1.make_array import make_array_utils, particle
from lfv_pdnn_code_v1.make_array.make_array_utils import get_lumi

# normalization file path
norm_file_284500 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_284500.txt"
norm_file_300000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_300000.txt"
norm_file_310000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_310000.txt"

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
  # load tree
  print "building array for: ", rootfile_path
  try:
    f = ROOT.TFile.Open(rootfile_path)
    tree = f.nominal  # nominal is the name of ntuple
  except:
    print_helper.print_error("Can not get tree.",
                             "in make_array.build_array_withcut")
    return None
  events_num = tree.GetEntries()
  data = np.zeros((events_num, ARRAY_ELEMENT_LENTH))
  # load nornalization dict
  norm_284500 = common_utils.read_dict_from_txt(norm_file_284500, 
                                                key_type='float', 
                                                value_type='float')
  norm_300000 = common_utils.read_dict_from_txt(norm_file_300000, 
                                                key_type='float', 
                                                value_type='float')
  norm_310000 = common_utils.read_dict_from_txt(norm_file_310000, 
                                                key_type='float', 
                                                value_type='float')
  def get_norm(mc_channel_number, run_number):
    if run_number == 284500:
      return norm_284500[float(mc_channel_number)]
    if run_number == 300000:
      return norm_300000[float(mc_channel_number)]
    if run_number == 310000:
      return norm_310000[float(mc_channel_number)]

  # Loop
  for n, event in enumerate(tree):
    # Observables
    # get particles
    electrons = particle.electron_candidates(event)
    muons = particle.muon_candidates(event)
    taus = particle.tau_candidates(event)
    jets = particle.jet_candidates(event)
    # get extra parameters needed
    weight_mc = event.weight_mc
    weight_pileup = event.weight_pileup
    weight_lepton_sf = event.weight_leptonSF
    weight_tau_sf = event.weight_tauSF
    weight_global_lepton_trigger_sf = event.weight_globalLeptonTriggerSF
    weight_jvt = event.weight_jvt
    event_number = event.eventNumber
    run_number = event.runNumber
    random_run_number = event.randomRunNumber
    mc_channel_number = event.mcChannelNumber
    background_flags = event.backgroundFlags
    emu_selection = event.emuSelection
    etau_selection = event.etauSelection
    mutau_selection = event.mutauSelection
    weight_kfactor = event.weight_KFactor
    # Weight
    luminosity = get_lumi(run_number)
    # calculate normalisation factor to be used for weight calculation
    try:
      norm_factor = get_norm(mc_channel_number, run_number)
    except KeyError:
      norm_factor = 0
      #print_helper.print_error("can't get norm factor for current",
      #                         "mc_channel_number:", mc_channel_number)
    weight = luminosity * weight_kfactor * weight_pileup \
             * weight_mc * norm_factor

    # Select electron
    electrons.select()
    muons.select()
    taus.select()
    jets.select()
    #print "selected pt: ", taus.selected_tau.Pt(), taus.selected_neutrino.Pt() ########
    #print "b_tag_jet:", jets.b_tag_jet

    if n < 10:
      print "event number:", event_number,
      print "lumi:", luminosity,
      print "weight:", weight
      print mc_channel_number, run_number