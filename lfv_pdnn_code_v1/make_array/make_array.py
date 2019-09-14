import array
import datetime
import math

import numpy as np
import ROOT

import lfv_pdnn_code_v1.common.common_utils
import lfv_pdnn_code_v1.make_array.make_array_utils
from lfv_pdnn_code_v1.make_array.make_array_utils import get_lumi

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
    mu = event.mu
    mu_actual = event.mu_actual
    background_flags = event.backgroundFlags
    has_bad_muon = event.hasBadMuon
    el_pt = event.el_pt
    el_eta = event.el_eta
    el_cl_eta = event.el_cl_eta
    el_phi = event.el_phi
    el_e = event.el_e
    el_charge = event.el_charge
    el_topoetcone20 = event.el_topoetcone20
    el_ptvarcone20 = event.el_ptvarcone20
    el_cf = event.el_CF
    el_d0sig = event.el_d0sig
    el_delta_z0_sintheta = event.el_delta_z0_sintheta
    el_true_type = event.el_true_type
    el_true_origin = event.el_true_origin
    mu_pt = event.mu_pt
    mu_eta = event.mu_eta
    mu_phi = event.mu_phi
    mu_e = event.mu_e
    mu_charge = event.mu_charge
    mu_d0sig = event.mu_d0sig
    mu_delta_z0_sintheta = event.mu_delta_z0_sintheta
    mu_true_type = event.mu_true_type
    mu_true_origin = event.mu_true_origin
    tau_pt = event.tau_pt
    tau_eta = event.tau_eta
    tau_phi = event.tau_phi
    tau_charge = event.tau_charge
    jet_pt = event.jet_pt
    jet_eta = event.jet_eta
    jet_phi = event.jet_phi
    jet_e = event.jet_e
    jet_mv2c10 = event.jet_mv2c10
    jet_jvt = event.jet_jvt
    jet_passfjvt = event.jet_passfjvt
    met_met = event.met_met
    met_phi = event.met_phi
    emu_selection = event.emuSelection
    etau_selection = event.etauSelection
    mutau_selection = event.mutauSelection
    weight_kfactor = event.weight_KFactor
    mu_is_high_pt = event.mu_isHighPt
    mu_isolation_fixed_cut_tight = event.mu_isolation_FixedCutTight
    mu_isolation_fixed_cut_loose = event.mu_isolation_FixedCutLoose
    el_isolation_fixed_cut_loose = event.el_isolation_FixedCutLoose
    el_isolation_fixed_cut_tight = event.el_isolation_FixedCutTight
    el_is_tight = event.el_isElTight
    el_is_medium = event.el_isElMedium
    el_is_loose = event.el_isElLoose
    tau_is_medium = event.tau_isMedium
    tau_is_Loose = event.tau_isLoose
    tau_bdt = event.tau_BDT
    tau_num_track = event.tau_nTracks

    # Basic info
    luminosity = get_lumi(run_number)
    # calculate normalisation factor to be used for weight calculation
    #norm_factor = Normalisation(mcChannelNumber,runNumber,weight_mc) ####
    norm_factor = 1
    weight = luminosity * weight_kfactor * weight_pileup \
             * weight_mc * norm_factor

    # Select 

    if n < 10:
      print "event number:", event_number,
      print "lumi:", luminosity,
      print "weight:", weight