import logging

import numpy as np
import ROOT
import uproot

from lfv_pdnn.common import common_utils, observable_cal, print_helper
from lfv_pdnn.common import array_utils
from lfv_pdnn.make_array import make_array_utils, particle
from lfv_pdnn.make_array.make_array_utils import *


# normalization file path
norm_file_284500 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn/data/make_array_info/norm_file_284500.txt"
norm_file_300000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn/data/make_array_info/norm_file_300000.txt"
norm_file_310000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn/data/make_array_info/norm_file_310000.txt"

# root_vector_double = vector('double')()
# _length_const = 24

def build_array_withcut(rootfile_path, should_clean_array=True):
  """builds array with cuts according to Marc's example code.

  Args:
    rootfile_path: str, path to the root file to be processed.
    should_clean_array: bool, flag to decide whether to remove zero weight 
    events.

  Returns:
    numpy array built from ntuple with selection applied.

  Note:
    For multi-channel events, only one channel will be chosen.
    Priority: emu > etau > mutau
  """
  ARRAY_ELEMENT_LENTH = 24  # number of variables to stored for each event

  # Prepare
  # load tree
  print("building array for: ", rootfile_path)
  try:
    f = ROOT.TFile.Open(rootfile_path)
    tree = f.nominal  # nominal is the name of ntuple
  except:
    raise OSError("Can not get tree. in make_array.build_array_withcut")
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
  num_muti_channel_event = 0
  for n, event in enumerate(tree):
    # Observables
    # get particles
    electrons = particle.ElectronCandidates(event)
    muons = particle.MuonCandidates(event)
    taus = particle.TauCandidates(event)
    jets = particle.JetCandidates(event)
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
      raise OSError("can't get norm factor for current mc_channel_number:{}".format(mc_channel_number))
    weight = luminosity * weight_kfactor * weight_pileup \
             * weight_mc * norm_factor

    # Select particles
    electrons.select()
    muons.select()
    taus.select(electrons, muons)
    jets.select()

    # Set propagator
    possible_channels = get_early_channel(electrons, muons, taus)
    channel_is_true = False
    selected_channel = "none"
    num_valid_channel = 0
    propagator_temp = ROOT.TLorentzVector(0, 0, 0, 0)
    propagator = ROOT.TLorentzVector(0, 0, 0, 0)
    for channel in possible_channels:
      channel_is_true, propagator_temp = get_final_channel(electrons, muons, 
                                                           taus, channel)
      if channel_is_true:
        selected_channel = channel
        num_valid_channel += 1
        propagator = propagator_temp
    if num_valid_channel == 0:
      continue
    if num_valid_channel == 1:
      _, propagator = get_final_channel(electrons, muons, taus, selected_channel)
    if num_valid_channel > 1:
      num_muti_channel_event += 1
      # chose one channel with priority emu > etau > mutau
      if "emu" in possible_channels:
        selected_channel = "emu"
        _, propagator = get_final_channel(electrons, muons, taus, "emu")
      elif "etau" in possible_channels:
        selected_channel = "etau"
        _, propagator = get_final_channel(electrons, muons, taus, "etau")
      elif "mutau" in possible_channels:
        selected_channel = "mutau"
        _, propagator = get_final_channel(electrons, muons, taus, "mutau")

    # Post calculation
    selected_electron = electrons.selected_particle
    selected_muon = muons.selected_particle
    selected_tau = taus.selected_particle
    selected_neutrino = taus.selected_neutrino
    is_emu = False
    is_etau = False
    is_mutau = False
    lepton_pair_dphi = 0
    lepton_pair_dR = 0
    
    if selected_channel == 'emu':
      if mc_channel_number != 0:
        weight *= weight_lepton_sf
        phi1 = selected_electron.Phi()
        eta1 = selected_electron.Eta()
        phi2 = selected_muon.Phi()
        eta2 = selected_muon.Eta()
        lepton_pair_dphi = observable_cal.delta_phi(phi1, phi2)
        lepton_pair_dR = observable_cal.delta_r(eta1, phi1, eta2, phi2)
        is_emu = True
    elif selected_channel == 'etau':
      if mc_channel_number != 0:
        weight *= weight_lepton_sf * weight_tau_sf
        phi1 = selected_electron.Phi()
        eta1 = selected_electron.Eta()
        phi2 = selected_tau.Phi()
        eta2 = selected_tau.Eta()
        lepton_pair_dphi = observable_cal.delta_phi(phi1, phi2)
        lepton_pair_dR = observable_cal.delta_r(eta1, phi1, eta2, phi2)
        is_etau = True
        # debug
        #print "etau channel, tau_pt =", taus.selected_particle.Pt()
    elif selected_channel == 'mutau':
      if mc_channel_number != 0:
        weight *= weight_lepton_sf * weight_tau_sf
        phi1 = selected_muon.Phi()
        eta1 = selected_muon.Eta()
        phi2 = selected_tau.Phi()
        eta2 = selected_tau.Eta()
        lepton_pair_dphi = observable_cal.delta_phi(phi1, phi2)
        lepton_pair_dR = observable_cal.delta_r(eta1, phi1, eta2, phi2)
        is_mutau = True
        # debug
        #print "mutau channel, tau_pt =", taus.selected_particle.Pt()
    
    # Save array
    data[n][0] = propagator.Mag()
    data[n][1] = selected_electron.Pt()
    data[n][2] = selected_electron.Eta()
    data[n][3] = selected_electron.Phi()
    data[n][4] = selected_electron.Mag()
    data[n][5] = selected_muon.Pt()
    data[n][6] = selected_muon.Eta()
    data[n][7] = selected_muon.Phi()
    data[n][8] = selected_muon.Mag()
    data[n][9] = selected_tau.Pt()
    data[n][10] = selected_tau.Eta()
    data[n][11] = selected_tau.Phi()
    data[n][12] = selected_tau.Mag()
    data[n][13] = selected_neutrino.E()
    data[n][14] = selected_neutrino.Phi()
    data[n][15] = propagator.Pt()
    data[n][16] = propagator.Eta()
    data[n][17] = propagator.Phi()
    data[n][18] = lepton_pair_dphi
    data[n][19] = lepton_pair_dR
    data[n][20] = is_emu
    data[n][21] = is_etau
    data[n][22] = is_mutau
    data[n][23] = weight

    # debug
    """
    if n < 10:
      for channel in possible_channels:
        print channel
      print "result:", channel_is_true
      print "event number:", event_number,
      print "lumi:", luminosity,
      print "weight:", weight
      print mc_channel_number, run_number
      print "dphi:", lepton_pair_dphi
      print "dr:", lepton_pair_dR
    """
  if num_muti_channel_event > 0:
    first_info = "{} muti-channel events among {} events were found"\
                 .format(num_muti_channel_event, tree.GetEntries())
    second_info = "(in build_array_withcut)"
    third_info = "channel was chosen with priority emu > etau > mutau" 
    print("Warning: " + first_info + second_info + ", " + third_info)

  # remove empty and zero-weight event
  data = array_utils.clean_array(data, -1, remove_negative=False)
  return data

def dump_flat_ntuple(
  rootfile_path, tree_name, feature_list,
  save_ntuple = False,
  save_path = None,
  should_clean_array=True
  ):
  """Dumps flat ntuples to numpy array and save."""
  # The conversion of the TTree to a numpy array is implemented with multi-
  # thread support.
  ROOT.ROOT.EnableImplicitMT()
  # load tree
  print("building array for: ", rootfile_path)
  try:
    f = ROOT.TFile.Open(rootfile_path)
    tree = getattr(f, tree_name)
  except:
    raise OSError("Can not get tree.")
    return None
  total_entries = tree.GetEntries()
  if total_entries == 0:
    logging.warning("Empty tree detected! Returning empty array.")
    return np.array([])
  array = tree.AsMatrix(columns=feature_list)
  ll_m_list = []
  lep_a = ROOT.TLorentzVector(0, 0, 0, 0)
  lep_b = ROOT.TLorentzVector(0, 0, 0, 0)
  propagator = ROOT.TLorentzVector(0, 0, 0, 0)
  for i, entry in enumerate(array):
    # if emu
    if entry[-4] == 1.0:
      lep_a.SetPtEtaPhiM(entry[0], entry[1], entry[2], entry[3])  # electron
      lep_b.SetPtEtaPhiM(entry[4], entry[5], entry[6], entry[7])  # muon
    # if etau
    elif entry[-3] == 1.0:
      lep_a.SetPtEtaPhiM(entry[0], entry[1], entry[2], entry[3])  # electron
      lep_b.SetPtEtaPhiM(entry[8], entry[9], entry[10], entry[11])  # tau
    # if mutau
    elif entry[-2] == 1.0:
      lep_a.SetPtEtaPhiM(entry[4], entry[5], entry[6], entry[7])  # muon
      lep_b.SetPtEtaPhiM(entry[8], entry[9], entry[10], entry[11])  # tau
    # calculate dilepton mass
    propagator = lep_a + lep_b
    ll_m_list.append(propagator.M())
  # add ll_m to list
  ll_m_list = np.array(ll_m_list).reshape((array.shape[0], 1))
  array = np.concatenate((ll_m_list, array), axis=1)
  print(array.shape)
  return array

def dump_flat_ntuple_individual(
  root_path:str,
  ntuple_name:str,
  variable_list:list,
  save_dir:str,
  save_pre_fix:str,
  use_lower_var_name:bool=False) -> None:
  """Reads numpy array from ROOT ntuple and convert to numpy array.
   
  Note:
    Each branch will be saved as an individual file.

  """
  try:
    events = uproot.open(root_path)[ntuple_name]
  except:
    raise IOError("Can not get ntuple")
  print("Read arrays from:", root_path)
  for var in variable_list:
    if use_lower_var_name:
      file_name = save_pre_fix + '_' + var.lower()
    else:
      file_name = save_pre_fix + '_' + var
    print("Generating:", file_name)
    temp_arr = events.array(var)
    save_array(temp_arr, save_dir, file_name)
