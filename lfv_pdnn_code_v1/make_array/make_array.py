import numpy as np
import ROOT

from lfv_pdnn_code_v1.common import common_utils, observable_cal, print_helper
from lfv_pdnn_code_v1.common import array_utils
from lfv_pdnn_code_v1.make_array import make_array_utils, particle
from make_array_utils import get_lumi, get_early_channel, get_final_channel


# normalization file path
norm_file_284500 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_284500.txt"
norm_file_300000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_300000.txt"
norm_file_310000 = "/afs/cern.ch/work/y/yangz/public/lfv/" + \
                   "lfv_pdnn_code_v1/data/make_array_info/norm_file_310000.txt"

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

  """
  ARRAY_ELEMENT_LENTH = 24  # number of variables to stored for each event

  # Prepare
  # load tree
  print "building array for: ", rootfile_path
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
  num_ignored_muti_channel_event = 0
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
    #print "selected pt: ", taus.selected_tau.Pt(), taus.selected_neutrino.Pt() ########
    #print "b_tag_jet:", jets.b_tag_jet #########################################

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
    if num_valid_channel > 1:
      num_ignored_muti_channel_event += 1
      #print_helper.print_warning("more than one valid channel detected", 
      #                           "in make_array module,", "event ignored")
      #print possible_channels
      continue

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
  if num_ignored_muti_channel_event > 0:
    first_part = "{} muti-channel events among {} events were"\
                 .format(num_ignored_muti_channel_event, tree.GetEntries())
    print_helper.print_warning(first_part, "ignored (in build_array_withcut)")

  # remove empty and zero-weight event
  data = array_utils.clean_array(data, -1, remove_negative=False)
  return data