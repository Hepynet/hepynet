import io
import os

import numpy as np
import ROOT

from lfv_pdnn_code_v1.common import print_helper
from lfv_pdnn_code_v1.common.observable_cal import delta_phi

# root_vector_double = vector('double')()
# _length_const = 24
INV_MASS_LIMIT = 130.0


def get_early_channel(electrons, muons, taus):
  """Gets current event channel based on selected leptons with simple judge
     based on number of good leptons and charge, further selection with 
     kinematic info should be made later. See:
     get_final_channel(electrons, muons, taus, channel)

  Args:
    electrons: ElectronCandidates class, electron collection
    muons: MuonCandidates class, muon collection
    taus: TauCandidates class, taus collection
  Raises:
    ValueError: If input particle collections is not all selected
  Returns:
    A list of judgement on possible event channel, "emu"/"etau"/"mutau"
  """
  # Check selected
  if not (electrons.is_selected and muons.is_selected and taus.is_selected):
    raise ValueError("All input particle collections must be selected.")
  # Judge
  is_emu = False
  is_etau = False
  is_mutau = False
  possible_channels = []
  # emu
  has_emu = electrons.num_good_particle == 1 and muons.num_good_particle == 1
  charge_multiply_emu = electrons.selected_particle_charge
  charge_multiply_emu *= muons.selected_particle_charge
  has_opposite_charge_emu = charge_multiply_emu == -1
  if has_emu and has_opposite_charge_emu:
    is_emu = True
    possible_channels.append("emu")
  # etau
  has_etau = electrons.num_good_particle == 1 and taus.num_good_particle == 1
  charge_multiply_etau = electrons.selected_particle_charge
  charge_multiply_etau *= taus.selected_particle_charge
  has_opposite_charge_etau = charge_multiply_etau == -1
  if has_etau and has_opposite_charge_etau:
    is_etau = True
    possible_channels.append("etau")
  # mutau
  has_mutao = muons.num_good_particle == 1 and taus.num_good_particle == 1
  charge_multiply_mutao = muons.selected_particle_charge
  charge_multiply_mutao *= taus.selected_particle_charge
  has_opposite_charge_mutao = charge_multiply_mutao == -1
  if has_mutao and has_opposite_charge_mutao == 1:
    is_mutau = True
    possible_channels.append("mutau")
  return possible_channels


def get_final_channel(electrons, muons, taus, channel):
  """Gives final decision whether a channel judgement is correct.
     Must be used with get_early_channel(electrons, muons, taus)
  Args:
    electrons: ElectronCandidates class, electron collection
    muons: MuonCandidates class, muon collection
    taus: TauCandidates class, taus collection
    channel: str, early selected channel
  Returns:
    channel_is_true: Judgement on whether the channel selection is True
    propagator: Lorentz vector of selected lepton pair
  """
  channel_is_true = False
  propagator = ROOT.TLorentzVector(0, 0, 0, 0)
  # Get selected particles
  electron = electrons.selected_particle
  muon = muons.selected_particle
  tau = taus.selected_particle
  neutrino = taus.selected_neutrino
  # Check
  if channel == "emu":
    propagator = electron + muon
    pair_delta_phi = delta_phi(electron.Phi(), muon.Phi())
    if propagator.Mag() > INV_MASS_LIMIT and pair_delta_phi > 2.7:
      channel_is_true = True
  elif channel == "etau":
    propagator = electron + tau + neutrino
    pair_delta_phi = delta_phi(electron.Phi(), tau.Phi())
    if propagator.Mag() > INV_MASS_LIMIT and pair_delta_phi > 2.7:
      channel_is_true = True
  elif channel == "mutau":
    propagator = muon + tau
    pair_delta_phi = delta_phi(muon.Phi(), tau.Phi())
    if propagator.Mag() > INV_MASS_LIMIT and pair_delta_phi > 2.7:
      channel_is_true = True
  else:
    raise ValueError("unexpected channel")
  return channel_is_true, propagator


def get_lumi(run_number):
  """Gets luminosity according to given run number.
  
  Args:
    run_number: float, run number of the event
  """
  if run_number == 284500.0:
    return 36.07456
  elif run_number == 300000.0:
    return 44.30
  elif run_number == 310000.0:
    return 58.4501
  else:
    print_helper.print_warning("unknown run_number, luminosity set to zero")
    return 0


def save_array(array, directory_path, file_name):
  """Saves numpy data as .npy file

  Args:
    array: numpy array, array to be saved
    directory_path: str, directory path to save the file
    file_name: str, file name used by .npy file
  """
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)
  with io.open(directory_path + '/' + file_name + '.npy', 'wb') as f:
    np.save(f, array)