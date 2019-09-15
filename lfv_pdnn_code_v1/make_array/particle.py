from ROOT import TLorentzVector

# Constants
ELECTRON_MASS = 0.511
MUON_MASS = 105.658367e-3
TAU_MASS = 1.776

class electron_candidates:
  """A electron_candidate has electron collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    self.selected_el = TLorentzVector(0, 0, 0, 0)
    self.selected_el_charge = 0
    self.num_good_el = 0

    # get info from event
    self.el_pt = event.el_pt
    self.el_eta = event.el_eta
    self.el_cl_eta = event.el_cl_eta
    self.el_phi = event.el_phi
    self.el_e = event.el_e
    self.el_charge = event.el_charge
    self.el_topoetcone20 = event.el_topoetcone20
    self.el_ptvarcone20 = event.el_ptvarcone20
    self.el_cf = event.el_CF
    self.el_d0sig = event.el_d0sig
    self.el_delta_z0_sintheta = event.el_delta_z0_sintheta
    self.el_true_type = event.el_true_type
    self.el_true_origin = event.el_true_origin
    self.el_isolation_fixed_cut_loose = event.el_isolation_FixedCutLoose
    self.el_isolation_fixed_cut_tight = event.el_isolation_FixedCutTight
    self.el_is_tight = event.el_isElTight
    self.el_is_medium = event.el_isElMedium
    self.el_is_loose = event.el_isElLoose

  def select(self):
    """Makes cuts to all electron candidate"""
    # - actually there is only one item in each el_pt/eta/... although it's 
    # saved as an collection. 
    # - variables like el_is_medium was saved as a single value
    # - it's same for muon and tau
    for i in range(self.el_pt.size()):
      if self.el_pt[i] < 65000.0:  # el_pt is an collection
        continue
      if abs(self.el_eta[i]) > 2.47:
        continue
      if abs(self.el_eta[i]) > 1.37 and abs(self.el_eta[i]) < 1.52:
        continue
      if not self.el_is_medium:  # el_is_medium is a single value
        continue
      if not self.el_isolation_fixed_cut_tight:
        continue
      if abs(self.el_delta_z0_sintheta[i]) > 0.5:
        continue
      if abs(self.el_d0sig[i]) > 5.0:
        continue
      self.num_good_el += 1
      # pick highest pt
      if 0.001 * self.el_pt[i] > self.selected_el.Pt():
        self.selected_el.SetPtEtaPhiM(0.001 * self.el_pt[i], self.el_eta[i], 
                                      self.el_phi[i], ELECTRON_MASS * 0.001)
        self.selected_el_charge = self.el_charge[i]


class muon_candidates:
  """A muon_candidate has muon collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    self.selected_mu = TLorentzVector(0, 0, 0, 0)
    self.selected_mu_charge = 0
    self.num_good_mu = 0

    # get info from event
    self.mu = event.mu
    self.mu_actual = event.mu_actual
    self.has_bad_muon = event.hasBadMuon
    self.mu_pt = event.mu_pt
    self.mu_eta = event.mu_eta
    self.mu_phi = event.mu_phi
    self.mu_e = event.mu_e
    self.mu_charge = event.mu_charge
    self.mu_d0sig = event.mu_d0sig
    self.mu_delta_z0_sintheta = event.mu_delta_z0_sintheta
    self.mu_true_type = event.mu_true_type
    self.mu_true_origin = event.mu_true_origin
    self.mu_is_high_pt = event.mu_isHighPt
    self.mu_isolation_fixed_cut_tight = event.mu_isolation_FixedCutTight
    self.mu_isolation_fixed_cut_loose = event.mu_isolation_FixedCutLoose

  def select(self):
    """Makes cuts to all muon candidate"""
    # see note in electron_candidates
    for i in range(self.mu_pt.size()):
      if self.mu_pt[i] < 65000.0:
        continue
      if abs(self.mu_eta[i]) > 2.5:
        continue
      if not self.mu_is_high_pt:
        continue
      if not self.mu_isolation_fixed_cut_loose:
        continue
      if abs(self.mu_delta_z0_sintheta[i]) > 0.5:
        continue
      if abs(self.mu_d0sig[i]) > 3.0:
        continue
      self.num_good_mu += 1
      # pick highest pt
      if 0.001 * self.mu_pt[i] > self.selected_mu.Pt():
        self.selected_mu.SetPtEtaPhiM(0.001 * self.mu_pt[i], self.mu_eta[i], 
                                      self.mu_phi[i], MUON_MASS * 0.001)
        self.selected_mu_charge = self.mu_charge[i]


class tau_candidates:
  """A tau_candidate has tau collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    self.selected_tau = TLorentzVector(0, 0, 0, 0)
    self.selected_neutrino = TLorentzVector(0, 0, 0, 0)
    self.selected_tau_charge = 0
    self.num_good_tau = 0

    # get info from event
    self.tau_pt = event.tau_pt
    self.tau_eta = event.tau_eta
    self.tau_phi = event.tau_phi
    self.tau_charge = event.tau_charge
    self.tau_is_medium = event.tau_isMedium
    self.tau_is_Loose = event.tau_isLoose
    self.tau_bdt = event.tau_BDT
    self.tau_num_track = event.tau_nTracks
    self.met_met = event.met_met
    self.met_phi = event.met_phi

  def select(self):
    """Makes cuts to all tau candidate"""
    # see note in electron_candidates
    for i in range(self.tau_pt.size()):
      if self.tau_pt[i] < 65000.0:
        continue
      if abs(self.tau_eta[i]) > 2.47:
        continue
      if abs(self.tau_eta[i]) > 1.37 and abs(self.tau_eta[i]) < 1.52:
        continue
      if not self.tau_is_medium:
        continue
      if abs(self.tau_charge[i]) != 1.0:
        continue
      if self.tau_num_track[i] != 1.0 and self.tau_num_track[i] != 3.0:
        continue
      self.num_good_tau += 1
      # pick highest pt
      if 0.001 * self.tau_pt[i] > self.selected_tau.Pt():
        self.selected_tau.SetPtEtaPhiM(0.001 * self.tau_pt[i], self.tau_eta[i], 
                                      self.tau_phi[i], TAU_MASS * 0.001)
        self.selected_neutrino.SetPtEtaPhiM(self.met_met, self.tau_eta[i], 
                                            self.met_phi, 0.0)
        self.selected_tau_charge = self.tau_charge[i]


class jet_candidates:
  """A jet_candidate has jet collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    self.num_b_jet = 0
    self.b_tag_jet = False

    # get info from event
    self.jet_pt = event.jet_pt
    self.jet_eta = event.jet_eta
    self.jet_phi = event.jet_phi
    self.jet_e = event.jet_e
    self.jet_mv2c10 = event.jet_mv2c10
    self.jet_jvt = event.jet_jvt
    self.jet_passfjvt = event.jet_passfjvt

  def select(self):
    """Makes cuts to all jet candidate"""
    # see note in electron_candidates
    for i in range(self.jet_pt.size()):
      if self.jet_pt[i] < 20000.0:
        continue
      if abs(self.jet_eta[i]) > 2.5:
        continue
      # check jet cleaning
      if self.jet_pt[i] * 0.001 < 60.0 and abs(self.jet_eta[i]) < 2.4 and \
         self.jet_jvt[i] >= 0.59:
        continue
      if self.jet_pt[i] * 0.001 < 60.0 and abs(self.jet_eta[i]) >= 2.4:
        continue
      if self.jet_mv2c10[i] > 0.64:
        self.num_b_jet += 1
        b_tag_jet = True