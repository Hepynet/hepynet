import math

from ROOT import TLorentzVector

from lfv_pdnn_code_v1.common.observable_cal import delta_phi, delta_r

# Constants unit: GeV
ELECTRON_MASS = 0.511e-3
MUON_MASS = 105.658367e-3
TAU_MASS = 1.776

class Particles(object):
  """A collection of particles"""
  def __init__(self):
    # intialize variables to be used
    self.selected_particle = TLorentzVector(0, 0, 0, 0)
    self.selected_particle_charge = 0
    self.num_good_particle = 0
    self.is_selected = False

  def select(self):
    """Makes cuts to all particles"""
    self.is_selected = True


class ElectronCandidates(Particles):
  """A electron_candidate has electron collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    Particles.__init__(self)
    # get info from event
    self.pt = event.el_pt
    self.eta = event.el_eta
    self.cl_eta = event.el_cl_eta
    self.phi = event.el_phi
    self.e = event.el_e
    self.charge = event.el_charge
    self.topoetcone20 = event.el_topoetcone20
    self.ptvarcone20 = event.el_ptvarcone20
    self.cf = event.el_CF
    self.d0sig = event.el_d0sig
    self.delta_z0_sintheta = event.el_delta_z0_sintheta
    self.true_type = event.el_true_type
    self.true_origin = event.el_true_origin
    self.isolation_fixed_cut_loose = event.el_isolation_FixedCutLoose
    self.isolation_fixed_cut_tight = event.el_isolation_FixedCutTight
    self.is_tight = event.el_isElTight
    self.is_medium = event.el_isElMedium
    self.is_loose = event.el_isElLoose

  def select(self):
    """Makes cuts to all electron candidate"""
    # - actually there is only one item in each pt/eta/... although it's 
    # saved as an collection. 
    # - variables like is_medium was saved as a single value
    # - it's same for muon and tau
    for i in range(self.pt.size()):
      if self.pt[i] < 65000.0:  # pt is an collection
        continue
      if abs(self.eta[i]) > 2.47:
        continue
      if abs(self.eta[i]) > 1.37 and abs(self.eta[i]) < 1.52:
        continue
      if not self.is_medium:  # is_medium is a single value
        continue
      if not self.isolation_fixed_cut_tight:
        continue
      if abs(self.delta_z0_sintheta[i]) > 0.5:
        continue
      if abs(self.d0sig[i]) > 5.0:
        continue
      self.num_good_particle += 1
      # pick highest pt
      if 0.001 * self.pt[i] > self.selected_particle.Pt():
        self.selected_particle.SetPtEtaPhiM(0.001 * self.pt[i], 
                                            self.eta[i], self.phi[i], 
                                            ELECTRON_MASS)
        self.selected_particle_charge = self.charge[i]
    self.is_selected = True


class MuonCandidates(Particles):
  """A muon_candidate has muon collection to be selected"""
  def __init__(self, event):
    # intialize variables to be used
    Particles.__init__(self)
    # get info from event
    self.mu = event.mu
    self.has_bad_muon = event.hasBadMuon
    self.pt = event.mu_pt
    self.eta = event.mu_eta
    self.phi = event.mu_phi
    self.e = event.mu_e
    self.charge = event.mu_charge
    self.d0sig = event.mu_d0sig
    self.delta_z0_sintheta = event.mu_delta_z0_sintheta
    self.true_type = event.mu_true_type
    self.true_origin = event.mu_true_origin
    self.is_high_pt = event.mu_isHighPt
    self.isolation_fixed_cut_tight = event.mu_isolation_FixedCutTight
    self.isolation_fixed_cut_loose = event.mu_isolation_FixedCutLoose

  def select(self):
    """Makes cuts to all muon candidate"""
    # see note in ElectronCandidates
    for i in range(self.pt.size()):
      if self.pt[i] < 65000.0:
        continue
      if abs(self.eta[i]) > 2.5:
        continue
      if not self.is_high_pt:
        continue
      if not self.isolation_fixed_cut_loose:
        continue
      if abs(self.delta_z0_sintheta[i]) > 0.5:
        continue
      if abs(self.d0sig[i]) > 3.0:
        continue
      self.num_good_particle += 1
      # pick highest pt
      if 0.001 * self.pt[i] > self.selected_particle.Pt():
        self.selected_particle.SetPtEtaPhiM(0.001 * self.pt[i], self.eta[i], 
                                      self.phi[i], MUON_MASS)
        self.selected_particle_charge = self.charge[i]
    self.is_selected = True


class TauCandidates(Particles):
  """A tau_candidate has tau collection to be selected"""
  def __init__(self, event):
   # intialize variables to be used
    Particles.__init__(self)
    self.selected_neutrino = TLorentzVector(0, 0, 0, 0)
    # get info from event
    self.pt = event.tau_pt
    self.eta = event.tau_eta
    self.phi = event.tau_phi
    self.charge = event.tau_charge
    self.is_medium = event.tau_isMedium
    #self.is_loose = event.tau_isLoose  # not available in all trees
    self.bdt = event.tau_BDT
    self.num_track = event.tau_nTracks
    self.met_met = event.met_met
    self.met_phi = event.met_phi

  def select(self, electrons, muons):
    """Makes cuts to all tau candidate
    
    Must be used after electron and muon selection
    """
    # see note in ElectronCandidates
    # Get electron/muon info
    electron = electrons.selected_particle
    muon = muons.selected_particle
    num_good_electron = electrons.num_good_particle
    num_good_muon = muons.num_good_particle
    # Loop
    for i in range(self.pt.size()):
      if self.pt[i] < 65000.0:
        continue
      if abs(self.eta[i]) > 2.47:
        continue
      if abs(self.eta[i]) > 1.37 and abs(self.eta[i]) < 1.52:
        continue
      if not self.is_medium:
        continue
      if abs(self.charge[i]) != 1.0:
        continue
      if self.num_track[i] != 1.0 and self.num_track[i] != 3.0:
        continue
      self.num_good_particle += 1
      # Remove fake tau
      # remove fake electron tau
      dphi_etau = delta_phi(electron.Phi(), self.phi[i])
      dphi_enu = delta_phi(electron.Phi(), self.met_phi)
      mt_etau = math.sqrt(2.0 * electron.Pt() * self.met_met 
                          * (1.0 - math.cos(dphi_enu)))
      is_fake_tau = False
      for el_id in range(electrons.pt.size()):
        dr = delta_r(electrons.eta[el_id], electrons.phi[el_id], 
                     self.eta[i], self.phi[i])
        if dr < 0.2:
          is_fake_tau = True
      if is_fake_tau:
        continue
      # remove fake muon tau
      dphi_mutau = delta_phi(muon.Phi(), self.phi[i])
      dphi_munu = delta_phi(muon.Phi(), self.met_phi)
      mt_mutau = math.sqrt(2.0 * muon.Pt() * self.met_met
                           * (1.0 - math.cos(dphi_munu)))
      is_fake_tau = False
      for el_id in range(muons.pt.size()):
        dr = delta_r(muons.eta[el_id], muons.phi[el_id], 
                     self.eta[i], self.phi[i])
        if dr < 0.2:
          is_fake_tau = True
      if is_fake_tau:
        continue
      # mt/dphi cut
      pass_etau = (mt_etau > 80.0) and (dphi_etau > 2.7)
      pass_mutau = (mt_mutau > 80.0) and (dphi_mutau > 2.7)
      if not (pass_etau or pass_mutau):
        continue
      # Pick highest pt
      if 0.001 * self.pt[i] > self.selected_particle.Pt():
        self.selected_particle.SetPtEtaPhiM(0.001 * self.pt[i], self.eta[i], 
                                      self.phi[i], TAU_MASS)
        self.selected_neutrino.SetPtEtaPhiM(self.met_met, self.eta[i], 
                                            self.met_phi, 0.0)
        self.selected_particle_charge = self.charge[i]
    self.is_selected = True    


class JetCandidates(Particles):
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
    # see note in ElectronCandidates
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
    self.is_selected = True    


class Propagator(Particles):
  """A Propagator of lepton pair"""
  def __init__(self, lepton1, lepton2):
    pass

