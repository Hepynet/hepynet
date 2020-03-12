import os
from lfv_pdnn.make_array.make_array import dump_flat_ntuple_individual

# Constants
# 'Pt_ll', 'Eta_ll', 'Phi_ll', 'DR_ll' not available now 
ntuple_name = "Emutau"
feature_list = [
  'ele_pt', 'ele_eta', 'ele_phi', 'ele_d0sig', 'ele_dz0', 'ele_isTight', 'ele_C',
  'mu_pt', 'mu_eta', 'mu_phi', 'mu_d0sig', 'mu_dz0', 'mu_isTight', 'mu_C',
  'tau_pt', 'tau_eta', 'tau_phi', 'tau_isTight', 'tau_C', 
  'Pt_ll', 'Eta_ll', 'Phi_ll', 'DPhi_ll', 'DR_ll',
  'met', 'met_eta', 'met_phi',
  'njets', 'nbjets', 'NTauLoose', 'NTauTight',
  'emu', 'etau', 'mutau', 'weight']
bkg_names = ['diboson', 'zll', 'top', 'wjets']
sig_names = ['rpv_500', 'rpv_700', 'rpv_1000', 'rpv_1500', 'rpv_2000',
             'zpr_500', 'zpr_700', 'zpr_1000', 'zpr_1500', 'zpr_2000']

# path in docker
ntup_dir = "/data/ntuples/rel_103/mc16e/merged"
arrays_dir = "/data/arrays/rel_103/mc16e"
if not os.path.exists(arrays_dir):
  os.makedirs(arrays_dir)

# Dump bkg
for bkg_name in bkg_names:
  root_path = ntup_dir + "/bkg_{}.root".format(bkg_name)
  print(root_path)
  dump_flat_ntuple_individual(root_path, ntuple_name, feature_list, arrays_dir, 
                              "bkg_{}".format(bkg_name))
# dump sig
for sig_name in sig_names:
  root_path = ntup_dir + "/sig_{}.root".format(sig_name)
  print(root_path)
  dump_flat_ntuple_individual(root_path, ntuple_name, feature_list, arrays_dir,
                              "sig_{}".format(sig_name))
