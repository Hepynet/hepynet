from lfv_pdnn.make_array.make_array import dump_flat_ntuple
from lfv_pdnn.make_array.make_array_utils import save_array
from lfv_pdnn.common.common_utils import get_file_list

input_ntuples_dir = "/home/paperspace/data/lfv/ntuples/current_run_jungao"
output_arrays_dir = "/home/paperspace/data/lfv/dnn_input_arrays/current_run_jungao"
mass_points = [500, 1000, 2000]

# Constants
ntuple_name = "Emutau"
feature_list = [
  'ele_pt', 'ele_eta', 'ele_phi', 'ele_e',
  'mu_pt', 'mu_eta', 'mu_phi', 'mu_e',
  'tau_pt', 'tau_eta', 'tau_phi', 'tau_e',
  'nu_e', 'nu_phi',
  'Pt_ll', 'Eta_ll', 'Phi_ll',
  'DPhi_ll', 'DR_ll',
  'emu', 'etau', 'mutau', 'weight'
  ]

# dump signal
output_arrays_subdir = output_arrays_dir + "/signal"
for mass in mass_points:
  print("*" * 80)
  for camp in ['a', 'd', 'e']:
    input_directory = input_ntuples_dir + "/MC16{}/RPV{}".format(camp, mass)
    output_directory = output_arrays_dir + "/signal/" + "{}GeV".format(mass)
    # emu
    search_pattern = "*emu*.root"
    absolute_file_list, file_name_list = get_file_list(
      input_directory, search_pattern,
      out_name_identifier = "rpv_emu_{}GeV.mc16{}".format(mass, camp)
      )
    for i, path in enumerate(absolute_file_list):
      array = dump_flat_ntuple(path, ntuple_name, feature_list)
      save_array(array, output_directory, file_name_list[i])
    # etau
    search_pattern = "*etau*.root"
    absolute_file_list, file_name_list = get_file_list(
      input_directory, search_pattern,
      out_name_identifier = "rpv_etau_{}GeV.mc16{}".format(mass, camp)
      )
    for i, path in enumerate(absolute_file_list):
      array = dump_flat_ntuple(path, ntuple_name, feature_list)
      save_array(array, output_directory, file_name_list[i])
    # mutau
    search_pattern = "*mutau*.root"
    absolute_file_list, file_name_list = get_file_list(
      input_directory, search_pattern,
      out_name_identifier = "rpv_mutau_{}GeV.mc16{}".format(mass, camp)
      )
    for i, path in enumerate(absolute_file_list):
      array = dump_flat_ntuple(path, ntuple_name, feature_list)
      save_array(array, output_directory, file_name_list[i])

# dump background
bkg_names = ['di_boson', 'top_quark', 'w_jets', 'z_ll']
bkg_folders = ['Diboson', 'Top', 'Wjets', 'LMass']
for (bkg_name, bkg_folder) in zip(bkg_names, bkg_folders):
  print("*" * 80)
  for camp in ['a', 'd', 'e']:
    input_directory = input_ntuples_dir + "/MC16{}/{}*/".format(camp, bkg_folder)
    output_directory = output_arrays_dir + "/" + bkg_name
    search_pattern = "*.root"
    absolute_file_list, file_name_list = get_file_list(
      input_directory, search_pattern,
      out_name_identifier = bkg_name + ".mc16{}".format(camp)
      )
    for i, path in enumerate(absolute_file_list):
      array = dump_flat_ntuple(path, ntuple_name, feature_list)
      save_array(array, output_directory, file_name_list[i])

print("*" * 80)