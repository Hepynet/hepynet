import datetime
import sys
sys.path.append("..") # add self-defined module in the parent pathS

from lfv_pdnn_code_v1.make_array import make_array
from lfv_pdnn_code_v1.make_array.make_array_utils import save_array
from lfv_pdnn_code_v1.common.common_utils import get_file_list

input_parent_directory = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC"
output_parent_directory = "/afs/cern.ch/work/y/yangz/public/lfv/lfv_pdnn_code_v1/run/array_output"\
                          + datetime.date.today().strftime("%m_%d_%Y")

# Build signal arrays
output_directory = output_parent_directory + "/signal"
for mass in [500, 2000]:  # currently only mc16a 500/2000GeV signal ready (09/17/2019)
  print("*" * 80)
  rootfile_path = input_parent_directory + "/mc16a/Signal/RPV/RPV_emu_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_emu_{}GeV".format(mass))

  rootfile_path = input_parent_directory + "/mc16a/Signal/RPV/RPV_etau_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_etau_{}GeV".format(mass))

  rootfile_path = input_parent_directory + "/mc16a/Signal/RPV/RPV_mutau_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_mutau_{}GeV".format(mass))

# Build background arrays
# di_boson
input_directory = input_parent_directory + "/mc16a/Diboson"
output_directory = output_parent_directory + "/di_boson"
search_pattern = "*/*.root"
absolute_file_list, file_name_list = get_file_list(input_directory, search_pattern, out_name_pattern = "None")
for i, path in enumerate(absolute_file_list):
  print("*" * 80)
  array = make_array.build_array_withcut(path)
  save_array(array, output_directory, file_name_list[i])

# top_quark
input_directory = input_parent_directory + "/mc16a/TopQuark"
output_directory = output_parent_directory + "/top_quark"
search_pattern = "*/*.root"
absolute_file_list, file_name_list = get_file_list(input_directory, search_pattern, out_name_pattern = "None")
for i, path in enumerate(absolute_file_list):
  print("*" * 80)
  array = make_array.build_array_withcut(path)
  save_array(array, output_directory, file_name_list[i])

# w_jets
input_directory = input_parent_directory + "/mc16a/Wjets"
output_directory = output_parent_directory + "/w_jets"
search_pattern = "*/*.root"
absolute_file_list, file_name_list = get_file_list(input_directory, search_pattern, out_name_pattern = "None")
for i, path in enumerate(absolute_file_list):
  print("*" * 80)
  array = make_array.build_array_withcut(path)
  save_array(array, output_directory, file_name_list[i])

# z_ll
input_directory = input_parent_directory + "/mc16a/Ztoll"
output_directory = output_parent_directory + "/z_ll"
search_pattern = "*/*.root"
absolute_file_list, file_name_list = get_file_list(input_directory, search_pattern, out_name_pattern = "None")
for i, path in enumerate(absolute_file_list):
  print("*" * 80)
  array = make_array.build_array_withcut(path)
  save_array(array, output_directory, file_name_list[i])