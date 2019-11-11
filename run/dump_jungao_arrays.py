from lfv_pdnn.make_array.make_array import dump_flat_ntuple
from lfv_pdnn.make_array.make_array_utils import save_array
from lfv_pdnn.common.common_utils import get_file_list

input_ntuples_dir = "/home/paperspace/data/lfv/ntuples/current_run_jungao/"
output_arrays_dir = "/home/paperspace/data/lfv/dnn_input_arrays/"

feature_list = [
  'ele_pt', 'ele_eta', 'ele_phi', 'ele_e',
  'emu', 'weight'
  ]

# dump signal
test_sig_path = input_ntuples_dir + "MC16a/RPV500/MadGraphPythia8EvtGen_A14NNPDF23LO_SVT_emu_m500_Skim.root"
dump_flat_ntuple(test_sig_path, "Emutau", feature_list, should_clean_array=True)