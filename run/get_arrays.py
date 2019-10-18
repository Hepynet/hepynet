import numpy as np

from lfv_pdnn_code_v1.common.common_utils import *
from lfv_pdnn_code_v1.train.train_utils import *

NEW_BKG_NAMES = ['di_boson', 'top_quark', 'w_jets', 'z_ll']
NEW_MASS_MAP = [500, 2000]
OLD_MASS_MAP = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000]

def get_new_bkg(data_path):
  # Load
  xb_dict_new = {}
  print("Loading new background array.")
  for bkg in NEW_BKG_NAMES:
    directory = data_path + "/{}".format(bkg)
    search_pattern = "*.npy"
    absolute_file_list, file_name_list = get_file_list(directory, search_pattern)
    xb_single = np.array([])
    for path in absolute_file_list:
        temp_array = np.load(path)
        if len(temp_array) == 0:
            continue
        elif len(xb_single) == 0:
            xb_single = temp_array.copy()
        else:
            xb_single = np.concatenate((xb_single, temp_array))
    xb_dict_new[bkg] = xb_single
    print("xb_{} shape:".format(bkg), xb_single.shape)
  print("New background organized with dict: xb_dict_new")
  # Add all background together
  print ("Adding all background together.")
  xb = np.concatenate(list(xb_dict_new.values()))
  print ("xb shape:", xb.shape)
  return xb_dict_new

def get_new_sig(data_path):
  # Initialize
  mass_min = 5000
  mass_max = 0
  xs_studied = np.array([])
  xs_dict_new = {}
  # Load
  print("Loading new signal array.")
  print("Organizing new signal with dict: xs_dict_new.")
  xs = np.array([])
  xs_norm = np.array([])
  for i, mass in enumerate(NEW_MASS_MAP):
      # load signal
      xs_add = np.load(data_path + "/signal/rpv_emu_{}GeV.npy".format(mass))
      xs_temp = np.load(data_path + "/signal/rpv_etau_{}GeV.npy".format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      xs_temp = np.load(data_path + "/signal/rpv_mutau_{}GeV.npy".format(mass))
      xs_add = np.concatenate((xs_add, xs_temp))
      # add to dict xs_dict_new
      print("adding {}GeV signal to xs_dict_new".format(mass), xs_add.shape)
      xs_dict_new['{}GeV'.format(mass)] = xs_add
      # add to full signals
      if len(xs) == 0:
          xs = xs_add.copy()
          xs_add_norm = modify_array(xs_add, weight_id = -1, norm = True)
          xs_norm = xs_add_norm.copy()
      else:
          xs = np.concatenate((xs, xs_add))
          xs_add_norm = modify_array(xs_add, weight_id = -1, norm = True)
          xs_norm = np.concatenate((xs_norm, xs_add_norm))
  print("adding all signal to xs_dict_new")
  xs_dict_new['all'] = xs
  print("adding all_norm signal to xs_dict_new")
  xs_dict_new['all_norm'] = xs_norm
  print("Done.")
  return xs_dict_new

def get_old_bkg(data_path):
  # Load
  print("Loading new background array.")
  xb_di_boson_old = np.load(data_path + '/tree_bkg1.npy')
  xb_drell_yan_old = np.load(data_path + '/tree_bkg2.npy')
  xb_top_quark_old = np.load(data_path + '/tree_bkg3.npy')
  xb_w_jets_old = np.load(data_path + '/tree_bkg4.npy')
  xb_z_ll_old = np.load(data_path + '/tree_bkg5.npy')
  xb_old = np.concatenate((xb_di_boson_old, xb_drell_yan_old, xb_top_quark_old, xb_w_jets_old, xb_z_ll_old))
  # Organize with dict
  print("Organizing new background with dict: xb_dict_old.")
  xb_dict_old = {}
  xb_dict_old['di_boson'] = xb_di_boson_old
  xb_dict_old['drell_yan'] = xb_drell_yan_old
  xb_dict_old['top_quark'] = xb_top_quark_old
  xb_dict_old['w_jets'] = xb_w_jets_old
  xb_dict_old['z_ll'] = xb_z_ll_old
  xb_dict_old['all'] = xb_old
  print("Done.")
  return xb_dict_old

def get_old_sig(data_path):
  # Initialize
  mass_min = 5000
  mass_max = 0
  xs_old = None
  xs_old_norm = None
  xs_dict_old = {}
  # Load
  print("Loading old signal array.")
  print("Organizing old signal with dict: xs_dict_old.")
  for i, mass in enumerate(OLD_MASS_MAP):
    # load signal
    xs_add = np.load(data_path + '/rpv_{}GeV.npy'.format(mass))
    xs_temp = np.load(data_path + '/rpv_{}GeV.npy'.format(mass))
    xs_add = np.concatenate((xs_add, xs_temp))
    xs_temp = np.load(data_path + '/rpv_{}GeV.npy'.format(mass))
    xs_add = np.concatenate((xs_add, xs_temp))
    xs_add_norm = modify_array(xs_add, weight_id=-1, norm=True)
    # add to dict xs_dict_new
    xs_dict_old['{}GeV'.format(mass)] = xs_add
    # add to full signals
    if xs_old is None:
      xs_old = xs_add.copy()
      xs_old_norm = xs_add_norm.copy()
    else:
      xs_old = np.concatenate((xs_old, xs_add))
      xs_old_norm = np.concatenate((xs_old_norm, xs_add_norm))
  xs_dict_old['all'] = xs_old
  xs_dict_old['all_norm'] = xs_old_norm
  print("Done.")
  return xs_dict_old
