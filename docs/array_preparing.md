# Prepare the input arrays

Input ntuple must be flat, each input feature should have same quantities.
Besides, "weight" and "channel" array is needed. If no "channel" branch is in
the ntuple or there is only one channel, one should create the "dummy channel"
branch.

## Example code
```python
import os
from hepynet.make_array.make_array import dump_flat_ntuple_individual

# Constants
ntuple_name = "Emutau"
feature_list = ['M_ll', 'ele_pt', 'mu_pt', 'emu', 'weight']
bkg_names = ['diboson', 'zll', 'top', 'wjets']
sig_names = ['rpv_500', 'rpv_700', 'rpv_1000', 'rpv_1500', 'rpv_2000']

# Set path in docker
ntup_dir = "/data/lfv/ntuples/rel_103_v4/merged"
arrays_dir = "/data/lfv/arrays/rel_103_v4"
if not os.path.exists(arrays_dir):
  os.makedirs(arrays_dir)

for camp in ["mc16a", "mc16d", "mc16e"]:
  # Dump bkg
  for bkg_name in bkg_names:
    root_path = ntup_dir + "/" + camp + "/bkg_{}.root".format(bkg_name)
    dump_flat_ntuple_individual(root_path, ntuple_name, feature_list,
      arrays_dir + "/" + camp, "bkg_{}".format(bkg_name),
      use_lower_var_name=True)
  # Dump sig
  for sig_name in sig_names:
    root_path = ntup_dir + "/" + camp + "/sig_{}.root".format(sig_name)
    dump_flat_ntuple_individual(root_path, ntuple_name, feature_list,
      arrays_dir + "/" + camp, "sig_{}".format(sig_name),
      use_lower_var_name=True)
  # Dump data
  root_path = ntup_dir + "/" + camp + "/data.root"
  dump_flat_ntuple_individual(root_path, ntuple_name, feature_list,
    arrays_dir + "/" + camp, "data_all", use_lower_var_name=True)

```

## Example notes
1. First import "dump_flat_ntuple_individual" and specify the ntuple name in
    root file, name list of input features and input samples names to be used
    later in arrays production.
2. Specify the input directory where the ntuples root file were stores. If Docker
    is used, please change the path defined in docker image.
3. Call "dump_flat_ntuple_individual" to dump arrays from ntuples. Note the
    arrays saving directory should be: /array_save_dir/campaign/