import sys
sys.path.append("..") # add self-defined module in the parent path
sys.path.append("../..") # add self-defined module in the parent path

import matplotlib.pyplot as plt

from lfv_pdnn.make_array import make_array
from lfv_pdnn.make_array.make_array_utils import save_array

output_directory = "/afs/cern.ch/work/y/yangz/public/lfv/lfv_pdnn/run/test_run"

print("*" * 80)
print("tests make_array")

# build signal arrays

for mass in [500]:
  rootfile_path = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC/mc16a/Signal/RPV/RPV_emu_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_emu_{}GeV".format(mass))

  print("num of entries", len(array))
  plt.hist(array[:,3], bins=40, weights=array[:,-1], histtype='step', label='signal', range=(-4, 4))
  plt.show()
  plt.savefig("phi_distribution.png")

  """
  rootfile_path = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC/mc16a/Signal/RPV/RPV_etau_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_etau_{}GeV".format(mass))

  rootfile_path = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC/mc16a/Signal/RPV/RPV_mutau_{}GeV.root".format(mass)
  array = make_array.build_array_withcut(rootfile_path)
  save_array(array, output_directory, "rpv_mutau_{}GeV".format(mass))
  """