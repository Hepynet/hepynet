import sys
sys.path.append("..") # add self-defined module in the parent path
sys.path.append("../..") # add self-defined module in the parent path

from lfv_pdnn_code_v1.make_array import make_array

print "*" * 80
print "tests make_array"
# rootfile_path = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC/mc16a/Signal/RPV/RPV_emu_2000GeV.root"
rootfile_path = "/eos/atlas/atlascerngroupdisk/phys-exotics/lpx/LFVZprime2018/MC/mc16a/Signal/RPV/RPV_etau_2000GeV.root"
make_array.build_array_withcut(rootfile_path)
