import array
import logging
import os
from pathlib import Path

try:
    import ROOT
    import uproot
except:
    logging.info("Can't import ROOT or uproot, root_io module can't be used")

def dump_ntup_from_npy(ntup_name, branch_list, branch_type, contents, out_path):
    """Generates ntuples from numpy arrays."""
    out_dir = Path(out_path).parent
    os.makedirs(out_dir, exist_ok=True)
    out_file = ROOT.TFile(out_path, "RECREATE")
    out_file.cd()
    out_ntuple = ROOT.TNtuple(ntup_name, ntup_name, ":".join(branch_list))
    n_branch = len(branch_list)
    n_entries = len(contents[0])
    for i in range(n_entries):
        fill_values = []
        for j in range(n_branch):
            fill_values.append(contents[j][i])
        out_ntuple.Fill(array.array(branch_type, fill_values))
    out_file.cd()
    out_ntuple.Write()
