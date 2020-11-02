"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to a sample component
  2. Each sample component is a sub-dictionary, each key is corresponding to a input feature's array 

"""

import array
import os
from pathlib import Path

import numpy as np
import ROOT
import uproot
from lfv_pdnn.common import array_utils
from lfv_pdnn.common.logging_cfg import *


def load_npy_arrays(
    directory,
    campaign,
    region,
    channel,
    samples,
    selected_features,
    validation_features=[],
    cut_features=[],
    cut_values=[],
    cut_types=[],
    weight_scale=1,
):
    """Gets individual npy arrays with given info.

    Arguments:
        sample: can be a string (or list of string) of input sample name(s)
        sample_combine_method: can be
            "norm": to norm each input sample and combine together
            None(default): use original weight to directly combine

    Return:
        A dict of different variables
        Example: signal dict of rpv_500 rpv_1000 rpv_2000

    """

    # check different situation
    if type(samples) != list:
        sample_list = [samples]
    else:
        sample_list = samples
    if campaign in ["run2", "all"]:
        campaign_list = ["mc16a", "mc16d", "mc16e"]
    else:
        campaign_list = [campaign]
    # load arrays
    included_features = selected_features[:]
    included_features = list(set().union(included_features, validation_features, ["weight"]))
    cut_features += [channel]
    cut_values += [1]
    cut_types += ["="]
    out_dict = {}
    for sample_component in sample_list:
        sample_array_dict = {}
        cut_array_dict = {}
        for feature in set().union(included_features, cut_features):
            feature_array = None
            for campaign in campaign_list:
                temp_array = np.load(
                    f"{directory}/{campaign}/{region}/{sample_component}_{feature}.npy"
                )
                temp_array = np.reshape(temp_array, (-1, 1))
                if feature_array is None:
                    feature_array = temp_array
                else:
                    feature_array = np.concatenate((feature_array, temp_array))
            if feature in included_features:
                sample_array_dict[feature] = feature_array
            if feature in cut_features:
                cut_array_dict[feature] = feature_array
        # apply cuts
        ## Get indexes that pass cuts
        if not (
            len(cut_features) == len(cut_values) and len(cut_features) == len(cut_types)
        ):
            logging.critical(
                "cut_features and cut_values and cut_types should have same length"
            )
        pass_index = None
        for cut_feature, cut_value, cut_type in zip(cut_features, cut_values, cut_types):
            temp_pass_index = array_utils.get_cut_index_value(
                cut_array_dict[cut_feature], cut_value, cut_type
            )
            if pass_index is None:
                pass_index = temp_pass_index
            else:
                pass_index = np.intersect1d(pass_index, temp_pass_index)
        ## keep the events that pass the selection
        for feature in sample_array_dict:
            sample_array_dict[feature] = sample_array_dict[feature][pass_index.flatten(), :]
        total_weights = np.sum(sample_array_dict["weight"])
        logging.info(f"Total input {sample_component} weights: {total_weights}")
        out_dict[sample_component] = sample_array_dict
    return out_dict


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
