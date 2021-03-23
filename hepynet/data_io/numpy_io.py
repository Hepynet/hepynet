"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to a sample component
  2. Each sample component is a sub-dictionary, each key is corresponding to a input feature's array 

"""
import logging
import pathlib
from typing import List, Optional

import numpy as np

from hepynet.common import common_utils, config_utils
from hepynet.data_io import array_utils

logger = logging.getLogger("hepynet")


def load_npy_arrays(job_config, array_type, part_features: Optional[List[str]] = None):
    """Gets individual npy arrays with given info.

    Return:
        A dict of subdict for different samples
            example: signal dict of rpv_500 rpv_1000 rpv_2000
        Each subdict is a dict of different variable
            examples: pT, eta, phi

    """
    # get settings from config file
    job_config_copy = job_config.clone()
    rc = job_config_copy.run
    ic = job_config_copy.input
    if array_type == "sig":
        samples = ic.sig_list
    elif array_type == "bkg":
        samples = ic.bkg_list
    elif array_type == "data":
        samples = ic.data_list
    else:
        logger.critical(f"Unsupported array_type: {array_type}")
        exit(1)

    # check different situation
    if type(samples) != list:
        sample_list = [samples]
    else:
        sample_list = samples

    # load arrays
    out_features = list()
    if part_features is None:
        out_features = ic.selected_features[:]
        out_features = list(
            set().union(out_features, ic.validation_features, ["weight"])
        )
    else:
        out_features = list(set().union(part_features, ["weight"]))
    ic.cut_features += [ic.channel]
    ic.cut_values += [1]
    ic.cut_types += ["="]
    out_dict = {}
    platform_meta = config_utils.load_current_platform_meta()
    data_directory = platform_meta["data_path"]
    ## check data_root_directory
    if not data_directory:
        current_hostname = common_utils.get_current_hostname()
        current_platform = common_utils.get_current_platform_name()
        logger.info(
            f"NO data_path setting for current host {current_hostname} with platform {current_platform} found at share/cross_platform/pc_meta.yaml, will consider using absolute path for input data"
        )
        data_directory = ""
    ## get campaign list
    campaign_list = list()
    if "+" in ic.campaign:
        campaign_list = [camp.strip() for camp in ic.campaign.split("+")]
    else:
        campaign_list = [ic.campaign]
    ## load sample array
    for sample_component in sample_list:
        sample_array_dict = {}
        cut_array_dict = {}
        for feature in set().union(out_features, ic.cut_features):
            feature_array = None
            for camp in campaign_list:
                temp_array = np.load(
                    f"{data_directory}/{rc.npy_path}/{camp}/{ic.region}/{sample_component}_{feature}.npy "
                )
                temp_array = np.reshape(temp_array, (-1, 1))
                if feature_array is None:
                    feature_array = temp_array
                else:
                    feature_array = np.concatenate((feature_array, temp_array))

            feature_array = feature_array.reshape((-1, 1))

            if feature in out_features:
                sample_array_dict[feature] = feature_array
            if feature in ic.cut_features:
                cut_array_dict[feature] = feature_array
        # apply cuts
        ## Get indexes that pass cuts
        if not (
            len(ic.cut_features) == len(ic.cut_values)
            and len(ic.cut_features) == len(ic.cut_types)
        ):
            logger.critical(
                "cut_features and cut_values and cut_types should have same length"
            )
            exit(1)
        pass_index = None
        for cut_feature, cut_value, cut_type in zip(
            ic.cut_features, ic.cut_values, ic.cut_types
        ):
            temp_pass_index = array_utils.get_cut_index_value(
                cut_array_dict[cut_feature], cut_value, cut_type
            )
            if pass_index is None:
                pass_index = temp_pass_index
            else:
                pass_index = np.intersect1d(pass_index, temp_pass_index)
        ## keep the events that pass the selection
        for feature in sample_array_dict:
            sample_array_dict[feature] = sample_array_dict[feature][
                pass_index.flatten(), :
            ]
        total_weights = np.sum(sample_array_dict["weight"])
        logger.debug(f"Total input {sample_component} weights: {total_weights}")
        out_dict[sample_component] = sample_array_dict
    return out_dict


def save_npy_array(
    npy_array, save_path,
):
    save_path = pathlib.Path(save_path)
    save_path.cwd().mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        np.save(f, npy_array)
    logger.debug(f"Array saved to: {save_path}")
    logger.debug(f"Array contents:\n {npy_array}")
    logger.debug(f"Array average:\n {np.average(npy_array)}")
    logger.debug(f"Array sum:\n {np.sum(npy_array)}")
