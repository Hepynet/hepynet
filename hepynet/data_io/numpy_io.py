"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to a sample component
  2. Each sample component is a sub-dictionary, each key is corresponding to a input feature's array 


TODO:
    Numpy input format has been replaced by pandas DataFrame.
    Should remove numpy_io or add support for different input formats?

"""
import logging
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hepynet.common import common_utils, config_utils
from hepynet.data_io import array_utils

logger = logging.getLogger("hepynet")


def get_array_key(job_config, sample_type):
    ic = job_config.input.clone()
    if sample_type == "sig":
        return ic.sig_key
    elif sample_type == "bkg":
        return ic.bkg_key
    elif sample_type == "data":
        return ic.data_key
    else:
        logger.error(f"Unknown sample type: {sample_type}")
        return None


def get_campaign_list(job_config):
    ic = job_config.input.clone()
    if "+" in ic.campaign:
        return [camp.strip() for camp in ic.campaign.split("+")]
    else:
        return [ic.campaign]


def get_cut_index(job_config, cut_feature_dict):
    ic = job_config.input.clone()
    cut_index = None
    for cut_feature, cut_value, cut_type in zip(
        ic.cut_features, ic.cut_values, ic.cut_types
    ):
        cut_array = cut_feature_dict[cut_feature]
        if cut_type == "=":
            temp_cut_index = np.argwhere(cut_array != cut_value)
        elif cut_type == ">":
            temp_cut_index = np.argwhere(cut_array <= cut_value)
        elif cut_type == "<":
            temp_cut_index = np.argwhere(cut_array >= cut_value)
        else:
            temp_cut_index = np.array([])
        # update cut_index
        if cut_index is None:
            cut_index = temp_cut_index
        else:
            cut_index = np.union1d(cut_index, temp_cut_index)
    return cut_index


def get_cut_pass_index_dict(
    job_config, array_type_list=None, sample_list_dict=None
) -> Dict[str, Dict[str, np.ndarray]]:
    job_config_copy = job_config.clone()
    ic = job_config_copy.input
    # rc = job_config_copy.run
    # ic.cut_features += [ic.channel]
    # ic.cut_values += [1]
    # ic.cut_types += ["="]
    pass_id_dict = dict()
    if array_type_list is None:
        array_type_list = ["sig", "bkg", "data"]
    for array_type in array_type_list:
        pass_id_dict[array_type] = dict()
        sample_load_list = list()
        if (sample_list_dict) is None or (array_type not in sample_list_dict):
            samples = getattr(ic, f"{array_type}_list")
            if type(samples) != list:
                sample_load_list = [samples]
            else:
                sample_load_list = samples
        else:
            sample_load_list = sample_list_dict[array_type]
        data_dir = get_data_dir()
        ## get campaign list
        campaign_list = list()
        if "+" in ic.campaign:
            campaign_list = [camp.strip() for camp in ic.campaign.split("+")]
        else:
            campaign_list = [ic.campaign]
        ## load sample array
        for sample_component in sample_load_list:
            cut_array_dict = {}
            for feature in ic.cut_features:
                feature_array = None
                for camp in campaign_list:
                    temp_array = np.load(
                        f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}/{sample_component}_{feature}.npy ",
                    )
                    if feature_array is None:
                        feature_array = temp_array
                    else:
                        feature_array = np.concatenate(
                            (feature_array, temp_array)
                        )

                feature_array = feature_array

                cut_array_dict[feature] = feature_array.flatten()
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
            logger.debug(
                f"Got cut-passed index for {array_type} {sample_component}"
            )
            if pass_index is not None:
                logger.debug(f"Shape {pass_index.shape}")
            pass_id_dict[array_type][sample_component] = pass_index
    return pass_id_dict


def get_data_dir():
    platform_meta = config_utils.load_current_platform_meta()
    data_dir = platform_meta["data_path"]
    ## check data_root_directory
    if not data_dir:
        current_hostname = common_utils.get_current_hostname()
        current_platform = common_utils.get_current_platform_name()
        logger.info(
            f"NO data_path setting for current host {current_hostname} with platform {current_platform} found at share/cross_platform/pc_meta.yaml, will consider using absolute path for input data"
        )
        data_dir = ""
    return data_dir


def get_samples_total_weight(job_config, sample_list):
    ic = job_config.input.clone()
    data_dir = get_data_dir()
    total_weight = 0
    for sample in sample_list:
        for camp in get_campaign_list(job_config):
            npy_in_dir = pathlib.Path(
                f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}"
            )
            wt_path = npy_in_dir / f"{sample}_weight.npy"
            wt = np.load(wt_path)
            total_weight += np.sum(wt)
    return total_weight


def get_sumofweight(job_config, sample_type):
    ic = job_config.input.clone()
    if sample_type == "sig":
        return ic.sig_sumofweight
    elif sample_type == "bkg":
        return ic.bkg_sumofweight
    elif sample_type == "data":
        return ic.data_sumofweight
    else:
        logger.error(f"Unknown sample type: {sample_type}")
        return None


def load_npy_arrays(
    job_config,
    array_type,
    part_features: Optional[List[str]] = None,
    cut_pass_index_dict=None,
    include_weight=True,
):
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
        out_features += ic.selected_features
        out_features += ic.validation_features
    else:
        out_features += part_features
    if include_weight:
        out_features += ["weight"]
    out_features = list(set().union(out_features))  # remove duplicates
    logger.debug(f"Out_features {out_features}")
    out_dict = {}
    ## check data_root_directory
    data_dir = get_data_dir()
    ## get campaign list
    campaign_list = list()
    if "+" in ic.campaign:
        campaign_list = [camp.strip() for camp in ic.campaign.split("+")]
    else:
        campaign_list = [ic.campaign]
    ## load sample array
    for sample_component in sample_list:
        pass_index = None
        if (
            isinstance(cut_pass_index_dict, dict)
            and array_type in cut_pass_index_dict
            and sample_component in cut_pass_index_dict[array_type]
        ):
            pass_index = cut_pass_index_dict[array_type][sample_component]
        else:
            feature_cut_dict = get_cut_pass_index_dict(
                job_config,
                array_type_list=[array_type],
                sample_list_dict={array_type: [sample_component]},
            )
            pass_index = feature_cut_dict[array_type][sample_component]
        sample_array_dict = {}
        for feature in out_features:
            feature_array = None
            for camp in campaign_list:
                load_path = f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}/{sample_component}_{feature}.npy "
                temp_array = np.load(load_path)
                if feature_array is None:
                    feature_array = temp_array
                else:
                    feature_array = np.concatenate((feature_array, temp_array))
            # use float32 to reduce memory consumption, risky?
            feature_array = np.array(feature_array, dtype=np.float32)
            if pass_index is not None:
                sample_array_dict[feature] = feature_array[
                    pass_index
                ].flatten()
            else:
                sample_array_dict[feature] = feature_array.flatten()
        if "weight" in out_features:
            weight_array = sample_array_dict["weight"]
            total_weights = np.sum(weight_array)
            logger.debug(
                f"Load sample {sample_component} total weights: {total_weights}, total events {weight_array.shape}"
            )
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
