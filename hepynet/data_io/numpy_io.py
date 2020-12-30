"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to a sample component
  2. Each sample component is a sub-dictionary, each key is corresponding to a input feature's array 

"""
import logging
import pathlib

import numpy as np

from hepynet.common import array_utils, common_utils, config_utils

logger = logging.getLogger("hepynet")


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
        A dict of subdict for different samples
            example: signal dict of rpv_500 rpv_1000 rpv_2000
        Each subdict is a dict of different variable
            examples: pT, eta, phi

    """

    # check different situation
    if type(samples) != list:
        sample_list = [samples]
    else:
        sample_list = samples
    if campaign in ["run2", "all"]:
        campaign_list = ["mc16a", "mc16d", "mc16e"]
    elif "+" in campaign:
        campaign_list = [camp.strip() for camp in campaign.split("+")]
    else:
        campaign_list = [campaign]
    # load arrays
    included_features = selected_features[:]
    if validation_features is None:
        validation_features = (
            []
        )  # TODO: need to solve this using a global default config setting in the future
    included_features = list(
        set().union(included_features, validation_features, ["weight"])
    )
    if (
        cut_features is None
    ):  # TODO: need to solve this using a global default config setting in the future
        cut_features = []
        cut_values = []
        cut_types = []
    cut_features += [channel]
    cut_values += [1]
    cut_types += ["="]
    out_dict = {}
    platform_meta = config_utils.load_current_platform_meta()
    data_directory = platform_meta["data_path"]
    if not data_directory:
        current_hostname = common_utils.get_current_hostname()
        current_platform = common_utils.get_current_platform_name()
        logger.critical(
            f"Can't find data_path setting for current host {current_hostname} with platform {current_platform}, please update the config at share/cross_platform/pc_meta.yaml"
        )
        raise KeyError
    for sample_component in sample_list:
        sample_array_dict = {}
        cut_array_dict = {}
        for feature in set().union(included_features, cut_features):
            feature_array = None
            if campaign in ["run2", "all"]:
                try:
                    for camp in campaign_list:
                        temp_array = np.load(
                            f"{data_directory}/{directory}/{camp}/{region}/{sample_component}_{feature}.npy"
                        )
                        temp_array = np.reshape(temp_array, (-1, 1))
                        if feature_array is None:
                            feature_array = temp_array
                        else:
                            feature_array = np.concatenate((feature_array, temp_array))
                except:
                    feature_array = np.load(
                        f"{data_directory}/{directory}/{campaign}/{region}/{sample_component}_{feature}.npy"
                    )
            # except:
            #    for campaign in campaign_list:
            #        temp_array = np.load(
            #            f"{data_directory}/{directory}/{campaign}/{region}/#{sample_component}_{feature}.npy"
            #        )
            #        temp_array = np.reshape(temp_array, (-1, 1))
            #        if feature_array is None:
            #            feature_array = temp_array
            #        else:
            #            feature_array = np.concatenate((feature_array, temp_array))
            else:
                feature_array = np.load(
                    f"{data_directory}/{directory}/{campaign}/{region}/{sample_component}_{feature}.npy"
                )
            feature_array = feature_array.reshape((-1, 1))

            if feature in included_features:
                sample_array_dict[feature] = feature_array
            if feature in cut_features:
                cut_array_dict[feature] = feature_array
        # apply cuts
        ## Get indexes that pass cuts
        if not (
            len(cut_features) == len(cut_values) and len(cut_features) == len(cut_types)
        ):
            logger.critical(
                "cut_features and cut_values and cut_types should have same length"
            )
        pass_index = None
        for cut_feature, cut_value, cut_type in zip(
            cut_features, cut_values, cut_types
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
