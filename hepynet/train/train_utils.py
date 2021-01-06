# -*- coding: utf-8 -*-
"""Functions used for pDNN training.

This module is a collection of functions used for pDNN training. Include: array
manipulation, making plots, evaluation functions and so on.

"""

import glob
import logging
import pathlib
import sys
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, classification_report

from hepynet.common import array_utils, config_utils
from hepynet.data_io import numpy_io

logger = logging.getLogger("hepynet")


SEPA_KEYS = [
    "xs_train",
    "xs_test",
    "ys_train",
    "ys_test",
    "wts_train",
    "wts_test",
    "xb_train",
    "xb_test",
    "yb_train",
    "yb_test",
    "wtb_train",
    "wtb_test",
]

COMB_KEYS = [
    "x_train",
    "x_test",
    "y_train",
    "y_test",
    "wt_train",
    "wt_test",
]


def dump_fit_npy(
    feedbox, keras_model, fit_ntup_branches, output_bkg_node_names, npy_dir="./"
):
    prefix_map = {"sig": "xs", "bkg": "xb"}
    if feedbox.get_job_config().input.apply_data:
        prefix_map["data"] = "xd"

    for map_key in list(prefix_map.keys()):
        sample_keys = getattr(feedbox.get_job_config().input, f"{map_key}_list")
        for sample_key in sample_keys:
            dump_branches = fit_ntup_branches + ["weight"]
            # prepare contents
            dump_array, dump_array_weight = feedbox.get_raw(
                prefix_map[map_key], array_key=sample_key, add_validation_features=True,
            )
            predict_input, _ = feedbox.get_reweight(
                prefix_map[map_key], array_key=sample_key, reset_mass=False
            )
            predictions = keras_model.predict(predict_input)
            # dump
            platform_meta = config_utils.load_current_platform_meta()
            data_path = platform_meta["data_path"]
            save_dir = f"{data_path}/{npy_dir}"
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            for branch in dump_branches:
                if branch == "weight":
                    branch_content = dump_array_weight
                else:
                    fd_val_features = feedbox.get_job_config().input.validation_features
                    if fd_val_features is None:
                        validation_features = []
                    else:
                        validation_features = fd_val_features
                    feature_list = (
                        feedbox.get_job_config().input.selected_features
                        + validation_features
                    )
                    branch_index = feature_list.index(branch)
                    branch_content = dump_array[:, branch_index]
                save_path = f"{save_dir}/{sample_key}_{branch}.npy"
                numpy_io.save_npy_array(branch_content, save_path)
            if len(output_bkg_node_names) == 0:
                save_path = f"{save_dir}/{sample_key}_dnn_out.npy"
                numpy_io.save_npy_array(predictions, save_path)
            else:
                for i, out_node in enumerate(["sig"] + output_bkg_node_names):
                    out_node = out_node.replace("+", "_")
                    save_path = f"{save_dir}/{sample_key}_dnn_out_{out_node}.npy"
                    numpy_io.save_npy_array(predictions[:, i], save_path)


def get_mass_range(mass_array, weights, nsig=1):
    """Gives a range of mean +- sigma

    Note:
        Only use for single peak distribution

    """
    average = np.average(mass_array, weights=weights)
    variance = np.average((mass_array - average) ** 2, weights=weights)
    lower_limit = average - np.sqrt(variance) * nsig
    upper_limit = average + np.sqrt(variance) * nsig
    return lower_limit, upper_limit


def get_model_epoch_path_list(
    load_dir, model_name, job_name="*", date="*", version="*"
):
    # Search possible files
    search_pattern = load_dir + "/" + date + "_" + job_name + "_" + version + "/models"
    model_dir_list = glob.glob(search_pattern)
    # Choose the newest one
    if len(model_dir_list) < 1:
        raise FileNotFoundError("Model file that matched the pattern not found.")
    model_dir = model_dir_list[-1]
    if len(model_dir_list) > 1:
        logger.warning(
            "More than one valid model file found, try to specify more infomation."
        )
        logger.info(f"Loading the last matched model path: {model_dir}")
    else:
        logger.info("Loading model at: {model_dir}")
    search_pattern = model_dir + "/" + model_name + "_epoch*.h5"
    model_path_list = glob.glob(search_pattern)
    return model_path_list


def generate_shuffle_index(array_len, shuffle_seed=None):
    """Generates array shuffle index.

    To use a consist shuffle index to have different arrays shuffle in same way.

    """
    shuffle_index = np.array(range(array_len))
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
    np.random.shuffle(shuffle_index)
    return shuffle_index


def get_mean_var(array, axis=None, weights=None):
    """Calculate average and variance of an array."""
    average = np.average(array, axis=axis, weights=weights)
    variance = np.average((array - average) ** 2, axis=axis, weights=weights)
    if 0 in variance:
        logger.warn("Encountered 0 variance, adding shift value 0.000001")
    return average, variance + 0.000001


def norarray(array, average=None, variance=None, axis=None, weights=None):
    """Normalizes input array for each feature.

    Note:
        Do not normalize bkg and sig separately, bkg and sig should be normalized
        in the same way. (i.e. use same average and variance for normalization.)

    """
    if len(array) == 0:
        return array
    else:
        if (average is None) or (variance is None):
            logger.warn("Unspecified average or variance.")
            average, variance = get_mean_var(array, axis=axis, weights=weights)
        output_array = (array.copy() - average) / np.sqrt(variance)
        return output_array


def norarray_min_max(array, min, max, axis=None):
    """Normalizes input array to (-1, +1)"""
    middle = (min + max) / 2.0
    output_array = array.copy() - middle
    if max < min:
        logger.error("ERROR: max shouldn't be smaller than min.")
        return None
    ratio = (max - min) / 2.0
    output_array = output_array / ratio


def split_and_combine(
    xs,
    xs_weight,
    xb,
    xb_weight,
    ys=None,
    yb=None,
    output_keys=None,
    test_rate=0.2,
    shuffle_combined_array=True,
    shuffle_seed=None,
):
    """Prepares array for training & validation

    Args:
        xs: numpy array
        Signal array for training.
        xb: numpy array
        Background array for training.
        test_rate: float, optional (default = 0.2)
        Portion of samples (array rows) to be used as independent test samples.
        shuffle_combined_array: bool, optional (default=True)
        Whether to shuffle outputs arrays before return.
        shuffle_seed: int or None, optional (default=None)
        Seed for randomization process.
        Set to None to use current time as seed.
        Set to a specific value to get an unchanged shuffle result.

    Returns:
        x_train/x_test/y_train/y_test: numpy array
        Array for training/testing.
        Contain mixed signal and background. 
        xs_test/xb_test: numpy array
        Array for scores plotting.
        Signal/background separated.

    """
    # prepare
    if output_keys is None:
        output_keys = COMB_KEYS + SEPA_KEYS
    has_sepa = output_keys_has_sepa(output_keys)
    has_comb = output_keys_has_comb(output_keys)
    arr_sepa = DNN_Arrays_Separate()
    arr_comb = DNN_Arrays_Combined()
    if ys is None:
        ys = np.ones(len(xs)).reshape(-1, 1)
    if yb is None:
        yb = np.zeros(len(xb)).reshape(-1, 1)

    (
        arr_sepa.xs_train,
        arr_sepa.xs_test,
        arr_sepa.ys_train,
        arr_sepa.ys_test,
        arr_sepa.wts_train,
        arr_sepa.wts_test,
    ) = array_utils.shuffle_and_split(
        xs, ys, xs_weight, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )
    (
        arr_sepa.xb_train,
        arr_sepa.xb_test,
        arr_sepa.yb_train,
        arr_sepa.yb_test,
        arr_sepa.wtb_train,
        arr_sepa.wtb_test,
    ) = array_utils.shuffle_and_split(
        xb, yb, xb_weight, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )

    if has_comb:
        arr_comb.x_train = np.concatenate((arr_sepa.xs_train, arr_sepa.xb_train))
        arr_comb.y_train = np.concatenate((arr_sepa.ys_train, arr_sepa.yb_train))
        arr_comb.wt_train = np.concatenate((arr_sepa.wts_train, arr_sepa.wtb_train))
        arr_comb.x_test = np.concatenate((arr_sepa.xs_test, arr_sepa.xb_test))
        arr_comb.y_test = np.concatenate((arr_sepa.ys_test, arr_sepa.yb_test))
        arr_comb.wt_test = np.concatenate((arr_sepa.wts_test, arr_sepa.wtb_test))
        if not has_sepa:
            arr_sepa = None
        if shuffle_combined_array:
            # shuffle train dataset
            shuffle_index = generate_shuffle_index(
                len(arr_comb.y_train), shuffle_seed=shuffle_seed
            )
            arr_comb.x_train = arr_comb.x_train[shuffle_index]
            arr_comb.y_train = arr_comb.y_train[shuffle_index]
            arr_comb.wt_train = arr_comb.wt_train[shuffle_index]
            # shuffle test dataset
            shuffle_index = generate_shuffle_index(
                len(arr_comb.y_test), shuffle_seed=shuffle_seed
            )
            arr_comb.x_test = arr_comb.x_test[shuffle_index]
            arr_comb.y_test = arr_comb.y_test[shuffle_index]
            arr_comb.wt_test = arr_comb.wt_test[shuffle_index]

    out_arrays = {}
    for key in output_keys:
        if key in SEPA_KEYS:
            out_arrays[key] = getattr(arr_sepa, key)
        elif key in COMB_KEYS:
            out_arrays[key] = getattr(arr_comb, key)
        else:
            logger.error(f"Unknown output_key: {key}")

    return out_arrays


def output_keys_has_sepa(output_keys):
    for key in output_keys:
        if key in SEPA_KEYS:
            return True
    return False


def output_keys_has_comb(output_keys):
    for key in output_keys:
        if key in COMB_KEYS:
            return True
    return False


class DNN_Arrays_Separate(object):
    def __init__(self):
        self.xs_train = None
        self.xs_test = None
        self.ys_train = None
        self.ys_test = None
        self.wts_train = None
        self.wts_test = None
        self.xb_train = None
        self.xb_test = None
        self.yb_train = None
        self.yb_test = None
        self.wtb_train = None
        self.wtb_test = None


class DNN_Arrays_Combined(object):
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.wt_train = None
        self.wt_test = None
