# -*- coding: utf-8 -*-
"""Functions used for pDNN training.

This module is a collection of functions used for pDNN training. Include: array
manipulation, making plots, evaluation functions and so on.

"""

import glob
import logging
import os
import sys
import time
from math import log, sqrt

import matplotlib.pyplot as plt
import numpy as np
from lfv_pdnn.common import array_utils
from lfv_pdnn.data_io import root_io
from sklearn.metrics import accuracy_score, auc, classification_report


def calculate_asimov(sig, bkg):
    return sqrt(2 * ((sig + bkg) * log(1 + sig / bkg) - sig))


def calculate_significance(sig, bkg, sig_total=None, bkg_total=None, algo="asimov"):
    """Returns asimov significance"""
    # check input
    if sig <= 0 or bkg <= 0 or sig_total <= 0 or bkg_total <= 0:
        logging.warn(
            "non-positive value found during significance calculation, using default value 0."
        )
        return 0
    if "_rel" in algo:
        if not sig_total:
            logging.error(
                "sig_total or bkg_total value is not specified to calculate relative type significance, please check input."
            )
        if not bkg_total:
            logging.error(
                "sig_total or bkg_total value is not specified to calculate relative type significance, please check input."
            )
    # calculation
    if algo == "asimov":
        return calculate_asimov(sig, bkg)
    elif algo == "s_b":
        return sig / bkg
    elif algo == "s_sqrt_b":
        return sig / sqrt(bkg)
    elif algo == "s_sqrt_sb":
        return sig / sqrt(sig + bkg)
    elif algo == "asimov_rel":
        return calculate_asimov(sig, bkg) / calculate_asimov(sig_total, bkg_total)
    elif algo == "s_b_rel":
        return (sig / sig_total) / (bkg / bkg_total)
    elif algo == "s_sqrt_b_rel":
        return (sig / sig_total) / sqrt(bkg / bkg_total)
    elif algo == "s_sqrt_sb_rel":
        return (sig / sig_total) / sqrt((bkg + sig) / (sig_total + bkg_total))
    else:
        logging.warn("Unrecognized significance algorithm, will use default 'asimov'")
        return calculate_asimov(sig, bkg)


def dump_fit_ntup(
    feedbox, keras_model, fit_ntup_branches, output_bkg_node_names, ntup_dir="./"
):
    prefix_map = {"sig": "xs", "bkg": "xb"}
    if feedbox.apply_data:
        prefix_map["data"] = "xd"
    for map_key in list(prefix_map.keys()):
        sample_keys = list(getattr(feedbox, prefix_map[map_key] + "_dict").keys())
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
            dump_contents = []
            for branch in dump_branches:
                if branch == "weight":
                    branch_content = dump_array_weight
                else:
                    feature_list = (
                        feedbox.selected_features + feedbox.validation_features
                    )
                    branch_index = feature_list.index(branch)
                    branch_content = dump_array[:, branch_index]
                dump_contents.append(branch_content)
            if len(output_bkg_node_names) == 0:
                dump_branches.append("dnn_out")
                dump_contents.append(predictions)
            else:
                for i, out_node in enumerate(["sig"] + output_bkg_node_names):
                    out_node = out_node.replace("+", "_")
                    dump_branches.append("dnn_out_" + out_node)
                    dump_contents.append(predictions[:, i])
            # dump
            ntup_path = ntup_dir + "/" + map_key + "_" + sample_key + ".root"
            root_io.dump_ntup_from_npy(
                "ntup", dump_branches, "f", dump_contents, ntup_path,
            )


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
        print("More than one valid model file found, try to specify more infomation.")
        print("Loading the last matched model path:", model_dir)
    else:
        print("Loading model at:", model_dir)
    search_pattern = model_dir + "/" + model_name + "_epoch*.h5"
    model_path_list = glob.glob(search_pattern)
    return model_path_list


def get_valid_feature(xtrain):
    """Gets valid inputs.

    Note:
        indice -2 is for channel
        indice -1 is for weight

    """
    xtrain = xtrain[:, :-2]
    return xtrain


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
        logging.warn("Encountered 0 variance, adding shift value 0.000001")
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
            print("Warning! unspecified average or variance.")
            average, variance = get_mean_var(array, axis=axis, weights=weights)
        output_array = (array.copy() - average) / np.sqrt(variance)
        return output_array


def norarray_min_max(array, min, max, axis=None):
    """Normalizes input array to (-1, +1)"""
    middle = (min + max) / 2.0
    output_array = array.copy() - middle
    if max < min:
        print("ERROR: max shouldn't be smaller than min.")
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
    if ys is None:
        ys = np.ones(len(xs)).reshape(-1, 1)
    if yb is None:
        yb = np.zeros(len(xb)).reshape(-1, 1)

    (
        xs_train,
        xs_test,
        ys_train,
        ys_test,
        xs_weight_train,
        xs_weight_test,
    ) = array_utils.shuffle_and_split(
        xs, ys, xs_weight, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )
    (
        xb_train,
        xb_test,
        yb_train,
        yb_test,
        xb_weight_train,
        xb_weight_test,
    ) = array_utils.shuffle_and_split(
        xb, yb, xb_weight, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )

    x_train = np.concatenate((xs_train, xb_train))
    y_train = np.concatenate((ys_train, yb_train))
    wt_train = np.concatenate((xs_weight_train, xb_weight_train))
    x_test = np.concatenate((xs_test, xb_test))
    y_test = np.concatenate((ys_test, yb_test))
    wt_test = np.concatenate((xs_weight_test, xb_weight_test))

    if shuffle_combined_array:
        # shuffle train dataset
        shuffle_index = generate_shuffle_index(len(y_train), shuffle_seed=shuffle_seed)
        x_train = x_train[shuffle_index]
        y_train = y_train[shuffle_index]
        wt_train = wt_train[shuffle_index]
        # shuffle test dataset
        shuffle_index = generate_shuffle_index(len(y_test), shuffle_seed=shuffle_seed)
        x_test = x_test[shuffle_index]
        y_test = y_test[shuffle_index]
        wt_test = wt_test[shuffle_index]

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        wt_train,
        wt_test,
        xs_train,
        xs_test,
        ys_train,
        ys_test,
        xs_weight_train,
        xs_weight_test,
        xb_train,
        xb_test,
        yb_train,
        yb_test,
        xb_weight_train,
        xb_weight_test,
    )

