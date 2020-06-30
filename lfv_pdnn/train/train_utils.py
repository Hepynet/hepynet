# -*- coding: utf-8 -*-
"""Functions used for pDNN training.

This module is a collection of functions used for pDNN training. Include: array
manipulation, making plots, evaluation functions and so on.

"""

import glob
import os
import sys
import time
import warnings
from math import log, sqrt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, classification_report

from lfv_pdnn.common import array_utils


def calculate_asimov(sig, bkg):
    return sqrt(2 * ((sig + bkg) * log(1 + sig / bkg) - sig))


def calculate_significance(sig, bkg, sig_total=1, bkg_total=1, algo="asimov"):
    """Returns asimov significance"""
    # check input
    if sig <= 0 or bkg <= 0 or sig_total <= 0 or bkg_total <= 0:
        warnings.warn(
            "non-positive value found during significance calculation, using default value 0."
        )
        return 0
    if "_rel" in algo:
        if sig_total == 1 or bkg_total == 1:
            warnings.warn(
                "sig_total or bkg_total value is equal to default 1, please check input."
            )
    # calculation
    if algo == "asimov":
        return calculate_asimov(sig, bkg)
    elif algo == "s_b":
        return sig / bkg
    elif algo == "s_sqrt_b":
        return sig / sqrt(bkg)
    elif algo == "asimov_rel":
        return calculate_asimov(sig, bkg) / calculate_asimov(sig_total, bkg_total)
    elif algo == "s_b_rel":
        return (sig / sig_total) / (bkg / bkg_total)
    elif algo == "s_sqrt_b_rel":
        return (sig / sig_total) / sqrt(bkg / bkg_total)
    else:
        warnings.warn("Unrecognized significance algorithm, will use default 'asimov'")
        return calculate_asimov(sig, bkg)


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


def prepare_array(
    xs_input,
    xb_input,
    selected_features,
    apply_data=False,
    xd_input=None,
    reshape_array=True,
    reset_mass=False,
    reset_mass_name=None,
    remove_negative_weight=False,
    cut_features=[],
    cut_values=[],
    cut_types=[],
    sig_weight=1000,
    bkg_weight=1000,
    data_weight=1000,
    test_rate=0.2,
    rdm_seed=None,
    model_meta=None,
    verbose=1,
):
    """Prepares array for training."""
    feed_box = {}
    xs = xs_input.copy()
    xb = xb_input.copy()
    if apply_data:
        xd = xd_input.copy()
    # cut array
    if len(cut_features) > 0:
        # Get indexes that pass cuts
        assert len(cut_features) == len(cut_values) and len(cut_features) == len(
            cut_types
        ), "cut_features and cut_values and cut_types should have same length."
        xs_pass_index = None
        xb_pass_index = None
        for (cut_feature, cut_value, cut_type) in zip(
            cut_features, cut_values, cut_types
        ):
            cut_feature_id = selected_features.index(cut_feature)
            # update signal cut index
            temp_index = array_utils.get_cut_index_value(
                xs[:, cut_feature_id], cut_value, cut_type
            )
            if xs_pass_index is None:
                xs_pass_index = temp_index
            else:
                xs_pass_index = np.intersect1d(xs_pass_index, temp_index)
            # update background cut index
            temp_index = array_utils.get_cut_index_value(
                xb[:, cut_feature_id], cut_value, cut_type
            )
            if xb_pass_index is None:
                xb_pass_index = temp_index
            else:
                xb_pass_index = np.intersect1d(xb_pass_index, temp_index)
        xs = xs[xs_pass_index.flatten(), :]
        xb = xb[xb_pass_index.flatten(), :]
    # record raw array
    feed_box["xs_raw"] = xs
    feed_box["xb_raw"] = xb
    # normalize input variables if norm_array is True
    xs_reshape = xs.copy()
    xb_reshape = xb.copy()
    # remove negative weights for variance calculation
    if remove_negative_weight:
        xb_pos_weihgt = array_utils.modify_array(xb, remove_negative_weight=True,)
    # reshape
    if reshape_array:
        if model_meta is not None:
            means = np.array(model_meta["norm_average"])
            variances = np.array(model_meta["norm_variance"])
        else:
            means, variances = get_mean_var(
                xb_pos_weihgt[:, 0:-2], axis=0, weights=xb_pos_weihgt[:, -1]
            )
        feed_box["norm_average"] = means
        feed_box["norm_variance"] = variances

        xs_reshape[:, 0:-2] = norarray(
            xs_reshape[:, 0:-2], average=means, variance=variances
        )
        xb_reshape[:, 0:-2] = norarray(
            xb_reshape[:, 0:-2], average=means, variance=variances
        )
        feed_box["xs_reshape"] = xs_reshape
        feed_box["xb_reshape"] = xb_reshape
    else:
        feed_box["norm_average"] = np.zeros(len(selected_features))
        feed_box["norm_variance"] = np.ones(len(selected_features))
        feed_box["xs_reshape"] = xs_reshape.copy()
        feed_box["xb_reshape"] = xb_reshape.copy()
    if rdm_seed is None:
        rdm_seed = int(time.time())
        feed_box["rdm_seed"] = rdm_seed
    # get bkg array with mass reset
    if reset_mass:
        reset_mass_id = selected_features.index(reset_mass_name)
        xb_reset_mass = array_utils.modify_array(
            xb_reshape,
            reset_mass=reset_mass,
            reset_mass_array=xs_reshape,
            reset_mass_id=reset_mass_id,
        )
        feed_box["xb_reset_mass"] = xb_reset_mass
        feed_box["is_mass_reset"] = True
    else:
        xb_reset_mass = xb_reshape
        feed_box["xb_reset_mass"] = xb_reshape
        feed_box["is_mass_reset"] = False
    # remove negative weights & normalize total weight
    xs_reweight = array_utils.modify_array(
        xs_reshape,
        remove_negative_weight=remove_negative_weight,
        norm=True,
        sumofweight=sig_weight,
    )
    xb_reweight = array_utils.modify_array(
        xb_reshape,
        remove_negative_weight=remove_negative_weight,
        norm=True,
        sumofweight=bkg_weight,
    )
    xb_reweight_reset_mass = array_utils.modify_array(
        xb_reset_mass,
        remove_negative_weight=remove_negative_weight,
        norm=True,
        sumofweight=bkg_weight,
    )
    feed_box["xs_reweight"] = xs_reweight
    feed_box["xb_reweight"] = xb_reweight
    feed_box["xb_reweight_reset_mass"] = xb_reweight_reset_mass
    # get train/test data set, split with ratio=test_rate
    (
        x_train,
        x_test,
        y_train,
        y_test,
        xs_train,
        xs_test,
        xb_train,
        xb_test,
    ) = split_and_combine(
        xs_reweight, xb_reweight_reset_mass, test_rate=test_rate, shuffle_seed=rdm_seed
    )
    feed_box["x_train"] = x_train
    feed_box["x_test"] = x_test
    feed_box["y_train"] = y_train
    feed_box["y_test"] = y_test
    feed_box["xs_train"] = xs_train
    feed_box["xs_test"] = xs_test
    feed_box["xb_train"] = xb_train
    feed_box["xb_test"] = xb_test
    (
        x_train_original_mass,
        x_test_original_mass,
        y_train_original_mass,
        y_test_original_mass,
        xs_train_original_mass,
        xs_test_original_mass,
        xb_train_original_mass,
        xb_test_original_mass,
    ) = split_and_combine(
        xs_reweight, xb_reweight, test_rate=test_rate, shuffle_seed=rdm_seed
    )
    feed_box["x_train_original_mass"] = x_train_original_mass
    feed_box["x_test_original_mass"] = x_test_original_mass
    feed_box["y_train_original_mass"] = y_train_original_mass
    feed_box["y_test_original_mass"] = y_test_original_mass
    feed_box["xs_train_original_mass"] = xs_train_original_mass
    feed_box["xs_test_original_mass"] = xs_test_original_mass
    feed_box["xb_train_original_mass"] = xb_train_original_mass
    feed_box["xb_test_original_mass"] = xb_test_original_mass
    # select features used for training
    feed_box["selected_features"] = selected_features
    feed_box["x_train_selected"] = get_valid_feature(x_train)
    feed_box["x_test_selected"] = get_valid_feature(x_test)
    feed_box["xs_train_selected"] = get_valid_feature(xs_train)
    feed_box["xb_train_selected"] = get_valid_feature(xb_train)
    feed_box["xs_test_selected"] = get_valid_feature(xs_test)
    feed_box["xb_test_selected"] = get_valid_feature(xb_test)
    feed_box["xs_selected"] = get_valid_feature(xs_reweight)
    feed_box["xb_selected"] = get_valid_feature(xb_reweight_reset_mass)
    feed_box["x_train_selected_original_mass"] = get_valid_feature(
        x_train_original_mass
    )
    feed_box["x_test_selected_original_mass"] = get_valid_feature(x_test_original_mass)
    feed_box["xs_train_selected_original_mass"] = get_valid_feature(
        xs_train_original_mass
    )
    feed_box["xb_train_selected_original_mass"] = get_valid_feature(
        xb_train_original_mass
    )
    feed_box["xs_test_selected_original_mass"] = get_valid_feature(
        xs_test_original_mass
    )
    feed_box["xb_test_selected_original_mass"] = get_valid_feature(
        xb_test_original_mass
    )
    feed_box["xs_selected_original_mass"] = get_valid_feature(xs_reweight)
    feed_box["xb_selected_original_mass"] = get_valid_feature(xb_reweight)
    feed_box["xs_reshape_selected"] = get_valid_feature(xs_reshape)
    feed_box["xb_reshape_selected"] = get_valid_feature(xb_reshape)
    # prepare data to apply model when apply_data is True
    if apply_data == True:
        if len(cut_features) > 0:
            # Get indexes that pass cuts
            assert len(cut_features) == len(cut_values) and len(cut_features) == len(
                cut_types
            ), "cut_features and cut_values and cut_types should have same length."
            xd_pass_index = None
            for (cut_feature, cut_value, cut_type) in zip(
                cut_features, cut_values, cut_types
            ):
                cut_feature_id = selected_features.index(cut_feature)
                # update data cut index
                temp_index = array_utils.get_cut_index_value(
                    xd[:, cut_feature_id], cut_value, cut_type
                )
                if xd_pass_index is None:
                    xd_pass_index = temp_index
                else:
                    xd_pass_index = np.intersect1d(xd_pass_index, temp_index)
            xd = xd[xd_pass_index.flatten(), :]
        # record raw array
        feed_box["xd_raw"] = xd
        # reshape
        xd_reshape = xd.copy()
        if reshape_array:
            xd_reshape[:, 0:-2] = norarray(
                xd_reshape[:, 0:-2],
                average=feed_box["norm_average"],
                variance=feed_box["norm_variance"],
            )
            feed_box["xd_reshape"] = xd_reshape.copy()
        else:
            feed_box["xd_reshape"] = xd_reshape.copy()
        if reset_mass:
            reset_mass_id = selected_features.index(reset_mass_name)
            xd_reset_mass = array_utils.modify_array(
                xd_reshape,
                reset_mass=reset_mass,
                reset_mass_array=xs_reshape,
                reset_mass_id=reset_mass_id,
            )
            feed_box["xd_reset_mass"] = xd_reset_mass
        else:
            xd_reset_mass = xd_reshape
            feed_box["xd_reset_mass"] = xd_reset_mass
        # normalize total weight
        xd_reweight = array_utils.modify_array(
            xd_reshape, norm=True, sumofweight=data_weight
        )
        feed_box["xd_reweight"] = xd_reweight
        xd_reweight_reset_mass = array_utils.modify_array(
            xd_reset_mass, norm=True, sumofweight=data_weight
        )
        feed_box["xd_reweight_reset_mass"] = xd_reweight_reset_mass
        # get selected features
        feed_box["xd_selected"] = get_valid_feature(xd_reweight_reset_mass)
        feed_box["xd_selected_original_mass"] = get_valid_feature(xd_reweight)
        feed_box["xd_reshape_selected"] = get_valid_feature(xd_reshape)
        feed_box["has_data"] = True
    feed_box["array_prepared"] = True
    if verbose == 1:
        print("Training array prepared.")
        print("> signal shape:", feed_box["xs_selected"].shape)
        print("> background shape:", feed_box["xb_selected"].shape)
    return feed_box


def split_and_combine(
    xs, xb, test_rate=0.2, shuffle_combined_array=True, shuffle_seed=None
):
    """Prepares array for training & validation

    Args:
        xs: numpy array
        Siganl array for training.
        xb: numpy array
        Background array for training.
        test_rate: float, optional (default = 0.2)
        Portion of samples (array rows) to be used as independant test samples.
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
        Array for scores plottind.
        Signal/bakcground separated.

    """
    ys = np.ones(len(xs))
    yb = np.zeros(len(xb))

    xs_train, xs_test, ys_train, ys_test = array_utils.shuffle_and_split(
        xs, ys, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )
    xb_train, xb_test, yb_train, yb_test = array_utils.shuffle_and_split(
        xb, yb, split_ratio=1 - test_rate, shuffle_seed=shuffle_seed
    )

    x_train = np.concatenate((xs_train, xb_train))
    y_train = np.concatenate((ys_train, yb_train))
    x_test = np.concatenate((xs_test, xb_test))
    y_test = np.concatenate((ys_test, yb_test))

    if shuffle_combined_array:
        # shuffle train dataset
        shuffle_index = generate_shuffle_index(len(y_train), shuffle_seed=shuffle_seed)
        x_train = x_train[shuffle_index]
        y_train = y_train[shuffle_index]
        # shuffle test dataset
        shuffle_index = generate_shuffle_index(len(y_test), shuffle_seed=shuffle_seed)
        x_test = x_test[shuffle_index]
        y_test = y_test[shuffle_index]

    return x_train, x_test, y_train, y_test, xs_train, xs_test, xb_train, xb_test
