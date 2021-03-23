# -*- coding: utf-8 -*-
import glob
import logging
import pathlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

from hepynet.common import config_utils
from hepynet.data_io import numpy_io

logger = logging.getLogger("hepynet")


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


def get_mean_var(array, axis=None, weights=None):
    """Calculate average and variance of an array."""
    average = np.average(array, axis=axis, weights=weights)
    variance = np.average((array - average) ** 2, axis=axis, weights=weights)
    if 0 in variance:
        logger.warn("Encountered 0 variance, adding shift value 0.000001")
    return average, variance + 0.000001


def get_train_val_indices(x, y, wt, val_split, k_folds=None):
    """Gets indices to separates train datasets to train/validation"""
    train_indices_list = list()
    validation_indices_list = list()
    if isinstance(k_folds, int) and k_folds >= 2:
        # skf = KFold(n_splits=k_folds, shuffle=True)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        for train_index, val_index in skf.split(x, y):
            train_indices_list.append(train_index)
            validation_indices_list.append(val_index)
    else:
        if isinstance(k_folds, int) and k_folds <= 2:
            logger.error(
                f"Invalid train.k_folds value {k_folds} detected, will not use k-fold validation"
            )
        array_len = len(wt)
        val_index = np.random.choice(
            range(array_len), int(array_len * 1.0 * val_split), replace=False
        )
        train_index = np.setdiff1d(np.array(range(array_len)), val_index)
        train_indices_list.append(train_index)
        validation_indices_list.append(val_index)
    return train_indices_list, validation_indices_list


def norm_array(array, average=None, variance=None):
    """Normalizes input array for each feature.

    Note:
        Do not normalize bkg and sig separately, bkg and sig should be normalized
        in the same way. (i.e. use same average and variance for normalization.)

    """
    if len(array) != 0:
        if (average is None) or (variance is None):
            logger.error("Unspecified average or variance.")
            return
        array[:] = (array - average) / np.sqrt(variance)


def norm_array_min_max(array, min, max, axis=None):
    """Normalizes input array to (-1, +1)"""
    middle = (min + max) / 2.0
    output_array = array.copy() - middle
    if max < min:
        logger.error("ERROR: max shouldn't be smaller than min.")
        return None
    ratio = (max - min) / 2.0
    output_array = output_array / ratio

