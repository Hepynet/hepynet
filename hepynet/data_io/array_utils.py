import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("hepynet")


def cov(x, y, w):
    """Calculates weighted covariance"""
    return np.sum(
        w
        * (x - np.average(x, axis=0, weights=w))
        * (y - np.average(y, axis=0, weights=w))
    ) / np.sum(w)


def corr(x, y, w):
    """Calculates weighted correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def corr_matrix(x, w=None):
    """Calculates correlation coefficient matrix"""
    if w is None:
        w = np.ones(x.shape[0])
    num_features = x.shape[1]
    # np.cov can't deal with negative weights, use self-defined function for now
    corr_m = np.zeros((num_features, num_features))
    for row in range(num_features):  # TODO: need to optimize the algorithm
        for col in range(row + 1):
            corr_m[row][col] = corr(x[:, row], x[:, col], w)
            corr_m[col][row] = corr_m[row][col]
    return corr_m


def extract_bkg_df(df: pd.DataFrame):
    return df.loc[(df["is_mc"] == True) & (df["is_sig"] == False)]


def extract_sig_df(df: pd.DataFrame):
    return df.loc[df["is_sig"] == True]


def get_cut_index(np_array, cut_values, cut_types):
    """Parses cuts arguments and returns cuts indexes."""
    assert len(cut_values) == len(
        cut_types
    ), "cut_values and cut_types should have same length."
    pass_index = None
    for cut_value, cut_type in zip(cut_values, cut_types):
        temp_index = get_cut_index_value(np_array, cut_value, cut_type)
        if pass_index is None:
            pass_index = temp_index
        else:
            pass_index = np.intersect1d(pass_index, temp_index)
    return pass_index


def get_cut_index_value(np_array: np.ndarray, cut_value, cut_type):
    """Returns cut indexes based on cut_value and cut_type.

    If cut_type is "=":
        returns all indexes that have values equal to cut_value
    If cut_type is ">":
        returns all indexes that have values lager than cut_value
    If cut_type is "<":
        returns all indexes that have values smaller than cut_value

    Args:
        array_dict: numpy array
        cut_feature: str
        cut_bool: bool
    """
    logger.debug("@ data_io.array_utils.get_cut_index_value")
    assert np_array.ndim == 1
    logger.debug(f"Cut input shape: {np_array.shape}, cut value {cut_value}")
    # Make cuts
    if cut_type == "=":
        pass_index = np.argwhere(np_array == cut_value)
    elif cut_type == ">":
        pass_index = np.argwhere(np_array > cut_value)
    elif cut_type == "<":
        pass_index = np.argwhere(np_array < cut_value)
    else:
        raise ValueError("Invalid cut_type specified.")
    return pass_index.flatten()
