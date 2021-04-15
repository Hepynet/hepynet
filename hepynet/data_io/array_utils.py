import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hepynet.common import common_utils

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


def check_keys_has_sepa(output_keys):
    for key in output_keys:
        if key in SEPA_KEYS:
            return True
    return False


def check_keys_has_comb(output_keys):
    for key in output_keys:
        if key in COMB_KEYS:
            return True
    return False


def clip_negative_weights(weights: np.ndarray):
    """Sets negative weights to zero"""
    logger.debug("Clip negative weights...")
    weights[:] = np.clip(weights, a_min=0, a_max=None)

def extract_bkg_df(df: pd.DataFrame):
    return df.loc[(df["is_mc"] == True) & (df["is_sig"] == False), :]

def extract_sig_df(df: pd.DataFrame):
    return df.loc[df["is_sig"] == True, :]


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


def generate_shuffle_index(array_len):
    """Generates array shuffle index.

    To use a consist shuffle index to have different arrays shuffle in same way.

    """
    shuffle_index = np.array(range(array_len))
    np.random.shuffle(shuffle_index)
    return shuffle_index


# def merge_dict_to_inputs(array_dict, feature_list, array_key="all", sumofweight=1000):
#     """Merges contents in array_dict to plain arrays for DNN training"""
#     if array_key in list(array_dict.keys()):
#         array_out = np.concatenate(
#             [array_dict[array_key][feature] for feature in feature_list], axis=1,
#         )
#         weight_out = array_dict[array_key]["weight"]
#     elif array_key == "all" or array_key == "all_norm":
#         array_components = []
#         weight_components = []
#         for temp_key in array_dict.keys():
#             temp_array = np.concatenate(
#                 [array_dict[temp_key][feature] for feature in feature_list], axis=1,
#             )
#             array_components.append(temp_array)
#             weight_component = array_dict[temp_key]["weight"]
#             if array_key == "all":
#                 weight_components.append(weight_component)
#             else:
#                 sumofweights = np.sum(weight_component)
#                 weight_components.append(weight_component / sumofweights)
#         array_out = np.concatenate(array_components)
#         weight_out = np.concatenate(weight_components)
#         normalize_weight(weight_out, norm=sumofweight)
#     return array_out, weight_out.reshape((-1,))


def merge_samples_df(
    df_dict: Dict[str, pd.DataFrame],
    features: List[str],
    array_key: str = "all",
) -> pd.DataFrame:
    ## always load "weight"
    #if "weight" not in features:
    #    features += ["weight"]
    df_out = pd.DataFrame(columns=features)
    if array_key in df_dict:
        df_out = df_out.append(df_dict[array_key][features], ignore_index=True)
    elif array_key == "all" or array_key == "all_norm":
        for sample_df in df_dict.values():
            df_out = df_out.append(sample_df[features], ignore_index=True)
    else:
        logger.error(f"Invalid array_key {array_key}")
    return df_out


'''
def modify_array(
    input_array,
    input_weights,
    remove_negative_weight=False,
    reset_mass=False,
    reset_mass_array=None,
    reset_mass_weights=None,
    reset_mass_id=None,
    norm=False,
    sumofweight=1000,
    shuffle=False,
):
    """Modifies numpy array with given setup.

    Args:
        input_array: numpy array
            Array to be modified.
        remove_negative_weight: bool, optional (default=False) 
            Whether to remove events with negative weight.
        reset_mass: bool, optional (default=None)
            Whether to reset mass with given array's value distribution.
            If set True, reset_mass_array/reset_mass_id shouldn't be None.
        reset_mass_array: numpy array or none, optional (default=None):
            Array used to reset input_array's mass distribution.
        reset_mass_id: int or None, optional (default=None)
            Column index of mass id to reset input_array.
        norm: bool, optional (default=False)
            Whether normalize array's weight to sumofweight.
        sumofweight: float or None, optional (default=None)
            Total normalized weight.
        shuffle: bool, optional (default=None)
            Whether to randomize the output array.

      Returns:
          new: numpy array
              Modified numpy array.

  """
    # Modify
    new_array = input_array.copy()  # copy data to avoid original data operation
    new_weight = input_weights.copy()
    if len(new_array) == 0:
        logger.warning("empty input detected in modify_array, no changes will be made.")
        return new_array
    # clean array
    if remove_negative_weight:
        clip_negative_weights(new_weight)
    # reset mass
    if reset_mass == True:
        if not common_utils.has_none([reset_mass_array, reset_mass_id]):
            new_array = reset_col(
                new_array, reset_mass_array, reset_mass_weights, col=reset_mass_id
            )
        else:
            logger.warning("missing parameters, skipping mass reset...")
    # normalize weight
    if norm == True:
        normalize_weight(new_weight, norm=sumofweight)
    # shuffle array
    if shuffle == True:
        new_array, _, _, _, new_weight, _ = shuffle_and_split(
            new_array, np.zeros(len(new_array)), new_weight, split_ratio=0.0,
        )
    # return result
    return new_array, new_weight


def normalize_weight(weight_array, norm=1000):
    """Normalize given weight array to certain value

    Args:
        weight_array: numpy array
            Array to be normalized.
        norm: float (default=1000)
            Value to be normalized to.

    Returns:
        new: numpy array
          normalized array.

    Example:
      arr has weight value saved in column -1.
      To normalize it's total weight to 1:
        arr[:, -1] = normalize_weight(arr[:, -1], norm=1)

    """
    total_weight = sum(weight_array)
    frac = norm / total_weight
    weight_array[:] = frac * weight_array
    return weight_array
'''


def norm_weight(
    input_weights: pd.Series, norm_factor: Optional[float] = None,
):
    """Re-weights array"""
    if len(input_weights) == 0:
        logger.warning("empty input_weights, no changes will be made.")
        return
    # normalize weight
    if norm_factor is not None:
        input_weights.update(norm_factor * input_weights)


def redistribute_array(
    reset_array: pd.Series, ref_array: pd.Series, ref_weight: pd.Series
):
    # TODO: currently don't use negative weights, how to deal with negative
    # weights better?
    ref_weight_positive = ref_weight.copy()
    ref_weight_positive[ref_weight_positive < 0] = 0
    sump = ref_weight_positive.sum()

    reset_values = np.random.choice(
        ref_array.values,
        size=len(reset_array),
        p=(1 / sump) * ref_weight_positive.values,
    )
    reset_array.update(pd.Series(reset_values))


def reset_col(reset_array, ref_array, ref_weights, col=0):
    """Resets one column in an array based on the distribution of reference."""
    total_events = len(reset_array)
    positive_weights = (ref_weights.copy()).clip(
        min=0
    )  ## TODO: how to deal with negative weights better?
    if (positive_weights != ref_weights).all():
        logger.warning("Non-positive weights detected, set to zero")
    sump = sum(positive_weights)
    reset_list = np.random.choice(
        ref_array[:, col],
        size=total_events,
        p=(1 / sump) * positive_weights.reshape((-1,)),
    )
    reset_array[:, col] = reset_list
    return reset_array


def shuffle_and_split(x, y, wt, split_ratio=0.0):
    """Self defined function to replace train_test_split in sklearn to allow
    more flexibility.
    """
    # Check consistence of length of x, y
    if len(x) != len(y):
        raise ValueError("Length of x and y is not same.")
    array_len = len(y)
    # get index for the first part of the split array
    first_part_index = np.random.choice(
        range(array_len), int(array_len * 1.0 * split_ratio), replace=False
    )
    # get index for last part of the splitted array
    last_part_index = np.setdiff1d(np.array(range(array_len)), first_part_index)
    first_part_x = x[first_part_index]
    first_part_y = y[first_part_index]
    first_part_wt = wt[first_part_index]
    last_part_x = x[last_part_index]
    last_part_y = y[last_part_index]
    last_part_wt = wt[last_part_index]
    return (
        first_part_x,
        last_part_x,
        first_part_y,
        last_part_y,
        first_part_wt,
        last_part_wt,
    )


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
    has_sepa = check_keys_has_sepa(output_keys)
    has_comb = check_keys_has_comb(output_keys)
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
    ) = shuffle_and_split(xs, ys, xs_weight, split_ratio=1 - test_rate)
    (
        arr_sepa.xb_train,
        arr_sepa.xb_test,
        arr_sepa.yb_train,
        arr_sepa.yb_test,
        arr_sepa.wtb_train,
        arr_sepa.wtb_test,
    ) = shuffle_and_split(xb, yb, xb_weight, split_ratio=1 - test_rate)

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
            shuffle_index = generate_shuffle_index(len(arr_comb.y_train))
            arr_comb.x_train = arr_comb.x_train[shuffle_index]
            arr_comb.y_train = arr_comb.y_train[shuffle_index]
            arr_comb.wt_train = arr_comb.wt_train[shuffle_index]
            # shuffle test dataset
            shuffle_index = generate_shuffle_index(len(arr_comb.y_test))
            arr_comb.x_test = arr_comb.x_test[shuffle_index]
            arr_comb.y_test = arr_comb.y_test[shuffle_index]
            arr_comb.wt_test = arr_comb.wt_test[shuffle_index]

    out_arrays = {}
    for key in output_keys:
        if key in SEPA_KEYS:
            ## TODO: components in SEPA_KEYS are not necessary as they can be
            ## generated with y as tag
            out_arrays[key] = getattr(arr_sepa, key)
        elif key in COMB_KEYS:
            out_arrays[key] = getattr(arr_comb, key)
        else:
            logger.error(f"Unknown output_key: {key}")

    return out_arrays
