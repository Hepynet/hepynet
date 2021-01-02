import logging
import time

import numpy as np

from hepynet.common import common_utils

logger = logging.getLogger("hepynet")


def clean_negative_weights(weights):
    """removes elements with negative weight.

    Args:
        array: numpy array, input array to be processed, must be numpy array
        weight_id: int, indicate which column is weight value.
        verbose: bool, optional, show more detailed message if set True.

    Returns:
        cleaned new numpy array.
    """
    # Start
    logger.debug("Cleaning array elements with negative weights...")
    new_weights = weights.copy()
    new_weights = np.clip(new_weights, a_min=0, a_max=None)
    logger.debug(
        f"Shape before clean negative: {weights.shape}, shape after: {new_weights.shape}"
    )
    return new_weights


def get_cut_index(array, cut_values, cut_types):
    """Parses cuts arguments and returns cuts indexes."""
    assert len(cut_values) == len(
        cut_types
    ), "cut_values and cut_types should have same length."
    pass_index = None
    for cut_value, cut_type in zip(cut_values, cut_types):
        temp_index = get_cut_index_value(array, cut_value, cut_type)
        if pass_index is None:
            pass_index = temp_index
        else:
            pass_index = np.intersect1d(pass_index, temp_index)
    return pass_index


def get_cut_index_value(array, cut_value, cut_type):
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
    logger.debug("Cutting input array")
    logger.debug(f"Input shape: {array.shape}")
    # Make cuts
    if cut_type == "=":
        pass_index = np.argwhere(array == cut_value)
    elif cut_type == ">":
        pass_index = np.argwhere(array > cut_value)
    elif cut_type == "<":
        pass_index = np.argwhere(array < cut_value)
    else:
        raise ValueError("Invalid cut_type specified.")
    return pass_index.flatten()


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
    shuffle_seed=None,
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
        shuffle_seed: int or None, optional (default=None)
            Seed for randomization process.
            Set to None to use current time as seed.
            Set to a specific value to get an unchanged shuffle result.

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
        new_weight = clean_negative_weights(new_weight)
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
        new_weight = norweight(new_weight, norm=sumofweight)
    # shuffle array
    if shuffle == True:
        new_array, _, _, _, new_weight, _ = shuffle_and_split(
            new_array,
            np.zeros(len(new_array)),
            new_weight,
            split_ratio=0.0,
            shuffle_seed=shuffle_seed,
        )
    # return result
    return new_array, new_weight


def norweight(weight_array, norm=1000):
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
        arr[:, -1] = norweight(arr[:, -1], norm=1)

    """
    new = weight_array.copy()  # copy data to avoid original data operation
    total_weight = sum(new)
    frac = norm / total_weight
    new = frac * new
    return new


def reset_col(reset_array, ref_array, ref_weights, col=0, shuffle_seed=None):
    """Resets one column in an array based on the distribution of reference."""
    if common_utils.has_none([shuffle_seed]):
        shuffle_seed = int(time.time())
    np.random.seed(shuffle_seed)
    new = reset_array.copy()
    total_events = len(new)
    positive_weights = (ref_weights.copy()).clip(min=0)
    if (positive_weights != ref_weights).all():
        logger.warning("Non-positive weights detected, set to zero")
    sump = sum(positive_weights)
    reset_list = np.random.choice(
        ref_array[:, col],
        size=total_events,
        p=(1 / sump) * positive_weights.reshape((-1,)),
    )
    new[:, col] = reset_list
    return new


def shuffle_and_split(x, y, wt, split_ratio=0.0, shuffle_seed=None):
    """Self defined function to replace train_test_split in sklearn to allow
    more flexibility.
    """
    # Check consistence of length of x, y
    if len(x) != len(y):
        raise ValueError("Length of x and y is not same.")
    array_len = len(y)
    np.random.seed(shuffle_seed)
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

