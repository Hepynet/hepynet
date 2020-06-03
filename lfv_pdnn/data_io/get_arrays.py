"""Loads arrays for training.

Note:
  1. Arrays organized by dictionary, each key is corresponding to on type of
  signal or background. For example: "top", "rpv_1000"
  2. Special key names that can be used for training:
    * all: all sig/bkg concatenated directly
    * all_norm: (sig only) with each mass point array's weight normed and then
    concatenated.

"""

import numpy as np

from lfv_pdnn.common import array_utils


def get_bkg(
    npy_path,
    campaign,
    channel,
    bkg_list,
    selected_features,
    cut_features=[],
    cut_values=[],
    cut_types=[],
):
    print("Loading raw background array.")
    # Load individual bkg
    bkg_dict = get_npy_individuals(
        npy_path,
        campaign,
        channel,
        bkg_list,
        selected_features,
        "bkg",
        cut_features=cut_features,
        cut_values=cut_values,
        cut_types=cut_types,
    )
    # Add all bkg together
    bkg_all_array = np.concatenate(list(bkg_dict.values()))
    bkg_dict["all"] = bkg_all_array
    # Add all bkg together with each mass point normalized
    bkg_all_array_norm = None
    for bkg in bkg_list:
        temp_array = bkg_dict[bkg]
        temp_array = array_utils.modify_array(temp_array, norm=True)
        if bkg_all_array_norm is None:
            bkg_all_array_norm = temp_array
        else:
            bkg_all_array_norm = np.concatenate((bkg_all_array_norm, temp_array))
    bkg_dict["all_norm"] = bkg_all_array_norm
    return bkg_dict


def get_data(
    npy_path,
    campaign,
    channel,
    data_list,
    selected_features,
    cut_features=[],
    cut_values=[],
    cut_types=[],
):
    print("Loading raw data array.")
    # Load data, only one tag "all"
    data_dict = get_npy_individuals(
        npy_path,
        campaign,
        channel,
        data_list,
        selected_features,
        "data",
        cut_features=cut_features,
        cut_values=cut_values,
        cut_types=cut_types,
    )
    return data_dict


def get_sig(
    npy_path,
    campaign,
    channel,
    sig_list,
    selected_features,
    cut_features=[],
    cut_values=[],
    cut_types=[],
):
    print("Loading raw signal array.")
    # Load individual sig
    sig_dict = get_npy_individuals(
        npy_path,
        campaign,
        channel,
        sig_list,
        selected_features,
        "sig",
        cut_features=cut_features,
        cut_values=cut_values,
        cut_types=cut_types,
    )
    # Add all sig together
    sig_all_array = np.concatenate(list(sig_dict.values()))
    sig_dict["all"] = sig_all_array
    # Add all sig together with each mass point normalized
    sig_all_array_norm = None
    for sig in sig_list:
        temp_array = sig_dict[sig]
        temp_array = array_utils.modify_array(temp_array, norm=True)
        if sig_all_array_norm is None:
            sig_all_array_norm = temp_array
        else:
            sig_all_array_norm = np.concatenate((sig_all_array_norm, temp_array))
    sig_dict["all_norm"] = sig_all_array_norm
    # For test
    # TODO need to delete/change later
    sig_all_array_test = None
    for i, sig in enumerate(sig_list):
        current_weight = 2000 - 150 * i
        temp_array = sig_dict[sig]
        temp_array = array_utils.modify_array(
            temp_array, norm=True, sumofweight=current_weight
        )
        if sig_all_array_test is None:
            sig_all_array_test = temp_array
        else:
            sig_all_array_test = np.concatenate((sig_all_array_test, temp_array))
    sig_dict["all_test"] = sig_all_array_test

    return sig_dict


def get_npy_individuals(
    npy_path,
    campaign,
    channel,
    npy_list,
    selected_features,
    npy_prefix,
    cut_features=[],
    cut_values=[],
    cut_types=[],
):
    """Gets individual npy arrays with given info.

  Return:
    A dict of individual npy arrays.
    Example: signal dict of rpv_500 rpv_1000 rpv_2000

  """
    return_dict = {}
    # Load individual npy array
    added_weights = 0
    for npy in npy_list:
        if campaign in ["run2", "all"]:
            npy_array = None
            try:
                directory = npy_path
                for feature in selected_features + [channel, "weight"]:
                    temp_array1 = np.load(
                        directory
                        + "/mc16a"
                        + "/"
                        + npy_prefix
                        + "_"
                        + npy
                        + "_"
                        + feature
                        + ".npy"
                    )
                    temp_array1 = np.reshape(temp_array1, (-1, 1))
                    temp_array2 = np.load(
                        directory
                        + "/mc16d"
                        + "/"
                        + npy_prefix
                        + "_"
                        + npy
                        + "_"
                        + feature
                        + ".npy"
                    )
                    temp_array2 = np.reshape(temp_array2, (-1, 1))
                    temp_array3 = np.load(
                        directory
                        + "/mc16e"
                        + "/"
                        + npy_prefix
                        + "_"
                        + npy
                        + "_"
                        + feature
                        + ".npy"
                    )
                    temp_array3 = np.reshape(temp_array3, (-1, 1))
                    temp_array = np.concatenate(
                        (temp_array1, temp_array2, temp_array3), axis=0
                    )
                    if npy_array is None:
                        npy_array = temp_array
                    else:
                        npy_array = np.concatenate((npy_array, temp_array), axis=1)
            except:
                directory = npy_path + "/" + campaign
                for feature in selected_features + [channel, "weight"]:
                    temp_array = np.load(
                        directory
                        + "/"
                        + npy_prefix
                        + "_"
                        + npy
                        + "_"
                        + feature
                        + ".npy"
                    )
                    temp_array = np.reshape(temp_array, (-1, 1))
                    if npy_array is None:
                        npy_array = temp_array
                    else:
                        npy_array = np.concatenate((npy_array, temp_array), axis=1)
        else:
            directory = npy_path + "/" + campaign
            npy_array = None
            for feature in selected_features + [channel, "weight"]:
                temp_array = np.load(
                    directory + "/" + npy_prefix + "_" + npy + "_" + feature + ".npy"
                )
                temp_array = np.reshape(temp_array, (-1, 1))
                if npy_array is None:
                    npy_array = temp_array
                else:
                    npy_array = np.concatenate((npy_array, temp_array), axis=1)
        total_weights = np.sum(npy_array[:, -1])
        print(npy, "shape:", npy_array.shape, "total weights:", total_weights)

        if len(cut_features) > 0:
            if campaign in ["run2", "all"]:
                cut_array = None
                try:
                    directory = npy_path
                    for feature in cut_features:
                        temp_array1 = np.load(
                            directory
                            + "/mc16a"
                            + "/"
                            + npy_prefix
                            + "_"
                            + npy
                            + "_"
                            + feature
                            + ".npy"
                        )
                        temp_array1 = np.reshape(temp_array1, (-1, 1))
                        temp_array2 = np.load(
                            directory
                            + "/mc16d"
                            + "/"
                            + npy_prefix
                            + "_"
                            + npy
                            + "_"
                            + feature
                            + ".npy"
                        )
                        temp_array2 = np.reshape(temp_array2, (-1, 1))
                        temp_array3 = np.load(
                            directory
                            + "/mc16e"
                            + "/"
                            + npy_prefix
                            + "_"
                            + npy
                            + "_"
                            + feature
                            + ".npy"
                        )
                        temp_array3 = np.reshape(temp_array3, (-1, 1))
                        temp_array = np.concatenate(
                            (temp_array1, temp_array2, temp_array3), axis=0
                        )
                        if cut_array is None:
                            cut_array = temp_array
                        else:
                            cut_array = np.concatenate((cut_array, temp_array), axis=1)
                except:
                    directory = npy_path + "/" + campaign
                    for feature in cut_features:
                        temp_array = np.load(
                            directory
                            + "/"
                            + npy_prefix
                            + "_"
                            + npy
                            + "_"
                            + feature
                            + ".npy"
                        )
                        temp_array = np.reshape(temp_array, (-1, 1))
                        if cut_array is None:
                            cut_array = temp_array
                        else:
                            cut_array = np.concatenate((cut_array, temp_array), axis=1)
                else:
                    raise ValueError("Invalid campaign name.")
            else:
                directory = cut_array + "/" + campaign
                cut_array = None
                for feature in cut_features:
                    temp_array = np.load(
                        directory
                        + "/"
                        + npy_prefix
                        + "_"
                        + npy
                        + "_"
                        + feature
                        + ".npy"
                    )
                    temp_array = np.reshape(temp_array, (-1, 1))
                    if cut_array is None:
                        cut_array = temp_array
                    else:
                        cut_array = np.concatenate((cut_array, temp_array), axis=1)
            # Get indexes that pass cuts
            assert len(cut_features) == len(cut_values) and len(cut_features) == len(
                cut_types
            ), "cut_features and cut_values and cut_types should have same lenth."
            pass_index = None
            for cut_feature_id, (cut_value, cut_type) in enumerate(
                zip(cut_values, cut_types)
            ):
                temp_index = array_utils.get_cut_index_value(
                    cut_array[:, cut_feature_id], cut_value, cut_type
                )
                if pass_index is None:
                    pass_index = temp_index
                else:
                    pass_index = np.intersect1d(pass_index, temp_index)
            npy_array = npy_array[pass_index.flatten(), :]

            total_weights = np.sum(npy_array[:, -1])
            print(
                npy,
                "shape:",
                npy_array.shape,
                "total weights:",
                total_weights,
                "(after cuts)",
            )
        return_dict[npy] = npy_array
        added_weights += total_weights

    print("All {} weight together:".format(npy_prefix), added_weights)
    return return_dict
