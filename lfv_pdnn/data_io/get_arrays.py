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
            cut_array = None
            if campaign in ["run2", "all"]:
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
                directory = npy_path + "/" + campaign
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
            ), "cut_features and cut_values and cut_types should have same length."
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
