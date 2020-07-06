import time
import warnings
import copy
import numpy as np

from lfv_pdnn.common import array_utils
from lfv_pdnn.train import train_utils


class Feedbox(object):
    """DNN inputs management class.

    Args:
        object ([type]): [description]
    """

    def __init__(
        self,
        xs_dict,
        xb_dict,
        xd_dict=None,
        apply_data=False,
        selected_features=[],
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
        # basic array collection
        self.xs_dict = copy.deepcopy(xs_dict)
        self.xb_dict = copy.deepcopy(xb_dict)
        self.xd_dict = copy.deepcopy(xd_dict)
        # meta info
        self.apply_data = apply_data
        self.selected_features = selected_features
        self.reshape_array = reshape_array
        self.reset_mass = reset_mass
        self.reset_mass_name = reset_mass_name
        if self.reset_mass:
            self.reset_mass_id = self.selected_features.index(self.reset_mass_name)
        else:
            self.reset_mass_id = None
        self.remove_negative_weight = remove_negative_weight
        self.cut_features = cut_features
        self.cut_values = cut_values
        self.cut_types = cut_types
        self.sig_weight = sig_weight
        self.bkg_weight = bkg_weight
        self.data_weight = data_weight
        self.test_rate = test_rate
        self.rdm_seed = rdm_seed
        self.model_meta = model_meta
        self.verbose = verbose
        self.array_prepared = False
        # set random seed
        if rdm_seed is None:
            rdm_seed = int(time.time())
            self.rdm_seed = rdm_seed
        # cut array
        for xs_key in xs_dict.keys():
            self.xs_dict[xs_key] = cut_array(
                self.xs_dict[xs_key],
                selected_features,
                cut_features,
                cut_values,
                cut_types,
            )
        for xb_key in xb_dict.keys():
            self.xb_dict[xb_key] = cut_array(
                self.xb_dict[xb_key],
                selected_features,
                cut_features,
                cut_values,
                cut_types,
            )
        if apply_data:
            for xd_key in xd_dict.keys():
                self.xd_dict[xd_key] = cut_array(
                    self.xd_dict[xd_key],
                    selected_features,
                    cut_features,
                    cut_values,
                    cut_types,
                )

        # get normalization parameters
        no_norm_paras = False
        if model_meta is None:
            no_norm_paras = True
        else:
            if (model_meta["norm_average"] is None) or (
                model_meta["norm_variance"] is None
            ):
                no_norm_paras = True
            else:
                no_norm_paras = False
        if no_norm_paras:
            xb_input = array_utils.modify_array(
                np.concatenate(list(self.xb_dict.values())), select_channel=True
            )
            means, variances = train_utils.get_mean_var(
                xb_input[:, 0:-2], axis=0, weights=xb_input[:, -1]
            )
        else:
            means = np.array(model_meta["norm_average"])
            variances = np.array(model_meta["norm_variance"])

        self.norm_means = means
        self.norm_variances = variances
        self.array_prepared = True

    def get_array(self, input_type, variation, array_key="all", reset_mass=None):
        if reset_mass == None:
            reset_mass = self.reset_mass
        if variation == "raw":
            return self.get_raw(input_type, array_key=array_key)
        elif variation == "reshape":
            return self.get_reshape(input_type, array_key=array_key)
        elif variation == "reweight":
            return self.get_reweight(
                input_type, array_key=array_key, reset_mass=reset_mass
            )
        elif variation == "selected":
            return self.get_selected(
                input_type, array_key=array_key, reset_mass=reset_mass
            )
        else:
            warnings.warn("Unknown variation")
            return None

    def get_raw(self, input_type, array_key="all"):
        array_out = None
        # get dict
        array_dict = None
        if input_type == "xs":
            array_dict = self.xs_dict
        elif input_type == "xb":
            array_dict = self.xb_dict
        elif input_type == "xd":
            if self.apply_data:
                array_dict = self.xd_dict
            else:
                warnings.warn(
                    "Trying to get data array as apply_data option is set to False"
                )
                return None
        else:
            warnings.warn("Unknown input_type")
            return None
        # read array from dict
        if array_key in list(array_dict.keys()):
            array_out = array_dict[array_key]
        elif array_key == "all":
            array_out = np.concatenate(list(array_dict.values()))
        elif array_key == "all_norm":
            array_norm = None
            for temp_key in array_dict.keys():
                temp_array = array_dict[temp_key]
                if len(temp_array) != 0:
                    temp_array = array_utils.modify_array(temp_array, norm=True)
                    if array_norm is None:
                        array_norm = temp_array
                    else:
                        array_norm = np.concatenate((array_norm, temp_array))
            array_out = array_norm
        if self.remove_negative_weight:
            array_out = array_utils.modify_array(
                array_out, remove_negative_weight=True,
            )
        return array_out

    def get_reshape(self, input_type, array_key="all"):
        x_reshape = self.get_raw(input_type, array_key=array_key)
        x_reshape[:, 0:-2] = train_utils.norarray(
            x_reshape[:, 0:-2], average=self.norm_means, variance=self.norm_variances,
        )
        return x_reshape

    def get_reweight(self, input_type, array_key="all", reset_mass=None):
        if reset_mass == None:
            reset_mass = self.reset_mass
        if input_type == "xs":
            xs_reshape = self.get_reshape("xs", array_key=array_key)
            xs_reweight = array_utils.modify_array(
                xs_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=True,
                sumofweight=self.sig_weight,
            )
            return xs_reweight
        elif input_type == "xb":
            xb_reshape = self.get_reshape("xb", array_key=array_key)
            if reset_mass:
                xb_reshape = array_utils.modify_array(
                    xb_reshape,
                    reset_mass=True,
                    reset_mass_array=self.get_reshape("xs", array_key=array_key),
                    reset_mass_id=self.reset_mass_id,
                )
            xb_reweight = array_utils.modify_array(
                xb_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=True,
                sumofweight=self.bkg_weight,
            )
            return xb_reweight
        elif input_type == "xd":
            xd_reshape = self.get_reshape("xd", array_key=array_key)
            if reset_mass:
                xd_reshape = array_utils.modify_array(
                    xd_reshape,
                    reset_mass=True,
                    reset_mass_array=self.get_reshape("xs", array_key=array_key),
                    reset_mass_id=self.reset_mass_id,
                )
            xd_reweight = array_utils.modify_array(
                xd_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=True,
                sumofweight=self.data_weight,
            )
            return xd_reweight
        else:
            warnings.warn("Unknown input_type")
            return None

    def get_train_test_arrays(
        self, sig_key="all", bkg_key="all", reset_mass=None, use_selected=False
    ):
        if reset_mass == None:
            reset_mass = self.reset_mass
        xs_reweight = self.get_reweight("xs", array_key=sig_key, reset_mass=reset_mass)
        xb_reweight = self.get_reweight("xb", array_key=bkg_key, reset_mass=reset_mass)
        (
            x_train,
            x_test,
            y_train,
            y_test,
            xs_train,
            xs_test,
            xb_train,
            xb_test,
        ) = train_utils.split_and_combine(
            xs_reweight,
            xb_reweight,
            test_rate=self.test_rate,
            shuffle_seed=self.rdm_seed,
        )
        if use_selected:
            return (
                train_utils.get_valid_feature(x_train),
                train_utils.get_valid_feature(x_test),
                y_train,
                y_test,
                train_utils.get_valid_feature(xs_train),
                train_utils.get_valid_feature(xs_test),
                train_utils.get_valid_feature(xb_train),
                train_utils.get_valid_feature(xb_test),
            )
        else:
            return (
                x_train,
                x_test,
                y_train,
                y_test,
                xs_train,
                xs_test,
                xb_train,
                xb_test,
            )

    def get_selected(self, input_type, array_key="all", reset_mass=None):
        if reset_mass == None:
            reset_mass = self.reset_mass
        if input_type == "xs":
            xs_reweight = self.get_reweight(
                "xs", array_key=array_key, reset_mass=reset_mass
            )
            return train_utils.get_valid_feature(xs_reweight)
        elif input_type == "xb":
            xb_reweight = self.get_reweight(
                "xb", array_key=array_key, reset_mass=reset_mass
            )
            return train_utils.get_valid_feature(xb_reweight)
        elif input_type == "xd":
            if self.apply_data:
                xd_reweight = self.get_reweight(
                    "xd", array_key=array_key, reset_mass=reset_mass
                )
                return train_utils.get_valid_feature(xd_reweight)
            else:
                return None
        else:
            warnings.warn("Unknown input_type")
            return None


def cut_array(
    input_array, selected_features, cut_features=[], cut_values=[], cut_types=[]
):
    # cut array
    if len(cut_features) > 0:
        # Get indexes that pass cuts
        assert len(cut_features) == len(cut_values) and len(cut_features) == len(
            cut_types
        ), "cut_features and cut_values and cut_types should have same length."
        pass_index_array = None
        for (cut_feature, cut_value, cut_type) in zip(
            cut_features, cut_values, cut_types
        ):
            cut_feature_id = selected_features.index(cut_feature)
            # update cut index
            temp_index = array_utils.get_cut_index_value(
                input_array[:, cut_feature_id], cut_value, cut_type
            )
            if pass_index_array is None:
                pass_index_array = temp_index
            else:
                pass_index_array = np.intersect1d(pass_index_array, temp_index)
        return input_array[pass_index_array.flatten(), :]
    else:
        return input_array.copy()
