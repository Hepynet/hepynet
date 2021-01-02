import copy
import logging
import time
from sys import getsizeof

import numpy as np

from hepynet.common import array_utils
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


class Feedbox(object):
    """DNN inputs management class."""

    def __init__(
        self, xs_dict, xb_dict, xd_dict, job_config, model_meta=None,
    ):
        # get config
        ic = job_config.input.clone()
        tc = job_config.train.clone()
        ac = job_config.apply.clone()
        # basic array collection
        self.xs_dict = copy.deepcopy(xs_dict)
        self.xb_dict = copy.deepcopy(xb_dict)
        self.xd_dict = copy.deepcopy(xd_dict)
        # meta info
        self.apply_data = ac.apply_data
        self.selected_features = ic.selected_features
        self.validation_features = ic.validation_features
        self.reset_mass = ic.reset_feature
        if self.reset_mass:
            if ic.reset_feature_name in self.selected_features:
                self.reset_mass_id = self.selected_features.index(ic.reset_feature_name)
            else:
                logger.critical(
                    f"input.reset_feature is True but input.reset_feature_name {ic.reset_feature_name} is invalid, please check the config"
                )
                exit(1)
        else:
            self.reset_mass_id = None
        self.remove_negative_weight = ic.rm_negative_weight_events
        self.sig_weight = ic.sig_sumofweight
        self.bkg_weight = ic.bkg_sumofweight
        self.data_weight = ic.data_sumofweight
        self.test_rate = tc.test_rate
        self.rdm_seed = ic.rdm_seed
        self._array_prepared = False
        # set random seed
        if self.rdm_seed is None:
            self.rdm_seed = int(time.time())

        # get normalization parameters
        no_norm_paras = False
        if model_meta is None:
            no_norm_paras = True
        else:
            if model_meta["norm_dict"] is None:
                no_norm_paras = True
            else:
                no_norm_paras = False
        norm_dict = {}
        if no_norm_paras:
            weight_array = np.concatenate(
                [sample_dict["weight"] for sample_dict in self.xb_dict.values()]
            )
            for feature in self.selected_features:
                feature_array = np.concatenate(
                    [sample_dict[feature] for sample_dict in self.xb_dict.values()]
                )
                mean = np.average(feature_array, weights=weight_array)
                variance = np.average((feature_array - mean) ** 2, weights=weight_array)
                norm_dict[feature] = {"mean": mean, "variance": variance}
                logger.debug(f"Feature {feature} mean: {mean}, variance: {variance}")
        else:
            norm_dict = model_meta["norm_dict"]

        self.norm_dict = norm_dict
        self._array_prepared = True

    def get_raw(self, input_type, array_key="all", add_validation_features=False):
        array_out = None
        weight_out = None
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
                logger.warn(
                    "Trying to get data array as apply_data option is set to False"
                )
                return None
        else:
            logger.warn("Unknown input type")
            return None
        if add_validation_features:
            if self.validation_features is None:
                validation_features = []
            else:
                validation_features = self.validation_features
            feature_list = self.selected_features + validation_features
        else:
            feature_list = self.selected_features
        if array_key in list(array_dict.keys()):
            array_out = np.concatenate(
                [array_dict[array_key][feature] for feature in feature_list], axis=1,
            )
            weight_out = array_dict[array_key]["weight"]
        elif array_key == "all" or array_key == "all_norm":
            array_components = []
            weight_components = []
            for temp_key in array_dict.keys():
                temp_array = np.concatenate(
                    [array_dict[temp_key][feature] for feature in feature_list], axis=1,
                )
                array_components.append(temp_array)
                weight_component = array_dict[temp_key]["weight"]
                if array_key == "all":
                    weight_components.append(weight_component)
                else:
                    sumofweights = np.sum(weight_component)
                    weight_components.append(weight_component / sumofweights)
            array_out = np.concatenate(array_components)
            weight_out = np.concatenate(weight_components)
        else:
            logger.warn("Unknown array_key")
            return None
        if self.remove_negative_weight:
            weight_out = array_utils.clean_negative_weights(weight_out)
        return array_out, weight_out.reshape((-1,))

    def get_reshape(self, input_type, array_key="all"):
        x_reshape, weight_reshape = self.get_raw(input_type, array_key=array_key)
        norm_means = []
        norm_variances = []
        for feature in self.selected_features:
            if feature not in self.norm_dict:
                feature_array = np.concatenate(
                    [sample_dict[feature] for sample_dict in self.xb_dict.values()]
                )
                weight_array = np.concatenate(
                    [sample_dict["weight"] for sample_dict in self.xb_dict.values()]
                )
                mean = np.average(feature_array, weights=weight_array)
                variance = np.average((feature_array - mean) ** 2, weights=weight_array)
                self.norm_dict[feature] = {"mean": mean, "variance": variance}
            norm_means.append(self.norm_dict[feature]["mean"])
            norm_variances.append(self.norm_dict[feature]["variance"])
        x_reshape = train_utils.norarray(
            x_reshape, average=norm_means, variance=norm_variances,
        )
        return x_reshape.copy(), weight_reshape.copy()

    def get_reweight(
        self,
        input_type,
        array_key="all",
        reset_mass=None,
        reset_array_key="all",
        norm=True,
    ):
        if reset_mass == None:
            reset_mass = self.reset_mass
        if input_type == "xs":
            xs_reshape, xs_weight_reshape = self.get_reshape("xs", array_key=array_key)
            return array_utils.modify_array(
                xs_reshape,
                xs_weight_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=norm,
                sumofweight=self.sig_weight,
            )
        elif input_type == "xb":
            xb_reshape, xb_weight_reshape = self.get_reshape("xb", array_key=array_key)
            if reset_mass:
                reset_mass_arr, reset_mass_wts = self.get_reshape(
                    "xs", array_key=reset_array_key
                )
                xb_reshape, xb_weight_reshape = array_utils.modify_array(
                    xb_reshape,
                    xb_weight_reshape,
                    reset_mass=True,
                    reset_mass_array=reset_mass_arr,
                    reset_mass_weights=reset_mass_wts,
                    reset_mass_id=self.reset_mass_id,
                )
            return array_utils.modify_array(
                xb_reshape,
                xb_weight_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=norm,
                sumofweight=self.bkg_weight,
            )
        elif input_type == "xd":
            xd_reshape, xd_weight_reshape = self.get_reshape("xd", array_key=array_key)
            if reset_mass:
                reset_mass_arr, reset_mass_wts = self.get_reshape(
                    "xs", array_key=reset_array_key
                )
                xd_reshape = array_utils.modify_array(
                    xd_reshape,
                    xd_weight_reshape,
                    reset_mass=True,
                    reset_mass_array=reset_mass_arr,
                    reset_mass_weights=reset_mass_wts,
                    reset_mass_id=self.reset_mass_id,
                )
            return array_utils.modify_array(
                xd_reshape,
                xd_weight_reshape,
                remove_negative_weight=self.remove_negative_weight,
                norm=norm,
                sumofweight=self.data_weight,
            )
        else:
            logger.warn("Unknown input_type")
            return None

    def get_train_test_arrays(
        self,
        sig_key="all",
        bkg_key="all",
        multi_class_bkgs=[],
        reset_mass=None,
        output_keys=[],
    ):
        if reset_mass == None:
            reset_mass = self.reset_mass

        # deal with different number of output nodes
        xs_reweight, xs_weight_reweight = self.get_reweight(
            "xs", array_key=sig_key, reset_mass=reset_mass, reset_array_key=sig_key
        )
        logger.debug(f"xs_weight_reweight shape: {xs_weight_reweight.shape}")
        if len(multi_class_bkgs) > 0:
            xb_reweight = None
            xb_weight_reweight = None
            yb = None
            num_bkg_nodes = len(multi_class_bkgs)
            ys_element = np.zeros(num_bkg_nodes + 1)
            ys_element[0] = 1
            ys = np.tile(ys_element, (len(xs_reweight), 1))
            for node_num, bkg_node in enumerate(multi_class_bkgs):
                bkg_node_list = ("".join(bkg_node.split())).split("+")
                xb_reweight_node = None
                wt_reweight_node = None
                for bkg_ele in bkg_node_list:
                    xb_reweight_ele, xb_wt_reweight_ele = self.get_reweight(
                        "xb",
                        array_key=bkg_ele,
                        reset_mass=reset_mass,
                        reset_array_key=sig_key,
                    )
                    if xb_reweight_node is None:
                        xb_reweight_node = xb_reweight_ele
                        wt_reweight_node = xb_wt_reweight_ele
                    else:
                        xb_reweight_node = np.concatenate(
                            (xb_reweight_node, xb_reweight_ele)
                        )
                        wt_reweight_node = np.concatenate(
                            (wt_reweight_node, xb_wt_reweight_ele)
                        )
                xb_reweight_node, wt_reweight_node = array_utils.modify_array(
                    xb_reweight_node, wt_reweight_node, norm=True, sumofweight=1000,
                )
                yb_single_element = np.zeros(num_bkg_nodes + 1)
                yb_single_element[node_num + 1] = 1
                yb_single = np.tile(yb_single_element, (len(xb_reweight_node), 1))
                if xb_reweight is None:
                    xb_reweight = xb_reweight_node
                    yb = yb_single
                    xb_weight_reweight = wt_reweight_node
                else:
                    xb_reweight = np.concatenate((xb_reweight, xb_reweight_node))
                    yb = np.concatenate((yb, yb_single))
                    xb_weight_reweight = np.concatenate(
                        (xb_weight_reweight, wt_reweight_node)
                    )
            xb_reweight, xb_weight_reweight = array_utils.modify_array(
                xb_reweight, xb_weight_reweight, norm=True, sumofweight=self.bkg_weight
            )
            (
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
                wts_train,
                wts_test,
                xb_train,
                xb_test,
                yb_train,
                yb_test,
                wtb_train,
                wtb_test,
            ) = train_utils.split_and_combine(
                xs_reweight,
                xs_weight_reweight,
                xb_reweight,
                xb_weight_reweight,
                ys=ys,
                yb=yb,
                test_rate=self.test_rate,
                shuffle_seed=self.rdm_seed,
            )
        else:
            xb_reweight, xb_weight_reweight = self.get_reweight(
                "xb", array_key=bkg_key, reset_mass=reset_mass, reset_array_key=sig_key
            )
            (
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
                wts_train,
                wts_test,
                xb_train,
                xb_test,
                yb_train,
                yb_test,
                wtb_train,
                wtb_test,
            ) = train_utils.split_and_combine(
                xs_reweight,
                xs_weight_reweight,
                xb_reweight,
                xb_weight_reweight,
                test_rate=self.test_rate,
                shuffle_seed=self.rdm_seed,
            )
        full_dict = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "wt_train": wt_train,
            "wt_test": wt_test,
            "xs_train": xs_train,
            "xs_test": xs_test,
            "ys_train": ys_train,
            "ys_test": ys_test,
            "wts_train": wts_train,
            "wts_test": wts_test,
            "xb_train": xb_train,
            "xb_test": xb_test,
            "yb_train": yb_train,
            "yb_test": yb_test,
            "wtb_train": wtb_train,
            "wtb_test": wtb_test,
        }
        if output_keys:
            partial_dict = {}
            for key in output_keys:
                partial_dict[key] = full_dict[key]
            return partial_dict
        else:
            return full_dict
