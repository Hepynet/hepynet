from __future__ import nested_scopes
import logging
from typing import Dict, List, Optional

import numpy as np

from hepynet.data_io import array_utils, numpy_io
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


class Feedbox(object):
    """DNN inputs management class."""

    def __init__(
        self, job_config, model_meta=None,
    ):
        # get config
        self._job_config = job_config.clone()
        self._jc = self._job_config.job
        self._ic = self._job_config.input
        self._tc = self._job_config.train
        self._ac = self._job_config.apply
        # meta info
        if self._ic.reset_feature:
            if self._ic.reset_feature_name not in self._ic.selected_features:
                logger.critical(
                    f"input.reset_feature is True but input.reset_feature_name {self._ic.reset_feature_name} is not in input.selected_features, please check the config"
                )
                exit(1)
        self._array_prepared = False

        # get normalization parameters
        no_norm_paras = False
        if model_meta is None:
            no_norm_paras = True
        else:
            if model_meta["norm_dict"] is None:
                no_norm_paras = True
            else:
                no_norm_paras = False
        self._norm_dict = {}
        if no_norm_paras:
            if self._jc.job_name == "apply":
                logger.warning(
                    f"No norm_dict found, but this is not expected as the job type is 'apply'! The normalization parameters may not be consistent with the training!"
                )
            logger.info(
                f"No norm_dict found, recalculating norm parameters according to background input distributions"
            )
            self.update_norm_dict()
        else:
            self._norm_dict = model_meta["norm_dict"]
        self._array_prepared = True

    def get_input_array_dict(
        self, input_type: str, part_features: Optional[List[str]] = None
    ):
        # get input dict
        array_dict = dict()
        if input_type == "xs":
            array_dict = numpy_io.load_npy_arrays(
                self._job_config, "sig", part_features=part_features
            )
        elif input_type == "xb":
            array_dict = numpy_io.load_npy_arrays(
                self._job_config, "bkg", part_features=part_features
            )
        elif input_type == "xd":
            if self._ac.apply_data:
                array_dict = numpy_io.load_npy_arrays(
                    self._job_config, "data", part_features=part_features
                )
            else:
                logger.warn(
                    "Trying to get data array but apply_data option is set to False"
                )
        else:
            logger.error("Unknown input type")
        return array_dict

    def get_job_config(self):
        return self._job_config

    def get_norm_dict(self):
        return self._norm_dict

    def get_raw(
        self, input_type, array_key="all", add_validation_features=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Loads necessary inputs without preprocessing

        Args:
            input_type: could be 'xs', 'xb', 'xd
            array_key: sample name to be loaded or use 'all' or 'all_norm' to load all
            add_validation_features: additional validation features to be loaded

        """
        array_dict_out = dict()
        if add_validation_features:
            feature_list = array_utils.merge_select_val_features(
                self._ic.selected_features,
                self._ic.validation_features,
                append_weight=True,
            )
        else:
            feature_list = self._ic.selected_features.copy() + ["weight"]
        # get input array dict
        array_dict = self.get_input_array_dict(input_type, part_features=feature_list)
        # get list of necessary samples
        sample_key_list = []
        if array_key in list(array_dict.keys()):
            sample_key_list = [array_key]
        elif array_key == "all" or array_key == "all_norm":
            sample_key_list = list(array_dict.keys())
        else:
            logger.error("Unknown array_key")
        # get output array_dict
        for sample in sample_key_list:
            dict_member = dict()
            for feature in feature_list:
                dict_member[feature] = array_dict[sample][feature]
                # TODO: use [:] here or not?
            array_dict_out[sample] = dict_member
        return array_dict_out

    def get_raw_merged(
        self, input_type, features=None, array_key="all", add_validation_features=False
    ):
        if features is None:
            features = self._job_config.input.selected_features.copy()
            if add_validation_features:
                features = array_utils.merge_select_val_features(
                    features, self._job_config.input.validation_features
                )
        array_dict = self.get_raw(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )
        inputs, weights = array_utils.merge_dict_to_inputs(
            array_dict, features, array_key=array_key
        )
        return inputs, weights

    def get_reshape(
        self, input_type: str, array_key: str = "all", add_validation_features=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Normalizes input distributions"""
        array_dict_out = self.get_raw(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )
        # check missing norm parameters
        missing_norms = []
        norm_alias = self._job_config.input.feature_norm_alias.get_config_dict()
        for feature in self._ic.selected_features:
            if feature not in self._norm_dict:
                found_norm_alias = False
                # check feature normalization alias
                if feature in norm_alias:
                    if norm_alias[feature] in self._norm_dict:
                        logger.debug(f"Found norm_alias for: {feature}")
                        found_norm_alias = True
                    else:
                        logger.warn(
                            f"Specified but can't find norm_alias for: {feature}"
                        )
                if not found_norm_alias:
                    missing_norms.append(feature)
        # recalculate missing norm parameters
        if len(missing_norms) > 0:
            logger.debug(f"Recalculating normalization parameters for {missing_norms}")
            self.update_norm_dict(missing_norms)
        # inputs pre-processing
        for sample_dict in array_dict_out.values():
            for feature, feature_array in sample_dict.items():
                if feature == "weight":
                    if self._ic.rm_negative_weight_events:
                        array_utils.clip_negative_weights(feature_array)
                else:
                    feature_key = feature
                    if feature in norm_alias:
                        feature_key = norm_alias[feature]
                    train_utils.norm_array(
                        feature_array,
                        average=self._norm_dict[feature_key]["mean"],
                        variance=self._norm_dict[feature_key]["variance"],
                    )
        return array_dict_out

    def get_reshape_merged(self, input_type, features=None, array_key="all"):
        if features is None:
            features = self._job_config.input.selected_features.copy()
        array_dict = self.get_reshape(input_type, array_key=array_key,)
        inputs, weights = array_utils.merge_dict_to_inputs(
            array_dict, features, array_key=array_key
        )
        return inputs, weights

    def get_reweight(
        self,
        input_type: str,
        array_key: str = "all",
        reset_mass: bool = None,
        reset_array_key: str = "all",
        add_validation_features=False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Scales weight and resets feature distributions"""
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        array_dict_out = self.get_reshape(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )
        # reweight
        sumofweight = 1000
        if input_type == "xs":
            sumofweight = self._ic.sig_sumofweight
        elif input_type == "xb":
            sumofweight = self._ic.bkg_sumofweight
        elif input_type == "xd":
            sumofweight = self._ic.data_sumofweight
        total_weight = 0
        for sample_name, sample_dict in array_dict_out.items():
            total_weight += np.sum(sample_dict["weight"])
        for sample_name, sample_dict in array_dict_out.items():
            weight_array = sample_dict["weight"]
            norm_weight = sumofweight * np.sum(weight_array) / total_weight
            array_utils.reweight_array(
                weight_array,
                remove_negative=self._ic.rm_negative_weight_events,
                norm_weight=norm_weight,
            )
        # reset feature for pDNN if needed
        if reset_mass and (input_type == "xb" or input_type == "xd"):
            reset_array_dict = self.get_reshape("xs", array_key=reset_array_key)
            reset_feature_name = self._ic.reset_feature_name
            reset_sumofweight = self._ic.sig_sumofweight
            ref_array, ref_weights = array_utils.merge_dict_to_inputs(
                reset_array_dict,
                [reset_feature_name],
                array_key=array_key,
                sumofweight=reset_sumofweight,
            )
            reset_array = array_dict_out[sample_name][self._ic.reset_feature_name]
            array_utils.redistribute_array(reset_array, ref_array, ref_weights)
        return array_dict_out

    def get_reweight_merged(
        self,
        input_type,
        features=None,
        array_key="all",
        reset_mass: bool = None,
        reset_array_key: str = "all",
    ):
        if features is None:
            features = self._job_config.input.selected_features.copy()
        array_dict = self.get_reweight(
            input_type,
            array_key=array_key,
            reset_mass=reset_mass,
            reset_array_key=reset_array_key,
        )
        inputs, weights = array_utils.merge_dict_to_inputs(
            array_dict, features, array_key=array_key
        )
        return inputs, weights

    def get_train_test_arrays(
        self,
        sig_key: str = "all",
        bkg_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
    ) -> Dict[str, np.ndarray]:
        """Gets train test input arrays

        TODO: should merge with get_train_test_arrays_multi_nodes, as binary
        case is a special multi-nodes case

        """
        if multi_class_bkgs is not None and len(multi_class_bkgs) > 0:
            return self.get_train_test_arrays_multi_nodes(
                sig_key=sig_key,
                multi_class_bkgs=multi_class_bkgs,
                reset_mass=reset_mass,
                output_keys=output_keys,
            )
        else:
            if reset_mass == None:
                reset_mass = self._ic.reset_feature
            # load sig
            xs_reweight, xs_weight_reweight = self.get_reweight_merged(
                "xs", array_key=sig_key, reset_mass=reset_mass, reset_array_key=sig_key
            )
            logger.debug(f"xs_weight_reweight shape: {xs_weight_reweight.shape}")
            # load bkg
            xb_reweight, xb_weight_reweight = self.get_reweight_merged(
                "xb", array_key=bkg_key, reset_mass=reset_mass, reset_array_key=sig_key
            )
            return array_utils.split_and_combine(
                xs_reweight,
                xs_weight_reweight,
                xb_reweight,
                xb_weight_reweight,
                output_keys=output_keys,
                test_rate=self._tc.test_rate,
            )

    def get_train_test_arrays_multi_nodes(
        self,
        sig_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
    ) -> Dict[str, np.ndarray]:
        """Gets train test input arrays for multi-nodes training"""
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        # load sig
        xs_dict = self.get_reweight(
            "xs", array_key=sig_key, reset_mass=reset_mass, reset_array_key=sig_key
        )
        xs_reweight, xs_weight_reweight = array_utils.merge_dict_to_inputs(
            xs_dict,
            self._ic.selected_features,
            array_key=sig_key,
            sumofweight=self._ic.sig_sumofweight,
        )
        logger.debug(f"xs_weight_reweight shape: {xs_weight_reweight.shape}")
        # load bkg
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
                xb_dict = self.get_reweight(
                    "xb",
                    array_key=bkg_ele,
                    reset_mass=reset_mass,
                    reset_array_key=sig_key,
                )
                xb_reweight_ele, xb_wt_reweight_ele = array_utils.merge_dict_to_inputs(
                    xb_dict, self._ic.selected_features, array_key=bkg_ele
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
            xb_reweight,
            xb_weight_reweight,
            norm=True,
            sumofweight=self._ic.bkg_sumofweight,
        )
        return array_utils.split_and_combine(
            xs_reweight,
            xs_weight_reweight,
            xb_reweight,
            xb_weight_reweight,
            ys=ys,
            yb=yb,
            output_keys=output_keys,
            test_rate=self._tc.test_rate,
        )

    def update_norm_dict(self, features: Optional[List[str]] = None):
        array_dict = self.get_raw(
            "xb", array_key="all"
        )  ## TODO: enable customized keys for norm dict
        weight_array = np.concatenate(
            [sample_dict["weight"] for sample_dict in array_dict.values()]
        )  ## TODO: enable negative weight cleaning here or not?
        feature_list = list()
        if features is not None:
            feature_list = features
        else:
            feature_list = self._ic.selected_features
        for feature in feature_list:
            feature_array = np.concatenate(
                [sample_dict[feature] for sample_dict in array_dict.values()]
            )
            mean = np.average(feature_array, weights=weight_array)
            variance = np.average((feature_array - mean) ** 2, weights=weight_array)
            self._norm_dict[feature] = {"mean": mean, "variance": variance}
            logger.debug(f"Feature {feature} mean: {mean}, variance: {variance}")
