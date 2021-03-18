import logging
import time
from typing import Dict, List, Optional

import numpy as np

from hepynet.common import array_utils
from hepynet.data_io import numpy_io
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
            if self._ic.reset_feature_name in self._ic.selected_features:
                self._reset_mass_id = self._ic.selected_features.index(
                    self._ic.reset_feature_name
                )
            else:
                logger.critical(
                    f"input.reset_feature is True but input.reset_feature_name {self._ic.reset_feature_name} is not in input.selected_features, please check the config"
                )
                exit(1)
        else:
            self._reset_mass_id = None
        self._rdm_seed = self._ic.rdm_seed
        self._array_prepared = False
        # set random seed
        if self._rdm_seed is None:
            self._rdm_seed = int(time.time())

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

    def update_norm_dict(self, features:Optional[List[str]]=None):
        xb_dict = numpy_io.load_npy_arrays(self._job_config, "bkg", part_features=features)
        weight_array = np.concatenate(
            [sample_dict["weight"] for sample_dict in xb_dict.values()]
        )
        if features is not None:
            feature_list = features
        else:
            feature_list = self._ic.selected_features
        for feature in feature_list:
            feature_array = np.concatenate(
                [sample_dict[feature] for sample_dict in xb_dict.values()]
            )
            mean = np.average(feature_array, weights=weight_array)
            variance = np.average((feature_array - mean) ** 2, weights=weight_array)
            self._norm_dict[feature] = {"mean": mean, "variance": variance}
            logger.debug(f"Feature {feature} mean: {mean}, variance: {variance}")

    def get_job_config(self):
        return self._job_config

    def get_norm_dict(self):
        return self._norm_dict

    def get_raw(self, input_type, array_key="all", add_validation_features=False):
        array_out = None
        weight_out = None
        # get dict
        array_dict = None
        if input_type == "xs":
            array_dict = numpy_io.load_npy_arrays(self._job_config, "sig")
        elif input_type == "xb":
            array_dict = numpy_io.load_npy_arrays(self._job_config, "bkg")
        elif input_type == "xd":
            if self._ac.apply_data:
                array_dict = numpy_io.load_npy_arrays(self._job_config, "data")
            else:
                logger.warn(
                    "Trying to get data array as apply_data option is set to False"
                )
                return None
        else:
            logger.warn("Unknown input type")
            return None
        if add_validation_features:
            if self._ic.validation_features is None:
                validation_features = []
            else:
                validation_features = self._ic.validation_features
            feature_list = self._ic.selected_features + validation_features
        else:
            feature_list = self._ic.selected_features
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
        if self._ic.rm_negative_weight_events:
            weight_out = array_utils.clean_negative_weights(weight_out)
        weight_out = weight_out.reshape((-1,))
        return array_out, weight_out

    def get_reshape(self, input_type, array_key="all"):
        x_reshape, weight_reshape = self.get_raw(input_type, array_key=array_key)
        norm_means = []
        norm_variances = []
        for feature in self._ic.selected_features:
            norm_alias = self._job_config.input.feature_norm_alias.get_config_dict()
            mean = None
            variance = None
            if feature in self._norm_dict:
                mean = self._norm_dict[feature]["mean"]
                variance = self._norm_dict[feature]["variance"]
            else:
                found_norm_alias = False
                # check feature normalization alias
                if feature in norm_alias:
                    alias_feature = norm_alias[feature]
                    if alias_feature in self._norm_dict:
                        logger.debug(f"Using norm_alias for: {feature}")
                        mean = self._norm_dict[alias_feature]["mean"]
                        variance = self._norm_dict[alias_feature]["variance"]
                        found_norm_alias = True
                    else:
                        logger.warn(
                            f"Specified but can't find norm_alias for: {feature}"
                        )
                # recalculate if no existing mean/variance found
                if not found_norm_alias:
                    logger.debug("Recalculating normalization parameters for {feature}")
                    self.update_norm_dict([feature])
            norm_means.append(mean)
            norm_variances.append(variance)
        x_reshape = train_utils.norarray(
            x_reshape, average=norm_means, variance=norm_variances,
        )
        return x_reshape, weight_reshape

    def get_reweight(
        self,
        input_type,
        array_key="all",
        reset_mass=None,
        reset_array_key="all",
        norm=True,
    ):
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        if input_type == "xs":
            xs_reshape, xs_weight_reshape = self.get_reshape("xs", array_key=array_key)
            return array_utils.modify_array(
                xs_reshape,
                xs_weight_reshape,
                remove_negative_weight=self._ic.rm_negative_weight_events,
                norm=norm,
                sumofweight=self._ic.sig_sumofweight,
            )
        elif input_type == "xb" or input_type == "xd":
            x_reshape, x_weight_reshape = self.get_reshape(
                input_type, array_key=array_key
            )
            if reset_mass:
                reset_mass_arr, reset_mass_wts = self.get_reshape(
                    "xs", array_key=reset_array_key
                )
                x_reshape, x_weight_reshape = array_utils.modify_array(
                    x_reshape,
                    x_weight_reshape,
                    reset_mass=True,
                    reset_mass_array=reset_mass_arr,
                    reset_mass_weights=reset_mass_wts,
                    reset_mass_id=self._reset_mass_id,
                )
            return array_utils.modify_array(
                x_reshape,
                x_weight_reshape,
                remove_negative_weight=self._ic.rm_negative_weight_events,
                norm=norm,
                sumofweight=self._ic.bkg_sumofweight,
            )
        else:
            logger.warn("Unknown input_type")
            return None

    def get_train_test_arrays(
        self,
        sig_key: str = "all",
        bkg_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
    ) -> Dict[str, np.ndarray]:
        if reset_mass == None:
            reset_mass = self._ic.reset_feature

        # deal with different number of output nodes
        ## load sig
        xs_reweight, xs_weight_reweight = self.get_reweight(
            "xs", array_key=sig_key, reset_mass=reset_mass, reset_array_key=sig_key
        )
        logger.debug(f"xs_weight_reweight shape: {xs_weight_reweight.shape}")
        ## load bkg
        if multi_class_bkgs is not None and len(multi_class_bkgs) > 0:
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
                xb_reweight,
                xb_weight_reweight,
                norm=True,
                sumofweight=self._ic.bkg_sumofweight,
            )
            return train_utils.split_and_combine(
                xs_reweight,
                xs_weight_reweight,
                xb_reweight,
                xb_weight_reweight,
                ys=ys,
                yb=yb,
                output_keys=output_keys,
                test_rate=self._tc.test_rate,
                shuffle_seed=self._rdm_seed,
            )
        else:
            xb_reweight, xb_weight_reweight = self.get_reweight(
                "xb", array_key=bkg_key, reset_mass=reset_mass, reset_array_key=sig_key
            )
            return train_utils.split_and_combine(
                xs_reweight,
                xs_weight_reweight,
                xb_reweight,
                xb_weight_reweight,
                output_keys=output_keys,
                test_rate=self._tc.test_rate,
                shuffle_seed=self._rdm_seed,
            )

    def get_train_test_arrays_k_fold(
        self,
        sig_key: str = "all",
        bkg_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
        num_folds: int = 2,
    ):
        """Gets the k-fold training train/test inputs

        Args:
            sig_key: signal key for input generation
            bkg_key: background key for input generation
            multi_class_bkgs: list of multi-class output nodes
            reset_mass: whether to reset mass according to signal distribution (for PNN)
            output_keys: out
            num_folds: 

        """
        pass
