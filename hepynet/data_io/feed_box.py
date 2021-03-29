from __future__ import nested_scopes

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

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
    ) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """Loads necessary inputs without preprocessing

        Args:
            input_type: could be 'xs', 'xb', 'xd
            array_key: sample name to be loaded or use 'all' or 'all_norm' to load all
            add_validation_features: additional validation features to be loaded

        """
        # get list of needed features
        features = ["weight"]
        features += self._ic.selected_features
        if add_validation_features:
            features += self._ic.validation_features
        features = list(set(features))
        # get input array dict
        array_dict = self.get_input_array_dict(input_type, part_features=features)
        # get list of necessary samples
        sample_key_list = []
        if array_key in list(array_dict.keys()):
            sample_key_list = [array_key]
        elif array_key == "all" or array_key == "all_norm":
            sample_key_list = list(array_dict.keys())
        else:
            logger.error("Unknown array_key")
        # get output array_dict
        df_dict = dict()
        for sample in sample_key_list:
            sample_df = pd.DataFrame(array_dict[sample])
            df_dict[sample] = sample_df

        return features, df_dict

    def get_raw_merged(
        self,
        input_type,
        features=[],
        array_key="all",
        add_validation_features=False,
        remove_duplicated_col=True,
    ) -> pd.DataFrame:
        df_features, df_dict = self.get_raw(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )
        if not features:
            features = df_features
        inputs_df = array_utils.merge_samples_df(df_dict, features, array_key=array_key)
        if remove_duplicated_col:
            return inputs_df.loc[:, ~inputs_df.columns.duplicated()]
        else:
            return inputs_df

    def get_reshape(
        self, input_type: str, array_key: str = "all", add_validation_features=False
    ) -> Dict[str, pd.DataFrame]:
        """Normalizes input distributions"""
        df_features, df_dict = self.get_raw(
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
        for sample_df in df_dict.values():
            for feature in df_features:
                if feature in norm_alias:
                    feature_key = norm_alias[feature]
                else:
                    feature_key = feature
                if feature_key == "weight":
                    sample_df.loc[sample_df["weight"] < 0, ["weight"]] = 0
                elif feature_key in self._norm_dict:
                    f_mean = self._norm_dict[feature_key]["mean"]
                    f_var = self._norm_dict[feature_key]["variance"]
                    sample_df[feature] = (sample_df[feature] - f_mean) / np.sqrt(f_var)
                else:
                    logger.debug(
                        f"{feature_key} is not a training feature, ignoring reshape process."
                    )
        return df_features, df_dict

    def get_reshape_merged(
        self, input_type, features=None, array_key="all", add_validation_features=False
    ):
        df_features, array_dict = self.get_reshape(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )
        if not features:
            features = df_features
        inputs_df = array_utils.merge_samples_df(
            array_dict, features, array_key=array_key
        )
        return inputs_df

    def get_reweight(
        self,
        input_type: str,
        array_key: str = "all",
        add_validation_features=False,
        reset_mass: bool = None,
        reset_array_key: str = "all",
    ) -> Dict[str, pd.DataFrame]:
        """Scales weight and resets feature distributions"""
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        df_features, df_dict = self.get_reshape(
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
        ## get total weight
        total_weight = 0
        for sample_df in df_dict.values():
            total_weight += sample_df["weight"].sum()
        ## reweight sample by sample
        for sample_df in df_dict.values():
            norm_weight = sumofweight * sample_df["weight"].sum() / total_weight
            array_utils.reweight_array(
                sample_df["weight"],
                remove_negative=self._ic.rm_negative_weight_events,
                norm_weight=norm_weight,
            )
        # reset feature for pDNN if needed
        if reset_mass and (input_type == "xb" or input_type == "xd"):
            reset_feature_name = self._ic.reset_feature_name
            reset_sumofweight = self._ic.sig_sumofweight
            ref_array_df = self.get_reweight_merged(
                "xs",
                features=[reset_feature_name, "weight"],
                array_key=reset_array_key,
                sumofweight=reset_sumofweight,
            )
            ref_array = ref_array_df[reset_feature_name]
            ref_weight = ref_array_df["weight"]
            for sample_key, sample_df in df_dict.items():
                reset_array = df_dict[sample_key][self._ic.reset_feature_name]
                array_utils.redistribute_array(reset_array, ref_array, ref_weight)
        return df_features, df_dict

    def get_reweight_merged(
        self,
        input_type,
        features=[],
        array_key="all",
        add_validation_features=False,
        reset_mass: bool = None,
        reset_array_key: str = "all",
        sumofweight=1000,
        tag_sample=False,
    ):
        df_features, df_dict = self.get_reweight(
            input_type,
            array_key=array_key,
            reset_mass=reset_mass,
            reset_array_key=reset_array_key,
            add_validation_features=add_validation_features,
        )
        if not features:
            features = df_features
        if tag_sample:
            for sample, sample_df in df_dict.items():
                sample_id = self.get_sample_id(sample)
                sample_df["sample_id"] = np.full(len(sample_df), sample_id)
            features += ["sample_id"]
        inputs_df = array_utils.merge_samples_df(
            df_dict, features, array_key=array_key, sumofweight=sumofweight
        )
        return inputs_df

    def get_sample_id(self, sample_key):
        """Gets ID for signal/background/data samples

        Note:
            0 for data
            1, 2, 3, ... for signal
            -1, -2, -3, ... for background 

        """
        ic = self._ic
        if sample_key in ic.sig_list:
            return 1 + ic.sig_list.index(sample_key)
        elif sample_key in ic.bkg_list:
            return -(1 + ic.bkg_list.index(sample_key))
        elif sample_key in ic.data_list:
            return 0
        else:
            logger.error(f"Unknown sample_key: {sample_key}")
            return None

    def get_train_test_df(
        self,
        sig_key: str = "all",
        bkg_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Gets train test input arrays

        TODO: should merge with get_train_test_df_multi_nodes, as binary
        case is a special multi-nodes case

        """
        if multi_class_bkgs is not None and len(multi_class_bkgs) > 0:
            return self.get_train_test_df_multi_nodes(
                sig_key=sig_key,
                multi_class_bkgs=multi_class_bkgs,
                reset_mass=reset_mass,
            )
        else:
            if reset_mass == None:
                reset_mass = self._ic.reset_feature
            # load sig
            sig_df = self.get_reweight_merged(
                "xs", array_key=sig_key, reset_array_key=sig_key, tag_sample=True
            )
            # load bkg
            bkg_df = self.get_reweight_merged(
                "xb",
                array_key=bkg_key,
                reset_mass=reset_mass,
                reset_array_key=sig_key,
                tag_sample=True,
            )
            # add tags
            sig_len = len(sig_df)
            sig_df["is_sig"] = np.full(sig_len, True)
            sig_df["is_mc"] = np.full(sig_len, True)
            sig_df["y"] = np.full(sig_len, 1, dtype=np.float32)
            bkg_len = len(bkg_df)
            bkg_df["is_sig"] = np.full(bkg_len, False)
            bkg_df["is_mc"] = np.full(bkg_len, True)
            bkg_df["y"] = np.full(bkg_len, 0, dtype=np.float32)
            # split & tag train test
            sample_df = pd.concat([sig_df, bkg_df], ignore_index=True)
            sss = StratifiedShuffleSplit(n_splits=1)
            for train_index, _ in sss.split(sample_df.values, sample_df["y"].values):
                break
            sample_df["is_train"] = np.full(len(sample_df), False)
            sample_df.loc[train_index, "is_train"] = True
            return sample_df

    def get_train_test_df_multi_nodes(
        self,
        sig_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
    ) -> Dict[str, np.ndarray]:
        """Gets train test input arrays for multi-nodes training"""
        # TODO: need to rewrite
        pass
        """
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        # load sig
        xs_dict = self.get_reweight(
            "xs", array_key=sig_key, reset_mass=reset_mass, reset_array_key=sig_key
        )
        xs_reweight_df = array_utils.merge_samples_df(
            xs_dict,
            self._ic.selected_features,
            array_key=sig_key,
            sumofweight=self._ic.sig_sumofweight,
        )
        # load bkg
        xb_reweight = None
        xb_weight_reweight = None
        yb = None
        num_bkg_nodes = len(multi_class_bkgs)
        ys_element = np.zeros(num_bkg_nodes + 1)
        ys_element[0] = 1
        ys = np.tile(ys_element, (len(xs_reweight_df), 1))
        for node_num, bkg_node in enumerate(multi_class_bkgs):
            bkg_node_list = ("".join(bkg_node.split())).split("+")
            xb_reweight_node_df = pd.DataFrame(columns=self._ic.selected_features)
            for bkg_ele in bkg_node_list:
                xb_reweight_ele_df = self.get_reweight_merged(
                    "xb",
                    features=self._ic.selected_features,
                    array_key=bkg_ele,
                    reset_mass=reset_mass,
                    reset_array_key=sig_key,
                )
                xb_reweight_node_df = xb_reweight_node_df.append(xb_reweight_ele_df)

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
            xs_reweight_df,
            xb_reweight_df,
            ys=ys,
            yb=yb,
            output_keys=output_keys,
            test_rate=self._tc.test_rate,
        )
        """

    def update_norm_dict(self, features: Optional[List[str]] = None):
        _, df_dict = self.get_raw(
            "xb", array_key="all"
        )  ## TODO: enable customized keys for norm dict
        weight_array = np.concatenate(
            [sample_dict["weight"] for sample_dict in df_dict.values()]
        )  ## TODO: enable negative weight cleaning here or not?
        feature_list = list()
        if features is not None:
            feature_list = features
        else:
            feature_list = self._ic.selected_features
        for feature in feature_list:
            feature_array = np.concatenate(
                [sample_dict[feature] for sample_dict in df_dict.values()]
            )
            mean = np.average(feature_array, weights=weight_array)
            variance = np.average((feature_array - mean) ** 2, weights=weight_array)
            self._norm_dict[feature] = {"mean": mean, "variance": variance}
            logger.debug(f"Feature {feature} mean: {mean}, variance: {variance}")
