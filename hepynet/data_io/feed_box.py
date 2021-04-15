from __future__ import nested_scopes

import copy
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from hepynet.common import config_utils
from hepynet.data_io import numpy_io

logger = logging.getLogger("hepynet")


class Feedbox(object):
    """DNN inputs management class."""

    def __init__(
        self, job_config: config_utils.Hepy_Config, norm_dict=None,
    ):
        # get config
        self._job_config = job_config.clone()
        self._data_dir = pathlib.Path(numpy_io.get_data_dir())
        jc = self._job_config.job.clone()
        self._array_prepared = False

        # get norm dict
        if norm_dict is None:
            if jc.job_name == "apply":
                logger.warning(
                    f"No norm_dict found, but this is not expected as the job type is 'apply'! The normalization parameters may not be consistent with the training!"
                )
            self._norm_dict = {}
        else:
            self._norm_dict = copy.deepcopy(norm_dict)

        self._array_prepared = True

    def get_raw_df(self):
        logger.info("Loading raw input DataFrame...")
        ic = self._job_config.input.clone()
        logger.info("> Successfully loaded raw input DataFrame...")
        return pd.read_feather(self._data_dir / ic.input_path)

    def get_processed_df(self):
        logger.info("Loading processed input DataFrame...")
        ic = self._job_config.input.clone()
        out_df: pd.DataFrame = pd.read_feather(self._data_dir / ic.input_path)
        # reshape inputs
        if ic.reshape_input:
            logger.info("> Reshaping inputs")
            # check missing norm parameters
            missing_norms = []
            norm_alias = ic.feature_norm_alias.get_config_dict()
            for feature in ic.selected_features:
                if feature not in self._norm_dict:
                    found_norm_alias = False
                    # check feature normalization alias
                    if feature in norm_alias:
                        feature_alias = norm_alias[feature]
                        if feature_alias in self._norm_dict:
                            logger.debug(f"Found norm_alias for: {feature}")
                            found_norm_alias = True
                            self._norm_dict[feature] = copy.deepcopy(
                                self._norm_dict[feature_alias]
                            )
                        else:
                            logger.warn(
                                f"Specified but can't find norm_alias for: {feature}"
                            )
                    if not found_norm_alias:
                        missing_norms.append(feature)
            # recalculate missing norm parameters
            if len(missing_norms) > 0:
                logger.debug(
                    f"Recalculating normalization parameters for {missing_norms}"
                )
                self.update_norm_dict(missing_norms)
            # reshape inputs
            for feature in ic.selected_features:
                if feature in self._norm_dict:
                    f_mean = self._norm_dict[feature]["mean"]
                    f_var = self._norm_dict[feature]["variance"]
                    new_values = (out_df[feature].values - f_mean) / np.sqrt(f_var)
                    out_df.loc[:, feature].update(new_values)
                else:
                    logger.debug(
                        f"{feature} is not in self._norm_dict, ignoring reshape process."
                    )
        # deal with negative weights
        nwt_method = ic.negative_weight_process
        logger.info(f"> Processing negative weights with method: {nwt_method}")
        if nwt_method == "to_zero":
            out_df.loc[out_df["weight"] < 0, "weight"] = 0
        elif nwt_method == "to_positive":
            out_df.loc[out_df["weight"] < 0, "weight"].update(
                out_df.loc[out_df["weight"] < 0, "weight"].abs()
            )
        elif nwt_method == "keep":
            pass
        else:
            logger.warn(f"> Unknown ic.negative_weight_process {nwt_method}")
        # reweight inputs
        if ic.reweight_input:
            logger.info("> Reweighting inputs")
            # reweight signal
            self.reweight_df(
                out_df, ic.sig_list, ic.sig_key, ic.sig_sumofweight, True, True
            )
            # reweight background
            self.reweight_df(
                out_df, ic.bkg_list, ic.bkg_key, ic.bkg_sumofweight, True, False
            )
        # reset physic para for pDNN
        if ic.reset_mass:
            logger.info("> Resetting physic para distribution for pDNN")
            ref_df = out_df.loc[
                out_df["is_sig"] == True, [ic.reset_feature_name, "weight"]
            ]
            ref_array = ref_df[ic.reset_feature_name]
            ref_weight = ref_df["weight"]
            ref_weight_positive = ref_weight.copy()
            ref_weight_positive[ref_weight_positive < 0] = 0
            sump = ref_weight_positive.sum()
            reset_values = np.random.choice(
                ref_array.values,
                size=len(ref_weight),
                p=(1 / sump) * ref_weight_positive.values,
            )
            out_df.loc[out_df["is_sig"] == False, ic.reset_feature_name].update(
                reset_values
            )
        # set y
        logger.info("> Setting up y values")
        out_df.loc[:, "y"] = 0
        out_df.loc[out_df["is_sig"] == True, "y"] = 1
        # tag train / test
        logger.info("> Tagging train / test")
        sss = StratifiedShuffleSplit(n_splits=1)
        y_arr = out_df["y"].values
        for train_index, _ in sss.split(y_arr, y_arr):
            break
        out_df.loc[:, "is_train"] = False
        out_df.loc[train_index, "is_train"] = True

        logger.info("> Successfully loaded processed input DataFrame...")
        return out_df

    def get_job_config(self):
        return self._job_config

    def get_norm_dict(self):
        return self._norm_dict

    def reweight_df(
        self, df: pd.DataFrame, sample_list, array_key, sumofweight, is_mc, is_sig
    ):
        total_wt = df.loc[
            (df["is_mc"] == is_mc) & (df["is_sig"] == is_sig), "weight"
        ].sum()
        for sample in sample_list:
            sample_wt = df.loc[df["sample_name"] == sample, "weight"]
            sample_sumw = sample_wt.sum()
            if sample_sumw <= 0:
                continue
            # get norm factor
            norm_factor = 1
            if array_key == "all":

                norm_factor = sumofweight / total_wt
            elif array_key == "all_norm":
                n_samples = len(sample_list)
                norm_factor = (sumofweight / n_samples) / sample_sumw
            elif array_key == sample:
                norm_factor = sumofweight / sample_sumw
            else:
                logger.warn(f"Unknown array_key {array_key}")
            df.loc[df["sample_name"] == sample, "weight"] = sample_wt * norm_factor

    def update_norm_dict(self, features: Optional[List[str]] = None):
        ic = self._job_config.input.clone()
        feature_list = list()
        if features is not None:
            feature_list = features
        else:
            feature_list = ic.selected_features + ic.validation_features
        feature_list = list(set().union(feature_list))  # remove duplicates
        df = self.get_raw_df()
        weight_array = df.loc[
            (df["is_mc"] == True) & (df["is_sig"] == False), "weight"
        ].values
        for feature in feature_list:
            feature_array = df.loc[
                (df["is_mc"] == True) & (df["is_sig"] == False), feature
            ].values
            mean = np.average(feature_array, weights=weight_array)
            variance = np.average((feature_array - mean) ** 2, weights=weight_array)
            self._norm_dict[feature] = {"mean": mean, "variance": variance}
            logger.info(
                f"> > Recalculated norm factors for feature {feature} mean: {mean}, variance: {variance}"
            )
