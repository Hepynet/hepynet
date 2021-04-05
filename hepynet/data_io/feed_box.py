from __future__ import nested_scopes

import copy
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from hepynet.common import config_utils
from hepynet.data_io import array_utils, numpy_io
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


class Feedbox(object):
    """DNN inputs management class."""

    def __init__(
        self, job_config: config_utils.Hepy_Config, norm_dict=None,
    ):
        # get config
        self._job_config = job_config.clone()
        self._jc = self._job_config.job
        self._ic = self._job_config.input
        self._tc = self._job_config.train
        self._ac = self._job_config.apply
        self._rc = self._job_config.run
        # meta info
        if self._ic.reset_feature:
            if self._ic.reset_feature_name not in self._ic.selected_features:
                logger.critical(
                    f"input.reset_feature is True but input.reset_feature_name {self._ic.reset_feature_name} is not in input.selected_features, please check the config"
                )
                exit(1)
        self._array_prepared = False

        # get cut index dict
        self._cut_index_dict = numpy_io.get_cut_pass_index_dict(self._job_config)
        # get norm dict
        if norm_dict is None:
            if self._jc.job_name == "apply":
                logger.warning(
                    f"No norm_dict found, but this is not expected as the job type is 'apply'! The normalization parameters may not be consistent with the training!"
                )
            self._norm_dict = {}
        else:
            self._norm_dict = copy.deepcopy(norm_dict)

        self._array_prepared = True

    def dump_processed_inputs(self):
        job_config = self._job_config.clone()
        ic = job_config.input.clone()
        self.update_norm_dict()
        camp_list = numpy_io.get_campaign_list(job_config)
        feature_list = ic.selected_features + ic.validation_features
        feature_list = list(set().union(feature_list))  # remove duplicates
        samples_dict = {"sig": ic.sig_list, "bkg": ic.bkg_list, "data": ic.data_list}
        data_dir = numpy_io.get_data_dir()
        for sample_type, sample_list in samples_dict.items():
            array_key = numpy_io.get_array_key(job_config, sample_type)
            sumw = numpy_io.get_sumofweight(job_config, sample_type)
            total_weight = numpy_io.get_samples_total_weight(job_config, sample_list)
            for sample in sample_list:
                for camp in camp_list:
                    npy_in_dir = pathlib.Path(
                        f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}"
                    )
                    # reshape inputs
                    for feature in feature_list:
                        npy_in_path = npy_in_dir / f"{sample}_{feature}.npy"
                        npy_in = np.load(npy_in_path)
                        f_mean = self._norm_dict[feature]["mean"]
                        f_var = self._norm_dict[feature]["variance"]
                        npy_out = (npy_in - f_mean) / np.sqrt(f_var)
                        npy_out_path = npy_in_dir / f"{sample}_{feature}.reshape.npy"
                        np.save(npy_out_path, npy_out)
                    # get cut index
                    cut_feature_dict = dict()
                    for cut_feature in ic.cut_features:
                        cut_var_in_path = npy_in_dir / f"{sample}_{cut_feature}.npy"
                        cut_var_in = np.load(cut_var_in_path)
                        cut_feature_dict[cut_feature] = cut_var_in
                    cut_index = numpy_io.get_cut_index(job_config, cut_feature_dict)
                    # reweight
                    wt_in_path = npy_in_dir / f"{sample}_weight.npy"
                    wt_in = np.load(wt_in_path)
                    if cut_index is not None:  # apply cut by setting weight to 0
                        wt_in[cut_index] = 0
                    sample_sumw = np.sum(wt_in)
                    if array_key == "all":
                        norm_factor = sumw / total_weight
                    elif array_key == "all_norm":
                        n_samples = len(sample_list)
                        norm_factor = (sumw / n_samples) / sample_sumw
                    elif array_key == sample:
                        norm_factor = sumw / sample_sumw
                    else:
                        logger.error(f"Unknown array_key: {array_key}")
                        norm_factor = 1
                    wt_out = wt_in * norm_factor
                    wt_out_path = npy_in_dir / f"{sample}_weight.reweight.npy"
                    np.save(wt_out_path, wt_out)
                    # tag sig/bkg
                    if sample_type == "sig":
                        sig_tag = np.ones(len(wt_in), dtype=np.int8)
                        sig_tag_path = npy_in_dir / f"{sample}.y.npy"
                        np.save(sig_tag_path, sig_tag)
                    if sample_type == "bkg":
                        bkg_tag = np.zeros(len(wt_in), dtype=np.int8)
                        bkg_tag_path = npy_in_dir / f"{sample}.y.npy"
                        np.save(bkg_tag_path, bkg_tag)
                    # tag train/test
                    rs = ShuffleSplit(n_splits=1, test_size=ic.test_rate)
                    train_tag = np.zeros(len(wt_in), dtype=np.int8)
                    for train_index, _ in rs.split(wt_in):
                        break
                    train_tag[train_index] = 1
                    train_tag_path = npy_in_dir / f"{sample}.tr_tag.npy"
                    np.save(train_tag_path, train_tag)

    def dump_training_df(self):
        ic = self._job_config.input.clone()
        rc = self._job_config.run.clone()
        save_dir = pathlib.Path(rc.save_sub_dir) / "input"
        save_dir.mkdir(parents=True, exist_ok=True)
        # dump signal
        _, sig_df_dict = self.get_reweight(
            "xs", array_key=ic.sig_key, add_validation_features=True
        )
        if logger.level <= logging.DEBUG:
            total_weight = 0
            for sample, sample_df in sig_df_dict.items():
                sample_sum_weight = sample_df["weight"].sum()
                total_weight += sample_sum_weight
                logger.debug(
                    f"#### sample {sample} total weight {sample_df['weight'].sum()}"
                )
            logger.debug(f"Signal total weights: {total_weight}")
        self.tag_signals(sig_df_dict)
        # for sample, sample_df in sig_df_dict.items():
        #    save_path = save_dir / f"{sample}.feather"
        #    logger.info(f"Dumping inputs for {sample}")
        #    sample_df.to_feather(save_path)
        #    sample_df = None
        # sig_df_dict.clear()
        # dump background
        _, bkg_df_dict = self.get_reweight(
            "xb",
            array_key=ic.bkg_key,
            add_validation_features=True,
            reset_mass=ic.reset_feature,
        )
        if logger.level <= logging.DEBUG:
            total_weight = 0
            for sample, sample_df in bkg_df_dict.items():
                sample_sum_weight = sample_df["weight"].sum()
                total_weight += sample_sum_weight
                logger.debug(f"Sample {sample} total weight {sample_sum_weight}")
            logger.debug(f"Background total weights: {total_weight}")
        self.tag_backgrounds(bkg_df_dict)
        # for sample, sample_df in bkg_df_dict.items():
        #    save_path = save_dir / f"{sample}.feather"
        #    logger.info(f"Dumping inputs for {sample}")
        #    sample_df.to_feather(save_path)
        #    sample_df = None
        # bkg_df_dict.clear()
        logger.info("Concatenating processed inputs")
        out_df = pd.concat(
            list(sig_df_dict.values()) + list(bkg_df_dict.values()), ignore_index=True
        )
        sig_df_dict.clear()
        bkg_df_dict.clear()
        # tag train/test
        sss = StratifiedShuffleSplit(n_splits=1)
        y_arr = out_df["y"].values
        # Note that providing y is sufficient to generate the splits and
        # hence np.zeros(n_samples) may be used as a placeholder for X
        # instead of actual training data.
        for train_index, _ in sss.split(y_arr, y_arr):
            break
        out_df["is_train"] = False
        out_df.loc[train_index, "is_train"] = True
        logger.info(f"Dumping inputs for training to: {save_dir}")
        # out_df.to_feather(save_dir / "train.feather")
        train_index = out_df["is_train"] == True
        test_index = out_df["is_train"] == False
        cols = ic.selected_features
        np.save(save_dir / "x_train.npy", out_df.loc[train_index, cols].to_numpy())
        np.save(save_dir / "x_test.npy", out_df.loc[test_index, cols].to_numpy())
        np.save(save_dir / "y_train.npy", out_df.loc[train_index, "y"].to_numpy())
        np.save(save_dir / "y_test.npy", out_df.loc[test_index, "y"].to_numpy())
        np.save(save_dir / "wt_train.npy", out_df.loc[train_index, "weight"].to_numpy())
        np.save(save_dir / "wt_test.npy", out_df.loc[test_index, "weight"].to_numpy())
        norm_dict_path = save_dir / "norm_dict.yaml"
        with open(norm_dict_path, "w") as norm_file:
            yaml.dump(self._norm_dict, norm_file, indent=2)

    def get_input_array_dict(
        self,
        input_type: str,
        part_features: Optional[List[str]] = None,
        include_weight=True,
    ):
        logger.debug("@ data_io.feed_box.get_input_array_dict")
        # get input dict
        array_dict = dict()
        if input_type == "xs":
            array_dict = numpy_io.load_npy_arrays(
                self._job_config,
                "sig",
                part_features=part_features,
                cut_pass_index_dict=self._cut_index_dict,
                include_weight=include_weight,
            )
        elif input_type == "xb":
            array_dict = numpy_io.load_npy_arrays(
                self._job_config,
                "bkg",
                part_features=part_features,
                cut_pass_index_dict=self._cut_index_dict,
                include_weight=include_weight,
            )
        elif input_type == "xd":
            if self._ac.apply_data:
                array_dict = numpy_io.load_npy_arrays(
                    self._job_config,
                    "data",
                    part_features=part_features,
                    cut_pass_index_dict=self._cut_index_dict,
                    include_weight=include_weight,
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

    """
    def get_processed_df(self):
        ic = self._job_config.input.clone()
        rc = self._job_config.run.clone()
        save_dir = pathlib.Path(rc.save_sub_dir) / "input"
        ## load signal
        # df_out = None
        # sig_df_dict = dict()
        # for sample in ic.sig_list:
        #    sample_path = save_dir / f"{sample}.feather"
        #    sig_df_dict[sample] = pd.read_feather(sample_path)
        # self.reweight_df_dict("xs", sig_df_dict)
        # for sample in sig_df_dict.keys():
        #    sample_df = sig_df_dict[sample]
        #    if df_out is None:
        #        df_out = sample_df
        #    else:
        #        df_out.append(sample_df, ignore_index=True)
        #
        ## load background
        # bkg_df_dict = dict()
        # for sample in ic.bkg_list:
        #    sample_path = save_dir / f"{sample}.feather"
        #    sample_df = pd.read_feather(sample_path)[ic.selected_features + ["weight"]#]
        #    bkg_df_dict[sample] = sample_df

        # if ic.reset_feature:
        #    ref_df = array_utils.merge_samples_df(
        #        sig_df_dict, [ic.reset_feature_name], array_key=ic.sig_key
        #    )
        #    ref_array = ref_df[ic.reset_feature_name]
        #    ref_weight = ref_df["weight"]
        #    self.reweight_df_dict(
        #        "xb",
        #        bkg_df_dict,
        #        reset_mass=True,
        #        ref_array=ref_array,
        #        ref_weight=ref_weight,
        #    )
        # else:
        #    self.reweight_df_dict("xb", bkg_df_dict)
        ## merge
        # df_out = pd.concat(
        #    list(sig_df_dict.values()) + list(bkg_df_dict.values()), ignore_index=True
        # )
        ## tag train/test
        # logger.debug("tagging train/test")
        # sss = StratifiedShuffleSplit(n_splits=1)
        # y_arr = df_out["y"].values
        # for train_index, _ in sss.split(y_arr, y_arr):
        #    break
        # df_out["is_train"] = False
        # df_out.loc[train_index, "is_train"] = True
        save_path = save_dir / "train.feather"

        return pd.read_feather(save_path)
    """

    def get_raw(
        self,
        input_type,
        array_key="all",
        add_validation_features=False,
        part_features=[],
        include_weight=True,
    ) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """Loads necessary inputs without preprocessing

        Args:
            input_type: could be 'xs', 'xb', 'xd
            array_key: sample name to be loaded or use 'all' or 'all_norm' to load all
            add_validation_features: additional validation features to be loaded

        """
        # get list of needed features
        features = list()
        if not part_features:
            features += self._ic.selected_features
            if add_validation_features:
                features += self._ic.validation_features
        else:
            features += part_features
        if include_weight:
            features += ["weight"]
        features = list(set(features))  # remove duplicates
        # get input array dict
        array_dict = self.get_input_array_dict(
            input_type, part_features=features, include_weight=include_weight
        )
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
            sample_df = pd.DataFrame(array_dict[sample], dtype=np.float32)
            df_dict[sample] = sample_df

        return features, df_dict

    def get_raw_merged(
        self,
        input_type,
        features=[],
        array_key="all",
        add_validation_features=False,
        remove_duplicated_col=True,
        include_weight=True,
    ) -> pd.DataFrame:
        df_features, df_dict = self.get_raw(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
            part_features=features,
            include_weight=include_weight,
        )
        if not features:
            features = df_features
        inputs_df = array_utils.merge_samples_df(df_dict, features, array_key=array_key)
        if remove_duplicated_col:
            inputs_df = inputs_df.loc[:, ~inputs_df.columns.duplicated()]

        return inputs_df

    def get_reshape(
        self,
        input_type: str,
        array_key: str = "all",
        add_validation_features=False,
        part_features=[],
        include_weight=True,
    ) -> Dict[str, pd.DataFrame]:
        """Normalizes input distributions"""
        df_features, df_dict = self.get_raw(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
            part_features=part_features,
            include_weight=include_weight,
        )
        # check missing norm parameters
        missing_norms = []
        norm_alias = self._ic.feature_norm_alias.get_config_dict()
        for feature in df_features:
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
            logger.debug(f"Recalculating normalization parameters for {missing_norms}")
            self.update_norm_dict(missing_norms)
        # inputs pre-processing
        for sample_df in df_dict.values():
            for feature in df_features:
                if feature == "weight":
                    continue
                elif feature in self._norm_dict:
                    f_mean = self._norm_dict[feature]["mean"]
                    f_var = self._norm_dict[feature]["variance"]
                    sample_df[feature] = (sample_df[feature] - f_mean) / np.sqrt(f_var)
                else:
                    logger.debug(
                        f"{feature} is not in self._norm_dict, ignoring reshape process."
                    )
        return df_features, df_dict

    def get_reshape_merged(
        self,
        input_type,
        features=None,
        array_key="all",
        add_validation_features=False,
        remove_duplicated_col=True,
        include_weight=True,
    ):
        df_features, df_dict = self.get_reshape(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
            part_features=features,
            include_weight=include_weight,
        )
        if not features:
            features = df_features
        inputs_df = array_utils.merge_samples_df(df_dict, features, array_key=array_key)
        if remove_duplicated_col:
            inputs_df = inputs_df.loc[:, ~inputs_df.columns.duplicated()]

        return inputs_df

    def get_reweight(
        self,
        input_type: str,
        array_key: str = "all",
        add_validation_features=False,
        reset_mass: bool = False,
    ) -> Tuple[list, Dict[str, pd.DataFrame]]:
        """Scales weight and resets feature distributions"""
        if reset_mass == None:
            reset_mass = self._ic.reset_feature
        df_features, df_dict = self.get_reshape(
            input_type,
            array_key=array_key,
            add_validation_features=add_validation_features,
        )

        # reset feature for pDNN if needed
        if reset_mass and (input_type == "xb" or input_type == "xd"):
            reset_feature_name = self._ic.reset_feature_name
            ref_array_df = self.get_reweight_merged(
                "xs",
                features=[reset_feature_name, "weight"],
                array_key=self._ic.sig_key,
            )
            ref_array = ref_array_df[reset_feature_name]
            ref_weight = ref_array_df["weight"]
        else:
            ref_array = None
            ref_weight = None
        df_dict = self.reweight_df_dict(
            input_type,
            df_dict,
            reset_mass=reset_mass,
            ref_array=ref_array,
            ref_weight=ref_weight,
        )
        return df_features, df_dict

    def reweight_df_dict(
        self, input_type, df_dict, reset_mass=False, ref_array=None, ref_weight=None
    ) -> Dict[str, pd.DataFrame]:
        logger.debug("@data_io.feed_box.Feedbox.reweight_df_dict")
        ic = self._job_config.input.clone()
        # reweight
        sumofweight = 1000
        if input_type == "xs":
            array_key = ic.sig_key
            sumofweight = ic.sig_sumofweight
        elif input_type == "xb":
            array_key = ic.bkg_key
            sumofweight = ic.bkg_sumofweight
        elif input_type == "xd":
            array_key = ic.data_key
            sumofweight = ic.data_sumofweight
        ## get total weight
        total_weight = 0
        for sample_df in df_dict.values():
            # get positive only weights
            sample_df["weight_p"] = sample_df["weight"]
            sample_df.loc[sample_df["weight_p"] < 0, ["weight_p"]] = 0
            total_weight += sample_df["weight"].sum()

        ## reweight sample by sample
        for sample, sample_df in df_dict.items():
            sample_sumw = sample_df["weight"].sum()
            if len(sample_df) == 0:
                logger.warn(f"Sample {sample} input is empty, skip reweighting!")
                continue
            # normalize weight
            norm_factor = 1
            if array_key == "all":
                norm_factor = sumofweight / total_weight
            elif array_key == "all_norm":
                n_samples = len(df_dict.keys())
                norm_factor = (sumofweight / n_samples) / sample_sumw
            elif array_key == sample:
                norm_factor = sumofweight / sample_sumw
            else:
                continue
            array_utils.norm_weight(
                sample_df["weight"], norm_factor=norm_factor,
            )
        # reset feature for pDNN if needed
        if reset_mass and (input_type == "xb" or input_type == "xd"):
            for sample_key, sample_df in df_dict.items():
                reset_array = df_dict[sample_key][ic.reset_feature_name]
                array_utils.redistribute_array(reset_array, ref_array, ref_weight)
        return df_dict

    def get_reweight_merged(
        self,
        input_type,
        features=[],
        array_key="all",
        add_validation_features=False,
        reset_mass: bool = None,
        tag_sample=False,
    ):
        df_features, df_dict = self.get_reweight(
            input_type,
            array_key=array_key,
            reset_mass=reset_mass,
            add_validation_features=add_validation_features,
        )
        if not features:
            features = df_features
        if tag_sample:
            for sample, sample_df in df_dict.items():
                sample_id = self.get_sample_id(sample)
                sample_df["sample_id"] = np.full(len(sample_df), sample_id)
            features += ["sample_id"]
        inputs_df = array_utils.merge_samples_df(df_dict, features, array_key=array_key)
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

    def get_train_test_inputs(self):
        job_config = self._job_config.clone()
        ic = job_config.input.clone()
        camp_list = numpy_io.get_campaign_list(job_config)
        samples_dict = {"sig": ic.sig_list, "bkg": ic.bkg_list}
        data_dir = numpy_io.get_data_dir()

        total_train = 0
        total_test = 0
        for sample_list in samples_dict.values():
            for sample in sample_list:
                for camp in camp_list:
                    npy_in_dir = pathlib.Path(
                        f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}"
                    )
                    camp_wt = np.load(
                        npy_in_dir / f"{sample}_weight.reweight.npy"
                    ).flatten()
                    camp_tr_tag = np.load(npy_in_dir / f"{sample}.tr_tag.npy")
                    camp_train_count = np.sum(camp_tr_tag)
                    camp_test_count = len(camp_tr_tag) - camp_train_count
                    total_train += camp_train_count
                    total_test += camp_test_count

        # allocate memory
        num_col = len(ic.selected_features)
        x_train = np.zeros((total_train, num_col))
        x_test = np.zeros((total_test, num_col))
        y_train = np.zeros((total_train, 1))
        y_test = np.zeros((total_test, 1))
        wt_train = np.zeros(total_train)
        wt_test = np.zeros(total_test)

        train_count = 0
        test_count = 0
        for sample_list in samples_dict.values():
            for sample in sample_list:
                for camp in camp_list:
                    npy_in_dir = pathlib.Path(
                        f"{data_dir}/{ic.arr_path}/{ic.arr_version}/{ic.variation}/{ic.channel}/{camp}/{ic.region}"
                    )
                    camp_x = None
                    for feature in ic.selected_features:
                        feature_arr = np.load(
                            npy_in_dir / f"{sample}_{feature}.reshape.npy",
                            mmap_mode="r",
                        ).reshape((-1, 1))
                        if camp_x is None:
                            camp_x = feature_arr
                        else:
                            camp_x = np.append(camp_x, feature_arr, axis=1)
                    camp_y = np.load(npy_in_dir / f"{sample}.y.npy", mmap_mode="r")
                    if camp_y.ndim == 1:
                        camp_y = camp_y.reshape((-1, 1))
                    camp_wt = np.load(
                        npy_in_dir / f"{sample}_weight.reweight.npy"
                    ).flatten()
                    camp_tr_tag = np.load(npy_in_dir / f"{sample}.tr_tag.npy")

                    camp_train_count = np.sum(camp_tr_tag)
                    camp_test_count = len(camp_tr_tag) - camp_train_count

                    print("#### camp_x.shape", camp_x.shape)
                    print("#### camp_y.shape", camp_y.shape)
                    print("#### camp_wt.shape", camp_wt.shape)
                    print("#### camp_tr_tag.shape", camp_tr_tag.shape)

                    train_index = np.argwhere(camp_tr_tag == 1).flatten()
                    test_index = np.argwhere(camp_tr_tag == 0).flatten()
                    print("#### train_index.shape", train_index.shape)
                    # x_train = np.concatenate((x_train, camp_x[train_index]), axis=0)
                    # x_test = np.concatenate((x_test, camp_x[test_index]), axis=0)
                    # y_train = np.concatenate((y_train, camp_y[train_index]), axis=0)
                    # y_test = np.concatenate((y_test, camp_y[test_index]), axis=0)
                    # wt_train = np.concatenate((wt_train, camp_wt[train_index]), #axis=0)
                    # wt_test = np.concatenate((wt_test, camp_wt[test_index]), axis=0)

                    x_train[train_count : train_count + camp_train_count, :] = camp_x[
                        train_index
                    ]
                    x_test[test_count : test_count + camp_test_count, :] = camp_x[
                        test_index
                    ]
                    y_train[train_count : train_count + camp_train_count, :] = camp_y[
                        train_index
                    ]
                    y_test[test_count : test_count + camp_test_count, :] = camp_y[
                        test_index
                    ]
                    wt_train[train_count : train_count + camp_train_count] = camp_wt[
                        train_index
                    ]
                    wt_test[test_count : test_count + camp_test_count] = camp_wt[
                        test_index
                    ]

                    train_count += camp_train_count
                    test_count += camp_test_count
        print("#### train count", train_count)
        print("#### test count", test_count)
        print("#### x_train shape", x_train.shape)
        print("#### y_train shape", y_train.shape)
        print("#### wt_train shape", wt_train.shape)
        print("#### x_test shape", x_test.shape)
        print("#### y_test shape", y_test.shape)
        print("#### wt_test shape", wt_test.shape)

        input_dict = dict()
        input_dict["x_train"] = x_train
        input_dict["x_test"] = x_test
        input_dict["y_train"] = y_train
        input_dict["y_test"] = y_test
        input_dict["wt_train"] = wt_train
        input_dict["wt_test"] = wt_test
        return input_dict

    def get_train_test_df(
        self,
        sig_key: str = "all",
        bkg_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        jump_reweight: bool = False,
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
            sig_df = self.get_reweight_merged("xs", array_key=sig_key, tag_sample=True)
            # load bkg
            bkg_df = self.get_reweight_merged(
                "xb", array_key=bkg_key, reset_mass=reset_mass, tag_sample=True,
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
            y_arr = sample_df["y"].values
            # Note that providing y is sufficient to generate the splits and
            # hence np.zeros(n_samples) may be used as a placeholder for X
            # instead of actual training data.
            for train_index, _ in sss.split(y_arr, y_arr):
                break
            sample_df["is_train"] = np.full(len(sample_df), False)
            sample_df.loc[train_index, "is_train"] = True
            return sample_df

    """
    def get_train_test_df_multi_nodes(
        self,
        sig_key: str = "all",
        multi_class_bkgs: List[str] = [],
        reset_mass: Optional[bool] = None,
        output_keys: List[str] = [],
    ) -> Dict[str, np.ndarray]:
        # TODO: need to rewrite
        pass
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

    # def load_norm_dict(self, path):
    #    norm_path = pathlib.Path(path)
    #    if norm_path.exists():
    #        self._norm_dict = yaml.load(norm_path)

    def tag_signals(self, df_dict: Dict[str, pd.DataFrame]):
        logger.debug("@ data_io.feed_box.Feedbox.tag_signals")
        for sample, sample_df in df_dict.items():
            sample_df["sample_name"] = sample
            sample_df["is_sig"] = True
            sample_df["is_mc"] = True
            sample_df["y"] = np.int32(1)

    def tag_backgrounds(self, df_dict: Dict[str, pd.DataFrame]):
        logger.debug("@ data_io.feed_box.Feedbox.tag_backgrounds")
        for sample, sample_df in df_dict.items():
            sample_df["sample_name"] = sample
            sample_df["is_sig"] = False
            sample_df["is_mc"] = True
            sample_df["y"] = np.int32(0)

    def update_norm_dict(self, features: Optional[List[str]] = None):
        logger.debug("@ data_io.feed_box.update_norm_dict")
        feature_list = list()
        if features is not None:
            feature_list = features
        else:
            feature_list = self._ic.selected_features + self._ic.validation_features
        feature_list = list(set().union(feature_list))  # remove duplicates
        weight_array = self.get_raw_merged(
            "xb", features=["weight"], array_key=self._job_config.input.bkg_key
        )["weight"].values
        for feature in feature_list:
            feature_array = self.get_raw_merged(
                "xb",
                features=[feature],
                array_key=self._job_config.input.bkg_key,
                include_weight=False,
            )[feature].values
            mean = np.average(feature_array, weights=weight_array)
            variance = np.average((feature_array - mean) ** 2, weights=weight_array)
            self._norm_dict[feature] = {"mean": mean, "variance": variance}
            logger.info(
                f"Recalculated norm factors for feature {feature} mean: {mean}, variance: {variance}"
            )

