# -*- coding: utf-8 -*-
import logging
import math
import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pyhf
import ray
import yaml
from ray import tune
from ray.tune import schedulers, stopper
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import hepynet.common.hepy_type as ht
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def calculate_custom_tune_metrics(
    model, config, epoch_report: dict, last_report: Optional[dict] = None,
):
    metrics = config["custom_tune_metrics"]
    metrics_weighted = config["custom_tune_metrics_weighted"]
    input_dir = pathlib.Path(config["input_dir"])
    if last_report is None:
        last_report = dict()

    # calculate customized metrics

    # calculate customized weighted metrics
    if "auc_unreset" in metrics_weighted:
        x_val_unreset = np.load(input_dir / "x_val_unreset.npy")
        y_val = np.load(input_dir / "y_val.npy")
        wt_val = np.load(input_dir / "wt_val.npy")
        y_pred_unreset = model.predict(x_val_unreset)
        auc_unreset = roc_auc_score(
            y_val, y_pred_unreset, sample_weight=wt_val
        )
        epoch_report["auc_unreset"] = auc_unreset
    if "auc_unreset_delta" in metrics_weighted:
        if ("auc_unreset" in metrics_weighted) and (
            "auc_unreset" in last_report
        ):
            epoch_report["auc_unreset_delta"] = (
                auc_unreset - last_report["auc_unreset"]
            )
        else:
            logger.error(
                f"auc_unrest not in custom_tune_metrics_weighted, cant' calculate auc_unreset_delta!"
            )
    if "min_limit" in metrics_weighted:
        logger.debug("Calculating limit for tuning")
        limit_cfg = config["metric_min_limit"]
        x_sig = np.load(input_dir / "fit_x_unreset_sig.npy")
        x_bkg = np.load(input_dir / "fit_x_unreset_bkg.npy")
        wt_sig = np.load(input_dir / "fit_wt_sig.npy")
        wt_bkg = np.load(input_dir / "fit_wt_bkg.npy")
        fit_var_sig = np.load(input_dir / "fit_var_sig.npy")
        fit_var_bkg = np.load(input_dir / "fit_var_bkg.npy")
        y_sig = model.predict(x_sig)
        y_bkg = model.predict(x_bkg)
        limits = [100000]  # default limit set to a large number
        for cut_ratio in np.linspace(*limit_cfg["dnn_scan_space"]):
            cut = cut_ratio / 10.0
            logger.debug(f"> checking cut")
            sig_arr = fit_var_sig[np.where(y_sig.flatten() > cut)]
            sig_wt = wt_sig[np.where(y_sig.flatten() > cut)]
            bkg_arr = fit_var_bkg[np.where(y_bkg.flatten() > cut)]
            bkg_wt = wt_bkg[np.where(y_bkg.flatten() > cut)]
            # prepare limit inputs
            sig_bins, _ = np.histogram(
                sig_arr,
                bins=limit_cfg["bins"],
                range=limit_cfg["range"],
                weights=sig_wt,
            )
            bkg_bins, _ = np.histogram(
                bkg_arr,
                bins=limit_cfg["bins"],
                range=limit_cfg["range"],
                weights=bkg_wt,
            )
            logger.debug(f"> sig_bins", sig_bins)
            logger.debug(f"> bkg_bins", bkg_bins)
            # calculate limit
            spec = {
                "channels": [
                    {
                        "name": "signal",
                        "samples": [
                            {
                                "name": "sig",
                                "data": sig_bins.tolist(),
                                "modifiers": [
                                    {
                                        "name": "mu",
                                        "type": "normfactor",
                                        "data": None,
                                    }
                                ],
                            },
                            {
                                "name": "bkg",
                                "data": bkg_bins.tolist(),
                                "modifiers": [],
                            },
                        ],
                    },
                ]
            }
            pdf = pyhf.Model(spec)
            init_pars = pdf.config.suggested_init()
            data = pdf.expected_data(init_pars)
            pdf.config.suggested_bounds()
            try:
                _, exp_limits, (_, _) = pyhf.infer.intervals.upperlimit(
                    data,
                    pdf,
                    np.linspace(*limit_cfg["poi_scan_space"]),
                    level=0.05,
                    return_results=True,
                )
                scan_limit = exp_limits[2]
                if np.isfinite(scan_limit):
                    limits.append(scan_limit)
                logger.debug(f"> limit {scan_limit}")
            except:
                logger.debug("> fitting failed")
                pass
        min_limit = min(limits)
        epoch_report["min_limit"] = min_limit
    if "min_limit_delta" in metrics_weighted:
        if ("min_limit" in metrics_weighted) and (
            "min_limit" in last_report
        ):
            epoch_report["min_limit_delta"] = (
                min_limit - last_report["min_limit"]
            )
        else:
            logger.error(
                f"min_limit not in custom_tune_metrics_weighted, cant' calculate min_limit_delta!"
            )


def get_hypers_tune(job_config: ht.config):
    """Gets a dict of hyperparameter (space) for auto-tuning"""
    logger.info("Collecting configs for hyperparameter tuning")
    ic = job_config.input.clone()
    rc = job_config.run.clone()
    model_cfg = job_config.tune.clone().model
    gs = get_single_hyper
    hypers = dict()
    for hyper_key in model_cfg.get_config_dict().keys():
        logger.info(f"> > Processing: {hyper_key}")
        hypers[hyper_key] = gs(getattr(model_cfg, hyper_key))
    hypers["input_dim"] = len(ic.selected_features)
    hypers["input_dir"] = str(pathlib.Path(rc.tune_input_cache).resolve())
    hypers[
        "metric_min_limit"
    ] = job_config.tune.metric_min_limit.get_config_dict()

    return hypers


def get_model_class(model_class: str):
    if model_class == "Model_Sequential_Flat":
        return hep_model.Model_Sequential_Flat
    else:
        logger.critical(f"Unknown model class: {model_class}")
        exit(1)


def get_mean_var(
    array: np.ndarray,
    axis: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
):
    """Calculate average and variance of an array."""
    average = np.average(array, axis=axis, weights=weights)
    variance = np.average((array - average) ** 2, axis=axis, weights=weights)
    if 0 in variance:
        logger.warn("Encountered 0 variance, adding shift value 0.000001")
    return average, variance + 0.000001


def get_single_hyper(hyper_config: ht.sub_config):
    """Determines one hyperparameter

    Note:
        If the hyper-parameter is a tuning dimension (with spacer specified), 
        spacer from ray.tune will be used to set up the dimension.
        If the hyper-parameter is a normal value, it will be returned directly.

    """
    if hasattr(hyper_config, "spacer"):
        paras = hyper_config.paras
        if isinstance(paras, ht.sub_config):
            paras_dict = hyper_config.paras.get_config_dict()
            return getattr(tune, hyper_config.spacer)(**paras_dict)
        else:
            logger.warn(
                f"Dimension spacer specified but invalid paras detected: {paras}"
            )
            logger.warn(f"Ignoring current dimension")
            return None
    else:
        return hyper_config


def get_train_val_indices(
    x: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    val_split: float,
    k_folds: Optional[int] = None,
):
    """Gets indices to separates train datasets to train/validation"""
    train_indices_list = list()
    validation_indices_list = list()
    if isinstance(k_folds, int) and k_folds >= 2:
        # skf = KFold(n_splits=k_folds, shuffle=True)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        for train_index, val_index in skf.split(x, y):
            np.random.shuffle(train_index)
            np.random.shuffle(val_index)
            train_indices_list.append(train_index)
            validation_indices_list.append(val_index)
    else:
        if isinstance(k_folds, int) and k_folds <= 2:
            logger.error(
                f"Invalid train.k_folds value {k_folds} detected, will not use k-fold validation"
            )
        array_len = len(wt)
        val_index = np.random.choice(
            range(array_len), int(array_len * 1.0 * val_split), replace=False
        )
        train_index = np.setdiff1d(np.array(range(array_len)), val_index)
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        train_indices_list.append(train_index)
        validation_indices_list.append(val_index)
    return train_indices_list, validation_indices_list


def merge_unequal_length_arrays(array_list: List[np.ndarray]):
    """Merges arrays with unequal length to average/min/max

    Note:
        mainly used to deal with k-fold results with early-stopping (which
        results in unequal length of results as different folds may stop at
        different epochs) enabled

    """
    folds_lengths = list()
    for single_array in array_list:
        folds_lengths.append(len(single_array))
    max_len = max(folds_lengths)

    mean = list()
    low = list()
    high = list()
    for i in range(max_len):
        sum_value = 0
        min_value = math.inf
        max_value = -math.inf
        num_values = 0
        for fold_num, single_array in enumerate(array_list):
            if i < folds_lengths[fold_num]:
                ele = single_array[i]
                sum_value += ele
                num_values += 1
                if ele < min_value:
                    min_value = ele
                if ele > max_value:
                    max_value = ele
        mean_value = sum_value / num_values
        mean.append(mean_value)
        low.append(min_value)
        high.append(max_value)
    return mean, low, high


def ray_tune(model_wrapper, job_config: ht.config, resume: bool = False):
    """Performs automatic hyper-parameters tuning with Ray"""
    # initialize
    tuner = job_config.tune.clone().tuner
    log_dir = pathlib.Path(job_config.run.save_sub_dir) / "tmp_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # set up config
    config = get_hypers_tune(job_config)
    # set up scheduler
    sched_class = getattr(schedulers, tuner.scheduler_class)
    logger.info(f"Setting up scheduler: {tuner.scheduler_class}")
    sched_config = tuner.scheduler.get_config_dict()
    sched = sched_class(**sched_config)
    # set up algorithm
    algo_class = tuner.algo_class
    logger.info(f"Setting up search algorithm: {tuner.algo_class}")
    algo_config = tuner.algo.get_config_dict()
    algo = None
    if algo_class is None:
        algo = None
    elif algo_class == "AxSearch":
        from ray.tune.suggest.ax import AxSearch

        algo = AxSearch(**algo_config)
    elif algo_class == "HyperOptSearch":
        from ray.tune.suggest.hyperopt import HyperOptSearch

        algo = HyperOptSearch(**algo_config)
    elif algo_class == "HEBOSearch":
        from ray.tune.suggest.hebo import HEBOSearch

        algo = HEBOSearch(**algo_config)
    else:
        logger.error(f"Unsupported search algorithm: {algo_class}")
        logger.info(f"Using default value None for search algorithm")
    # set stopper
    if tuner.stopper_class is None:
        stop = None
    else:
        stop_class = getattr(ray.tune.stopper, tuner.stopper_class)
        stop_config = tuner.stopper.get_config_dict()
        stop = stop_class(**stop_config)
    # set up extra run configs
    run_config = (
        tuner.run.get_config_dict()
    )  # important: convert Hepy_Config class to dict
    tune_func = getattr(hep_model, model_wrapper._tune_fun_name)

    # start tuning jobs
    if os.name == "posix":
        logger.info(f"Ignoring tune.tmp.tmp_dir setting on Unix OS")
        ray.init(**(tuner.init.get_config_dict()))
    else:
        ray.init(
            _temp_dir=str(job_config.tune.tmp_dir),
            **(tuner.init.get_config_dict()),
        )
    analysis = tune.run(
        tune_func,
        name="ray_tunes",
        stop=stop,
        search_alg=algo,
        scheduler=sched,
        config=config,
        local_dir=job_config.run.save_sub_dir,
        resume=resume,
        **run_config,
    )
    logger.info("Best hyperparameters found were:")
    print(yaml.dump(analysis.best_config))

    return analysis
