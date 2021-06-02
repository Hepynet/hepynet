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
    model,
    epoch_report: dict,
    metrics: List[str] = list(),
    metrics_weighted: List[str] = list(),
    x_val_unreset: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    wt_val: Optional[np.ndarray] = None,
    last_auc_unreset: Optional[float] = None,
):
    # calculate customized metrics

    # calculate customized weighted metrics
    if "auc_unreset" in metrics_weighted:
        y_pred_unreset = model.predict(x_val_unreset)
        auc_unreset = roc_auc_score(
            y_val, y_pred_unreset, sample_weight=wt_val
        )
        epoch_report["auc_unreset"] = auc_unreset
    if "auc_unreset_improvement" in metrics_weighted:
        if ("auc_unreset" in metrics_weighted) and (
            last_auc_unreset is not None
        ):
            auc_unreset_improvement = auc_unreset - last_auc_unreset
            epoch_report["auc_unreset_improvement"] = auc_unreset_improvement
            last_auc_unreset = auc_unreset
        else:
            logger.error(
                f"auc_unrest not in custom_tune_metrics_weighted, cant' calculate auc_unreset_improvement!"
            )
    if "limit" in metrics_weighted:
        pass


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
        stop = getattr(stopper, tuner.stopper_class)(**tuner.stopper)
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
