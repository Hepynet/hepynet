import csv
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from hepynet.common import common_utils

logger = logging.getLogger("hepynet")


def calculate_asimov(sig, bkg):
    return math.sqrt(2 * ((sig + bkg) * math.log(1 + sig / bkg) - sig))


def calculate_significance(sig, bkg, sig_total=None, bkg_total=None, algo="asimov"):
    """Returns asimov significance"""
    # check input
    if sig <= 0 or bkg <= 0 or sig_total <= 0 or bkg_total <= 0:
        logger.warning(
            "non-positive value found during significance calculation, using default value 0."
        )
        return 0
    if "_rel" in algo:
        if not sig_total:
            logger.error(
                "sig_total or bkg_total value is not specified to calculate relative type significance, please check input."
            )
        if not bkg_total:
            logger.error(
                "sig_total or bkg_total value is not specified to calculate relative type significance, please check input."
            )
    # calculation
    if algo == "asimov":
        return calculate_asimov(sig, bkg)
    elif algo == "s_b":
        return sig / bkg
    elif algo == "s_sqrt_b":
        return sig / math.sqrt(bkg)
    elif algo == "s_sqrt_sb":
        return sig / math.sqrt(sig + bkg)
    elif algo == "asimov_rel":
        return calculate_asimov(sig, bkg) / calculate_asimov(sig_total, bkg_total)
    elif algo == "s_b_rel":
        return (sig / sig_total) / (bkg / bkg_total)
    elif algo == "s_sqrt_b_rel":
        return (sig / sig_total) / math.sqrt(bkg / bkg_total)
    elif algo == "s_sqrt_sb_rel":
        return (sig / sig_total) / math.sqrt((bkg + sig) / (sig_total + bkg_total))
    else:
        logger.warning("Unrecognized significance algorithm, will use default 'asimov'")
        return calculate_asimov(sig, bkg)


def get_significances(
    model_wrapper,
    sig_key,
    bkg_key,
    significance_algo="asimov",
    multi_class_cut_branch=0,
):
    """Gets significances scan arrays.
    
    Return:
        Tuple of 4 arrays: (
            threshold array,
            significances,
            sig_total_weight_above_threshold,
            bkg_total_weight_above_threshold,
            )
    
    """
    feedbox = model_wrapper.feedbox
    # prepare signal
    sig_arr, sig_weights = feedbox.get_reshape("xs", array_key=sig_key)
    sig_predictions = model_wrapper.get_model().predict(sig_arr)
    if sig_predictions.ndim == 2:
        sig_predictions = sig_predictions[:, multi_class_cut_branch]
    # prepare background
    bkg_arr, bkg_weights = feedbox.get_reshape("xb", array_key=bkg_key)
    bkg_predictions = model_wrapper.get_model().predict(bkg_arr)
    if bkg_predictions.ndim == 2:
        bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
    # prepare thresholds
    bin_array = np.array(range(-1000, 1000))
    thresholds = 1.0 / (1.0 + 1.0 / np.exp(bin_array * 0.02))
    thresholds = np.insert(thresholds, 0, 0)
    # scan
    significances = []
    plot_thresholds = []
    sig_above_threshold = []
    bkg_above_threshold = []
    total_sig_weight = np.sum(sig_weights)
    total_bkg_weight = np.sum(bkg_weights)
    for dnn_cut in thresholds:
        sig_ids_passed = sig_predictions > dnn_cut
        total_sig_weights_passed = np.sum(sig_weights[sig_ids_passed])
        bkg_ids_passed = bkg_predictions > dnn_cut
        total_bkg_weights_passed = np.sum(bkg_weights[bkg_ids_passed])
        if total_bkg_weights_passed > 0 and total_sig_weights_passed > 0:
            plot_thresholds.append(dnn_cut)
            current_significance = calculate_significance(
                total_sig_weights_passed,
                total_bkg_weights_passed,
                sig_total=total_sig_weight,
                bkg_total=total_bkg_weight,
                algo=significance_algo,
            )
            # current_significance = total_sig_weights_passed / total_bkg_weights_passed
            significances.append(current_significance)
            sig_above_threshold.append(total_sig_weights_passed)
            bkg_above_threshold.append(total_bkg_weights_passed)
    total_sig_weight = np.sum(sig_weights)
    total_bkg_weight = np.sum(bkg_weights)
    return (plot_thresholds, significances, sig_above_threshold, bkg_above_threshold)


def plot_significance_scan(
    ax,
    model_wrapper,
    sig_key,
    bkg_key,
    save_dir=".",
    significance_algo="asimov",
    suffix="",
) -> None:
    """Shows significance change with threshold.

    Note:
        significance is calculated by s/sqrt(b)
    """
    logger.info("Plotting significance scan.")
    (
        plot_thresholds,
        significances,
        sig_above_threshold,
        bkg_above_threshold,
    ) = get_significances(
        model_wrapper, sig_key, bkg_key, significance_algo=significance_algo
    )

    significances_no_nan = np.nan_to_num(significances)
    max_significance = np.amax(significances_no_nan)
    index = np.argmax(significances_no_nan)
    max_significance_threshold = plot_thresholds[index]
    max_significance_sig_total = sig_above_threshold[index]
    max_significance_bkg_total = bkg_above_threshold[index]
    total_sig_weight = sig_above_threshold[0]
    total_bkg_weight = bkg_above_threshold[0]
    # make plots
    # plot original significance
    original_significance = calculate_significance(
        total_sig_weight,
        total_bkg_weight,
        sig_total=total_sig_weight,
        bkg_total=total_bkg_weight,
        algo=significance_algo,
    )
    ax.axhline(y=original_significance, color="grey", linestyle="--")
    # significance scan curve
    ax.plot(plot_thresholds, significances_no_nan, color="r", label=significance_algo)
    # signal/background events scan curve
    ax2 = ax.twinx()
    max_sig_events = sig_above_threshold[0]
    max_bkg_events = bkg_above_threshold[0]
    sig_eff_above_threshold = np.array(sig_above_threshold) / max_sig_events
    bkg_eff_above_threshold = np.array(bkg_above_threshold) / max_bkg_events
    ax2.plot(plot_thresholds, sig_eff_above_threshold, color="orange", label="sig")
    ax2.plot(plot_thresholds, bkg_eff_above_threshold, color="blue", label="bkg")
    ax2.set_ylabel("sig(bkg) ratio after cut")
    # reference threshold
    ax.axvline(x=max_significance_threshold, color="green", linestyle="-.")
    # more infomation
    content = (
        "best threshold:"
        + str(common_utils.get_significant_digits(max_significance_threshold, 6))
        + "\nmax significance:"
        + str(common_utils.get_significant_digits(max_significance, 6))
        + "\nbase significance:"
        + str(common_utils.get_significant_digits(original_significance, 6))
        + "\nsig events above threshold:"
        + str(common_utils.get_significant_digits(max_significance_sig_total, 6))
        + "\nbkg events above threshold:"
        + str(common_utils.get_significant_digits(max_significance_bkg_total, 6))
    )
    ax.text(
        0.05,
        0.9,
        content,
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="green",
        fontsize=12,
    )
    # set up plot
    ax.set_title("significance scan")
    ax.set_xscale("logit")
    ax.set_xlabel("DNN score threshold")
    ax.set_ylabel("significance")
    ax.set_ylim(bottom=0)
    ax.locator_params(nbins=10, axis="x")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="center left")
    ax2.legend(loc="center right")
    # ax2.set_yscale("log")
    # collect meta data
    model_wrapper.original_significance = original_significance
    model_wrapper.max_significance = max_significance
    model_wrapper.max_significance_threshold = max_significance_threshold
    # make extra cut table 0.1, 0.2 ... 0.8, 0.9
    # make table for different DNN cut scores
    save_path = save_dir + "/scan_DNN_cut" + suffix + ".csv"
    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        row_list = [
            [
                "DNN cut",
                "sig events",
                "sig efficiency",
                "bkg events",
                "bkg efficiency",
                "significance",
            ]
        ]
        for index in range(1, 100):
            dnn_cut = (100 - index) / 100.0
            threshold_id = (np.abs(np.array(plot_thresholds) - dnn_cut)).argmin()
            sig_events = sig_above_threshold[threshold_id]
            sig_eff = sig_eff_above_threshold[threshold_id]
            bkg_events = bkg_above_threshold[threshold_id]
            bkg_eff = bkg_eff_above_threshold[threshold_id]
            significance = significances[threshold_id]
            new_row = [dnn_cut, sig_events, sig_eff, bkg_events, bkg_eff, significance]
            row_list.append(new_row)
        row_list.append([""])
        row_list.append(
            [
                "total sig",
                max_sig_events,
                "total bkg",
                max_bkg_events,
                "base significance",
                original_significance,
            ]
        )
        writer.writerows(row_list)
    # make table for different sig efficiency
    save_path = save_dir + "/scan_sig_eff" + suffix + ".csv"
    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        row_list = [
            [
                "DNN cut",
                "sig events",
                "sig efficiency",
                "bkg events",
                "bkg efficiency",
                "significance",
            ]
        ]
        for index in range(1, 100):
            sig_eff_cut = (100 - index) / 100.0
            threshold_id = (
                np.abs(np.array(sig_eff_above_threshold) - sig_eff_cut)
            ).argmin()
            dnn_cut = plot_thresholds[threshold_id]
            sig_events = sig_above_threshold[threshold_id]
            sig_eff = sig_eff_cut
            bkg_events = bkg_above_threshold[threshold_id]
            bkg_eff = bkg_eff_above_threshold[threshold_id]
            significance = significances[threshold_id]
            new_row = [dnn_cut, sig_events, sig_eff, bkg_events, bkg_eff, significance]
            row_list.append(new_row)
        row_list.append([""])
        row_list.append(
            [
                "total sig",
                max_sig_events,
                "total bkg",
                max_bkg_events,
                "base significance",
                original_significance,
            ]
        )
        writer.writerows(row_list)
    # make table for different bkg efficiency
    save_path = save_dir + "/scan_bkg_eff" + suffix + ".csv"
    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        row_list = [
            [
                "DNN cut",
                "sig events",
                "sig efficiency",
                "bkg events",
                "bkg efficiency",
                "significance",
            ]
        ]
        for index in range(1, 100):
            bkg_eff_cut = (100 - index) / 100.0
            threshold_id = (
                np.abs(np.array(bkg_eff_above_threshold) - bkg_eff_cut)
            ).argmin()
            dnn_cut = plot_thresholds[threshold_id]
            sig_events = sig_above_threshold[threshold_id]
            sig_eff = sig_eff_above_threshold[threshold_id]
            bkg_events = bkg_above_threshold[threshold_id]
            bkg_eff = bkg_eff_cut
            significance = significances[threshold_id]
            new_row = [dnn_cut, sig_events, sig_eff, bkg_events, bkg_eff, significance]
            row_list.append(new_row)
        row_list.append([""])
        row_list.append(
            [
                "total sig",
                max_sig_events,
                "total bkg",
                max_bkg_events,
                "base significance",
                original_significance,
            ]
        )
        writer.writerows(row_list)
