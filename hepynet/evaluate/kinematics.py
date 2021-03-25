from hepynet.evaluate import evaluate_utils
import logging
import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hepynet.common import config_utils
from hepynet.data_io import array_utils
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def plot_correlation_matrix(model_wrapper, save_dir="."):
    save_dir = pathlib.Path(save_dir)
    save_dir = save_dir.joinpath("kinematics")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(ncols=2, figsize=(16.667, 8.333))
    ax[0].set_title("bkg correlation")
    corr_matrix_dict = model_wrapper.get_corrcoef()
    paint_correlation_matrix(ax[0], corr_matrix_dict, matrix_key="bkg")
    ax[1].set_title("sig correlation")
    paint_correlation_matrix(ax[1], corr_matrix_dict, matrix_key="sig")
    fig_save_path = save_dir.joinpath("correlation_matrix.png")
    logger.debug(f"Save correlation matrix to: {fig_save_path}")
    fig.savefig(fig_save_path)


def paint_correlation_matrix(ax, corr_matrix_dict, matrix_key="bkg"):
    # Get matrix
    corr_matrix = corr_matrix_dict[matrix_key]
    labels = corr_matrix_dict["labels"]
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )


def plot_input(
    model_wrapper: hep_model.Model_Base, job_config, save_dir=None, show_reshaped=False,
):
    """Plots input distributions comparision plots for sig/bkg/data"""
    logger.info("Plotting input distributions.")
    # setup config
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_kine_study
    # prepare
    feedbox = model_wrapper.get_feedbox()
    if show_reshaped:  # validation features not supported in get_reshape yet
        plot_feature_list = ic.selected_features
        bkg_array, bkg_fill_weights = feedbox.get_reshape_merged(
            "xb", array_key=ic.bkg_key
        )
        sig_array, sig_fill_weights = feedbox.get_reshape_merged(
            "xs", array_key=ic.sig_key
        )
    else:
        plot_feature_list = array_utils.merge_select_val_features(
            ic.selected_features, ic.validation_features
        )
        bkg_array, bkg_fill_weights = feedbox.get_raw_merged(
            "xb", array_key=ic.bkg_key, add_validation_features=True,
        )
        sig_array, sig_fill_weights = feedbox.get_raw_merged(
            "xs", array_key=ic.sig_key, add_validation_features=True,
        )
    # normalize
    if plot_cfg.density:
        bkg_fill_weights = bkg_fill_weights / np.sum(bkg_fill_weights)
        sig_fill_weights = sig_fill_weights / np.sum(sig_fill_weights)
    # plot
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for feature_id, feature in enumerate(plot_feature_list):
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_fill_array = np.reshape(bkg_array[:, feature_id], (-1, 1))
        sig_fill_array = np.reshape(sig_array[:, feature_id], (-1, 1))
        plot_input_plt(
            feature,
            sig_fill_array,
            sig_fill_weights,
            bkg_fill_array,
            bkg_fill_weights,
            plot_config=plot_cfg,
            save_dir=save_dir,
        )


def plot_input_dnn(
    model_wrapper: hep_model.Model_Base,
    job_config,
    dnn_cut=None,
    multi_class_cut_branch=0,
    save_dir=None,
    compare_cut_sb_separated=False,
):
    """Plots input distributions comparision plots with DNN cuts applied"""
    logger.info("Plotting input distributions with DNN cuts applied.")
    # prepare
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_cut_kine_study
    feedbox = model_wrapper.get_feedbox()
    # get fill weights with dnn cut
    plot_feature_list = ic.selected_features + ic.validation_features
    bkg_array, bkg_fill_weights = feedbox.get_raw_merged(
        "xb",
        features=plot_feature_list,
        array_key=ic.bkg_key,
        add_validation_features=True,
    )
    sig_array, sig_fill_weights = feedbox.get_raw_merged(
        "xs",
        features=plot_feature_list,
        array_key=ic.sig_key,
        add_validation_features=True,
    )
    # normalize
    if plot_cfg.density:
        bkg_fill_weights = bkg_fill_weights / np.sum(bkg_fill_weights)
        sig_fill_weights = sig_fill_weights / np.sum(sig_fill_weights)
    # plot kinematics with dnn cuts
    if dnn_cut < 0 or dnn_cut > 1:
        logger.error(f"DNN cut {dnn_cut} is out of range [0, 1]!")
        return
    # prepare signal
    sig_selected_arr, _ = feedbox.get_reweight_merged(
        "xs", array_key=ic.sig_key, reset_mass=False
    )
    sig_predictions, _, _ = evaluate_utils.k_folds_predict(
        model_wrapper.get_model(), sig_selected_arr
    )
    if sig_predictions.ndim == 2:
        sig_predictions = sig_predictions[:, multi_class_cut_branch]
    sig_cut_index = array_utils.get_cut_index(sig_predictions, [dnn_cut], ["<"])
    sig_fill_weights_dnn = sig_fill_weights.copy()
    sig_fill_weights_dnn[sig_cut_index] = 0
    # prepare background
    bkg_selected_arr, _ = feedbox.get_reweight_merged(
        "xb", array_key=ic.bkg_key, reset_mass=False
    )
    bkg_predictions, _, _ = evaluate_utils.k_folds_predict(
        model_wrapper.get_model(), bkg_selected_arr
    )
    if bkg_predictions.ndim == 2:
        bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
    bkg_cut_index = array_utils.get_cut_index(bkg_predictions, [dnn_cut], ["<"])
    bkg_fill_weights_dnn = bkg_fill_weights.copy()
    bkg_fill_weights_dnn[bkg_cut_index] = 0
    # normalize weights for density plots
    if plot_cfg.density:
        bkg_fill_weights_dnn = bkg_fill_weights_dnn / np.sum(bkg_fill_weights)
        sig_fill_weights_dnn = sig_fill_weights_dnn / np.sum(sig_fill_weights)
    # plot
    for feature_id, feature in enumerate(plot_feature_list):
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_fill_array = np.reshape(bkg_array[:, feature_id], (-1, 1))
        sig_fill_array = np.reshape(sig_array[:, feature_id], (-1, 1))
        plot_inputs_cut_dnn(
            feature,
            sig_fill_array,
            sig_fill_weights,
            sig_fill_weights_dnn,
            bkg_fill_array,
            bkg_fill_weights,
            bkg_fill_weights_dnn,
            plot_config=plot_cfg,
            save_dir=save_dir,
        )


def plot_hist_plt(
    ax, values, weights, plot_cfg,
):
    ax.hist(
        values,
        bins=plot_cfg.bins,
        range=plot_cfg.range,
        weights=weights,
        # histtype=plot_cfg.histtype,
        facecolor=plot_cfg.facecolor,
        edgecolor=plot_cfg.edgecolor,
        label=plot_cfg.label,
        histtype=plot_cfg.histtype,
        # alpha=plot_cfg.alpha,
        # hatch=plot_cfg.hatch,
    )


def plot_hist_ratio_plt(
    ax,
    numerator_values,
    numerator_weights,
    denominator_values,
    denominator_weights,
    plot_cfg,
):
    numerator_ys, bin_edges = np.histogram(
        numerator_values,
        bins=plot_cfg.bins,
        range=plot_cfg.range,
        weights=numerator_weights.reshape((-1, 1)),
    )  # np.histogram requires "weights should have the same shape as a."
    denominator_ys, _ = np.histogram(
        denominator_values,
        bins=plot_cfg.bins,
        range=plot_cfg.range,
        weights=denominator_weights.reshape((-1, 1)),
    )  # same reason to reshape
    # Only plot ratio when bin is not 0.
    bin_centers = np.array([])
    bin_ys = np.array([])
    for i, (y1, y2) in enumerate(zip(numerator_ys, denominator_ys)):
        if y1 != 0:
            ele_center = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1])])
            bin_centers = np.concatenate((bin_centers, ele_center))
            ele_y = np.array([y1 / y2])
            bin_ys = np.concatenate((bin_ys, ele_y))
    # plot ratio
    bin_size = bin_edges[1] - bin_edges[0]
    ax.set_ylim([0, 1.3])
    ax.errorbar(
        bin_centers,
        bin_ys,
        xerr=bin_size / 2.0,
        yerr=None,
        fmt="_",
        color=plot_cfg.edgecolor,
        markerfacecolor=plot_cfg.edgecolor,
        markeredgecolor=plot_cfg.edgecolor,
    )


def plot_input_plt(
    feature,
    sig_fill_array,
    sig_fill_weights,
    bkg_fill_array,
    bkg_fill_weights,
    plot_config=config_utils.Hepy_Config_Section({}),
    save_dir=".",
):
    # prepare config of chosen feature
    feature_cfg = plot_config.clone()
    if feature in plot_config.__dict__.keys():
        feature_cfg_tmp = getattr(plot_config, feature)
        feature_cfg.update(feature_cfg_tmp.get_config_dict())
    # make plot
    if plot_config.separate_sig_bkg:
        # plot bkg
        fig, ax = plt.subplots()
        feature_cfg.update(
            {
                "facecolor": plot_config.bkg_color,
                "edgecolor": "black",
                "label": "background",
            }
        )
        plot_hist_plt(
            ax, bkg_fill_array, bkg_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature)
        fig.savefig(f"{save_dir}/{feature}_bkg.{plot_config.save_format}")
        plt.close()
        # plot sig
        fig, ax = plt.subplots()
        feature_cfg.update(
            {
                # "facecolor": plot_config.sig_color,
                "facecolor": "none",
                "edgecolor": "red",
                "label": "signal",
            }
        )
        plot_hist_plt(
            ax, sig_fill_array, sig_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature)
        fig.savefig(f"{save_dir}/{feature}_sig.{plot_config.save_format}")
        plt.close()
    else:
        fig, ax = plt.subplots()
        # plot bkg
        feature_cfg.update(
            {
                "facecolor": plot_config.bkg_color,
                "edgecolor": "black",
                "label": "background",
            }
        )
        plot_hist_plt(
            ax, bkg_fill_array, bkg_fill_weights, feature_cfg,
        )
        # plot sig
        feature_cfg.update(
            {
                # "facecolor": plot_config.sig_color,
                "facecolor": "none",
                "edgecolor": "red",
                "label": "signal",
            }
        )
        plot_hist_plt(
            ax, sig_fill_array, sig_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature)
        fig.savefig(f"{save_dir}/{feature}.{plot_config.save_format}")
        plt.close()


def plot_inputs_cut_dnn(
    feature,
    sig_fill_array,
    sig_fill_weights,
    sig_fill_weights_dnn,
    bkg_fill_array,
    bkg_fill_weights,
    bkg_fill_weights_dnn,
    plot_config={},
    save_dir=".",
):
    feature_cfg = plot_config.clone()
    if feature in plot_config.__dict__.keys():
        feature_cfg_tmp = getattr(plot_config, feature)
        feature_cfg.update(feature_cfg_tmp.get_config_dict())
    # plot sig
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0)
    ax0 = plt.subplot(gs[0])
    feature_cfg.update(
        {
            "facecolor": "none",
            "edgecolor": plot_config.sig_color,
            "label": "signal-not-cut",
            "hatch": "///",
        }
    )
    plot_hist_plt(
        ax0, sig_fill_array, sig_fill_weights, feature_cfg,
    )
    feature_cfg.update(
        {
            "facecolor": plot_config.sig_color,
            "edgecolor": "none",
            "label": "signal-cut-dnn",
            "hatch": None,
        }
    )
    plot_hist_plt(
        ax0, sig_fill_array, sig_fill_weights_dnn, feature_cfg,
    )
    ax0.legend(loc="upper right")
    xlim = ax0.get_xlim()
    ax1 = plt.subplot(gs[1])
    ax1.set_xlim(xlim)
    feature_cfg.update(
        {
            "facecolor": "none",
            "edgecolor": plot_config.sig_color,
            "label": None,
            "hatch": None,
        }
    )
    plot_hist_ratio_plt(
        ax1,
        sig_fill_array,
        sig_fill_weights_dnn,
        sig_fill_array,
        sig_fill_weights,
        feature_cfg,
    )
    fig.suptitle(feature, fontsize=16)
    fig.savefig(f"{save_dir}/{feature}_sig.{plot_config.save_format}")
    plt.close()

    # plot bkg
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0)
    ax0 = plt.subplot(gs[0])
    feature_cfg.update(
        {
            "facecolor": "none",
            "edgecolor": plot_config.bkg_color,
            "label": "background-no-cut",
            "hatch": "///",
        }
    )
    plot_hist_plt(
        ax0, bkg_fill_array, bkg_fill_weights, feature_cfg,
    )
    feature_cfg.update(
        {
            "facecolor": plot_config.bkg_color,
            "edgecolor": "none",
            "label": "background-cut-dnn",
            "hatch": "///",
        }
    )
    plot_hist_plt(
        ax0, bkg_fill_array, bkg_fill_weights_dnn, feature_cfg,
    )
    ax0.legend(loc="upper right")
    ax0.set_xticks([])
    xlim = ax0.get_xlim()
    ax1 = plt.subplot(gs[1])
    ax1.set_xlim(xlim)
    feature_cfg.update(
        {
            "facecolor": "none",
            "edgecolor": plot_config.bkg_color,
            "label": None,
            "hatch": None,
        }
    )
    plot_hist_ratio_plt(
        ax1,
        bkg_fill_array,
        bkg_fill_weights_dnn,
        bkg_fill_array,
        bkg_fill_weights,
        feature_cfg,
    )
    fig.suptitle(feature, fontsize=16)
    fig.savefig(f"{save_dir}/{feature}_bkg.{plot_config.save_format}")
    plt.close()

