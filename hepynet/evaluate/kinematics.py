import logging
import pathlib

import atlas_mpl_style as ampl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hepynet.common import config_utils
from hepynet.data_io import array_utils, feed_box
from hepynet.evaluate import evaluate_utils
from hepynet.train import hep_model, train_utils

logger = logging.getLogger("hepynet")


def plot_correlation_matrix(df, job_config, save_dir="."):
    ic = job_config.input.clone()
    features = ic.selected_features
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_scale = len(features) / 10 + 1
    figsize = (8.333 * fig_scale, 4.167 * fig_scale)
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    # plot bkg
    ax[0].set_title("bkg correlation")
    bkg_df = array_utils.extract_bkg_df(df)
    if bkg_df.shape[0] > 1000000:
        logger.warn(
            f"Too large input detected ({bkg_df.shape[0]} rows), randomly sampling 1000000 rows for background corr_matrix calculation"
        )
        bkg_df = bkg_df.sample(n=1000000)
    bkg_matrix = array_utils.corr_matrix(
        bkg_df[features].values, bkg_df["weight"].values
    )
    paint_correlation_matrix(ax[0], bkg_matrix, features)
    # plot sig
    ax[1].set_title("sig correlation")
    sig_df = array_utils.extract_sig_df(df)
    if sig_df.shape[0] > 1000000:
        logger.warn(
            f"Too large input detected ({sig_df.shape[0]} rows), randomly sampling 1000000 rows for signal corr_matrix calculation"
        )
        sig_df = sig_df.sample(n=1000000)
    sig_matrix = array_utils.corr_matrix(
        sig_df[features].values, sig_df["weight"].values
    )
    paint_correlation_matrix(ax[1], sig_matrix, features)
    fig_save_path = save_dir / "correlation_matrix.png"
    logger.debug(f"Save correlation matrix to: {fig_save_path}")
    fig.savefig(fig_save_path)


def paint_correlation_matrix(ax, corr_matrix, labels):
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


def plot_input(df: pd.DataFrame, job_config, save_dir=None):
    """Plots input distributions comparision plots for sig/bkg/data"""
    # setup config
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_kine
    # prepare
    plot_feature_list = list(set(ic.selected_features + ic.validation_features))
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # plot
    for feature in plot_feature_list:
        feature_cfg = plot_cfg.clone()
        if feature in plot_cfg.__dict__.keys():
            feature_cfg_tmp = getattr(plot_cfg, feature)
            feature_cfg.update(feature_cfg_tmp.get_config_dict())
        # plot bkg
        bkg_df = array_utils.extract_bkg_df(df)
        bkg_wt = bkg_df["weight"].values
        fig, ax = plt.subplots()
        ax.hist(
            bkg_df[feature].values,
            weights=bkg_wt,
            label="background",
            **(feature_cfg.hist_kwargs_bkg.get_config_dict()),
        )
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.4)
        ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        fig.suptitle(feature)
        fig.savefig(f"{save_dir}/{feature}_bkg.{plot_cfg.save_format}")
        plt.close()
        # plot sig
        sig_df = array_utils.extract_sig_df(df)
        sig_wt = sig_df["weight"].values
        fig, ax = plt.subplots()
        ax.hist(
            sig_df[feature].values,
            weights=sig_wt,
            label="signal",
            **(feature_cfg.hist_kwargs_sig.get_config_dict()),
        )
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.4)
        ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.1, 0.9, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        fig.suptitle(feature)
        fig.savefig(f"{save_dir}/{feature}_sig.{plot_cfg.save_format}")
        plt.close()


def plot_input_dnn(
    model_wrapper: hep_model.Model_Base,
    df: pd.DataFrame,
    job_config,
    dnn_cut=None,
    multi_class_cut_branch=0,
    save_dir=None,
):
    """Plots input distributions comparision plots with DNN cuts applied"""
    logger.info("Plotting input distributions with DNN cuts applied.")
    # prepare
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_cut_kine_study
    # get fill weights with dnn cut
    plot_feature_list = ic.selected_features + ic.validation_features
    bkg_df = array_utils.extract_bkg_df(df)
    sig_df = array_utils.extract_sig_df(df)
    bkg_weights = bkg_df["weight"].values
    sig_weights = sig_df["weight"].values
    # normalize
    if plot_cfg.density:
        bkg_weights = bkg_weights / np.sum(bkg_weights)
        sig_weights = sig_weights / np.sum(sig_weights)
    # plot kinematics with dnn cuts
    if dnn_cut < 0 or dnn_cut > 1:
        logger.error(f"DNN cut {dnn_cut} is out of range [0, 1]!")
        return
    # prepare signal
    sig_predictions, _, _ = evaluate_utils.k_folds_predict(
        model_wrapper.get_model(), sig_df[ic.selected_features].values
    )
    if sig_predictions.ndim == 2:
        sig_predictions = sig_predictions[:, multi_class_cut_branch]
    sig_cut_index = array_utils.get_cut_index(sig_predictions, [dnn_cut], ["<"])
    sig_weights_dnn = sig_weights.copy()
    sig_weights_dnn[sig_cut_index] = 0
    # prepare background
    bkg_predictions, _, _ = evaluate_utils.k_folds_predict(
        model_wrapper.get_model(), bkg_df[ic.selected_features].values
    )
    if bkg_predictions.ndim == 2:
        bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
    bkg_cut_index = array_utils.get_cut_index(bkg_predictions, [dnn_cut], ["<"])
    bkg_weights_dnn = bkg_weights.copy()
    bkg_weights_dnn[bkg_cut_index] = 0
    # normalize weights for density plots
    if plot_cfg.density:
        bkg_weights_dnn = bkg_weights_dnn / np.sum(bkg_weights)
        sig_weights_dnn = sig_weights_dnn / np.sum(sig_weights)
    # plot
    for feature in plot_feature_list:
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_array = bkg_df[feature].values
        sig_array = sig_df[feature].values

        feature_cfg = plot_cfg.clone()
        if feature in plot_cfg.__dict__.keys():
            feature_cfg_tmp = getattr(plot_cfg, feature)
            feature_cfg.update(feature_cfg_tmp.get_config_dict())

        # plot sig
        fig, main_ax, ratio_ax = ampl.ratio_axes()
        main_ax.hist(
            sig_array,
            bins=feature_cfg.bins,
            weights=sig_weights,
            histtype="step",
            color=feature_cfg.sig_color,
            label="before DNN cut",
        )
        main_ax.hist(
            sig_array,
            bins=feature_cfg.bins,
            weights=sig_weights_dnn,
            histtype="stepfilled",
            color=feature_cfg.sig_color,
            label="after DNN cut",
        )
        sig_bins, sig_edges = np.histogram(
            sig_array, bins=feature_cfg.bins, weights=sig_weights,
        )
        sig_bins_dnn, _ = np.histogram(
            sig_array, bins=feature_cfg.bins, weights=sig_weights_dnn,
        )
        ampl.plot.plot_ratio(
            sig_edges,
            sig_bins_dnn,
            np.zeros(len(sig_bins)),
            sig_bins,
            np.zeros(len(sig_bins)),
            ratio_ax,
            plottype="raw",
        )
        if feature_cfg.log:
            main_ax.set_yscale("log")
            _, y_max = main_ax.get_ylim()
            main_ax.set_ylim(
                feature_cfg.logy_min, y_max * np.power(10, np.log10(y_max) / 2)
            )
        else:
            _, y_max = main_ax.get_ylim()
            main_ax.set_ylim(0, y_max * 1.4)
        main_ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=main_ax, **(ac.atlas_label.get_config_dict())
            )
        fig.suptitle(f"{feature} (signal)")
        fig.savefig(f"{save_dir}/{feature}(sig).{plot_cfg.save_format}")
        plt.close()
        # plot bkg
        fig, main_ax, ratio_ax = ampl.ratio_axes()
        main_ax.hist(
            bkg_array,
            bins=feature_cfg.bins,
            weights=bkg_weights,
            histtype="step",
            color=feature_cfg.bkg_color,
            label="before DNN cut",
        )
        main_ax.hist(
            bkg_array,
            bins=feature_cfg.bins,
            weights=bkg_weights_dnn,
            histtype="stepfilled",
            color=feature_cfg.bkg_color,
            label="after DNN cut",
        )
        bkg_bins, bkg_edges = np.histogram(
            bkg_array, bins=feature_cfg.bins, weights=bkg_weights,
        )
        bkg_bins_dnn, _ = np.histogram(
            bkg_array, bins=feature_cfg.bins, weights=bkg_weights_dnn,
        )
        ampl.plot.plot_ratio(
            bkg_edges,
            bkg_bins_dnn,
            np.zeros(len(bkg_bins)),
            bkg_bins,
            np.zeros(len(bkg_bins)),
            ratio_ax,
            plottype="raw",
        )
        if feature_cfg.log:
            main_ax.set_yscale("log")
            _, y_max = main_ax.get_ylim()
            main_ax.set_ylim(
                feature_cfg.logy_min, y_max * np.power(10, np.log10(y_max) / 2)
            )
        else:
            _, y_max = main_ax.get_ylim()
            main_ax.set_ylim(0, y_max * 1.4)
        main_ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=main_ax, **(ac.atlas_label.get_config_dict())
            )
        fig.suptitle(f"{feature} (background)")
        fig.savefig(f"{save_dir}/{feature}(bkg).{plot_cfg.save_format}")
        plt.close()
