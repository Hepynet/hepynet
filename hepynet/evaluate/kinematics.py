import logging
import pathlib
from typing import List

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, LogLocator

import hepynet.common.hepy_type as ht
from hepynet.data_io import array_utils

logger = logging.getLogger("hepynet")


def plot_correlation_matrix(
    df: pd.DataFrame, job_config: ht.config, save_dir: ht.pathlike = "."
):
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
        bkg_df[features].to_numpy("float32"),
        bkg_df["weight"].to_numpy("float32"),
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
        sig_df[features].to_numpy("float32"),
        sig_df["weight"].to_numpy("float32"),
    )
    paint_correlation_matrix(ax[1], sig_matrix, features)
    fig_save_path = save_dir / "correlation_matrix.png"
    logger.debug(f"Save correlation matrix to: {fig_save_path}")
    fig.savefig(fig_save_path)


def paint_correlation_matrix(
    ax: ht.ax, corr_matrix: np.ndarray, labels: List[str]
):
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
    df: pd.DataFrame,
    job_config: ht.config,
    save_dir: ht.pathlike = None,
    is_raw: bool = True,
):
    """Plots input distributions comparision plots for sig/bkg/data"""
    # setup config
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_kine
    # prepare
    plot_feature_list = list(
        set(ic.selected_features + ic.validation_features)
    )
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # plot
    for feature in plot_feature_list:
        feature_cfg = plot_cfg.clone()
        if feature in plot_cfg.__dict__.keys():
            feature_cfg_tmp = getattr(plot_cfg, feature)
            feature_cfg.update(feature_cfg_tmp.get_config_dict())
        if is_raw:
            plot_range = feature_cfg.range_raw
            bkg_scale = feature_cfg.bkg_scale_raw
            sig_scale = feature_cfg.sig_scale_raw
        else:
            plot_range = feature_cfg.range_processed
            bkg_scale = feature_cfg.bkg_scale_processed
            sig_scale = feature_cfg.sig_scale_processed
        if feature_cfg.logbin and plot_range:
            plot_bins = np.logspace(
                np.log10(plot_range[0]),
                np.log10(plot_range[1]),
                feature_cfg.bins,
            )
        else:
            plot_bins = feature_cfg.bins
        # plot bkg
        # bkg_df = array_utils.extract_bkg_df(df)
        bkg_wt = df.loc[
            (df["is_mc"] == True) & (df["is_sig"] == False), "weight"
        ].to_numpy("float32")
        fig, ax = plt.subplots()
        hist_kwargs = feature_cfg.hist_kwargs_bkg.get_config_dict()
        remove_hist_kwargs_duplicates(hist_kwargs)
        ax.hist(
            df.loc[
                (df["is_mc"] == True) & (df["is_sig"] == False), feature
            ].to_numpy("float32"),
            bins=plot_bins,
            range=plot_range,
            weights=bkg_wt * bkg_scale,
            label="background",
            **(hist_kwargs),
        )
        modify_hist(ax, feature_cfg, plot_range)
        ax.set_xlabel(feature)
        # decide wether plot in same place
        if plot_cfg.separate_bkg_sig:
            if ac.plot_atlas_label:
                ampl.plot.draw_atlas_label(
                    0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
                )
            # fig.suptitle(feature)
            fig.savefig(f"{save_dir}/{feature}_bkg.{plot_cfg.save_format}")
            plt.close()
            fig, ax = plt.subplots()
        # plot sig
        sig_df = array_utils.extract_sig_df(df)
        sig_wt = sig_df["weight"].to_numpy("float32")
        hist_kwargs = feature_cfg.hist_kwargs_sig.get_config_dict()
        remove_hist_kwargs_duplicates(hist_kwargs)
        ax.hist(
            sig_df[feature].to_numpy("float32"),
            bins=plot_bins,
            range=plot_range,
            weights=sig_wt * sig_scale,
            label="signal",
            **(hist_kwargs),
        )
        modify_hist(ax, feature_cfg, plot_range)
        ax.set_xlabel(feature)
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        # fig.suptitle(feature)
        if plot_cfg.separate_bkg_sig:
            fig.savefig(f"{save_dir}/{feature}_sig.{plot_cfg.save_format}")
        else:
            fig.savefig(f"{save_dir}/{feature}.{plot_cfg.save_format}")
        plt.close()


def plot_input_dnn(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    job_config: ht.config,
    dnn_cut: float = None,
    multi_class_cut_branch: int = 0,
    save_dir: ht.pathlike = None,
):
    """Plots input distributions comparision plots with DNN cuts applied"""
    logger.info(
        f"Plotting input distributions with DNN cuts {dnn_cut} applied."
    )
    # prepare
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_cut_kine_study
    # get sig/bkg DataFrame and weights
    bkg_df_raw = array_utils.extract_bkg_df(df_raw)
    sig_df_raw = array_utils.extract_sig_df(df_raw)
    bkg_weights = bkg_df_raw["weight"].to_numpy("float32")
    sig_weights = sig_df_raw["weight"].to_numpy("float32")
    # get predictions
    bkg_predictions = array_utils.extract_bkg_df(df)[["y_pred"]].to_numpy(
        "float32"
    )
    sig_predictions = array_utils.extract_sig_df(df)[["y_pred"]].to_numpy(
        "float32"
    )
    # normalize
    if plot_cfg.density:
        bkg_weights = bkg_weights / np.sum(bkg_weights)
        sig_weights = sig_weights / np.sum(sig_weights)
    # plot kinematics with dnn cuts
    if dnn_cut < 0 or dnn_cut > 1:
        logger.error(f"DNN cut {dnn_cut} is out of range [0, 1]!")
        return
    # prepare signal
    sig_predictions = sig_predictions[:, multi_class_cut_branch]
    sig_cut_index = array_utils.get_cut_index(
        sig_predictions, [dnn_cut], ["<"]
    )
    sig_weights_dnn = sig_weights.copy()
    sig_weights_dnn[sig_cut_index] = 0
    # prepare background
    bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
    bkg_cut_index = array_utils.get_cut_index(
        bkg_predictions, [dnn_cut], ["<"]
    )
    bkg_weights_dnn = bkg_weights.copy()
    bkg_weights_dnn[bkg_cut_index] = 0
    # normalize weights for density plots
    if plot_cfg.density:
        bkg_weights_dnn = bkg_weights_dnn / np.sum(bkg_weights)
        sig_weights_dnn = sig_weights_dnn / np.sum(sig_weights)
    # plot
    plot_feature_list = ic.selected_features + ic.validation_features
    for feature in plot_feature_list:
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_array = bkg_df_raw[feature].to_numpy("float32")
        sig_array = sig_df_raw[feature].to_numpy("float32")

        feature_cfg = plot_cfg.clone()
        if feature in plot_cfg.__dict__.keys():
            feature_cfg_tmp = getattr(plot_cfg, feature)
            feature_cfg.update(feature_cfg_tmp.get_config_dict())
        plot_range = feature_cfg.range_processed

        # plot sig
        fig, main_ax, ratio_ax = ampl.ratio_axes()
        main_ax.hist(
            sig_array,
            bins=feature_cfg.bins,
            range=plot_range,
            weights=sig_weights,
            histtype="step",
            color=feature_cfg.sig_color,
            label="before DNN cut",
        )
        main_ax.hist(
            sig_array,
            bins=feature_cfg.bins,
            range=plot_range,
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
        if plot_range:
            main_ax.set_xlim(plot_range[0], plot_range[1])
            ratio_ax.set_xlim(plot_range[0], plot_range[1])
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
        # main_ax.set_ylabel(feature_cfg.y_label)
        ratio_ax.set_xlabel(feature)
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
            range=plot_range,
            weights=bkg_weights,
            histtype="step",
            color=feature_cfg.bkg_color,
            label="before DNN cut",
        )
        main_ax.hist(
            bkg_array,
            bins=feature_cfg.bins,
            range=plot_range,
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
        if plot_range:
            main_ax.set_xlim(plot_range[0], plot_range[1])
            ratio_ax.set_xlim(plot_range[0], plot_range[1])
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
        # main_ax.set_ylabel(feature_cfg.y_label)
        ratio_ax.set_xlabel(feature)
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=main_ax, **(ac.atlas_label.get_config_dict())
            )
        fig.suptitle(f"{feature} (background)")
        fig.savefig(f"{save_dir}/{feature}(bkg).{plot_cfg.save_format}")
        plt.close()


def remove_hist_kwargs_duplicates(hist_kwargs: dict):
    if "bins" in hist_kwargs:
        del hist_kwargs["bins"]
    if "logbin" in hist_kwargs:
        del hist_kwargs["logbin"]
    if "logx" in hist_kwargs:
        del hist_kwargs["logx"]
    if "logy" in hist_kwargs:
        del hist_kwargs["logy"]
    if "range" in hist_kwargs:
        del hist_kwargs["range"]
    if "weights" in hist_kwargs:
        del hist_kwargs["weights"]
    if "label" in hist_kwargs:
        del hist_kwargs["label"]
    return hist_kwargs


def modify_hist(ax: ht.ax, feature_cfg: ht.sub_config, plot_range: ht.bound):
    ax.set_xlabel(feature_cfg.x_label)
    ax.set_ylabel(feature_cfg.y_label)
    log_minor_locator = LogLocator(
        base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    if feature_cfg.logx:
        ax.set_xscale("symlog")
        ax.xaxis.set_minor_locator(log_minor_locator)
    if plot_range:
        ax.set_xlim(plot_range[0], plot_range[1])
    y_min, y_max = ax.get_ylim()
    if feature_cfg.logy:
        ax.set_yscale("symlog")
        log_range = np.log10(y_max)
        y_max = pow(10, np.log10(y_max) + log_range * 0.4)
        ax.set_ylim(y_min, y_max * 1.4)
        ax.yaxis.set_minor_locator(log_minor_locator)
    else:
        ax.set_ylim(y_min, y_max * 1.4)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
