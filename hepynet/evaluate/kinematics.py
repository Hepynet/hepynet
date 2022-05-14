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
    bkg_df = df.loc[df["sample_name"].isin(ic.bkg_list)]
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
    sig_df = df.loc[df["sample_name"].isin(ic.sig_list)]
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
    plot_features = list(set(ic.selected_features + ic.validation_features))
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # get bkg/sig dataframes and weights (to be reused)
    bkg_df = df.loc[df["sample_name"].isin(ic.bkg_list)]
    bkg_wt = bkg_df["weight"]
    sig_df = df.loc[df["sample_name"].isin(ic.sig_list)]
    sig_wt = sig_df["weight"]
    # plot
    for feature in plot_features:
        # overwrite with sub-level settings if any
        f_cfg = plot_cfg.clone()
        if feature in plot_cfg.__dict__.keys():
            f_cfg_tmp = getattr(plot_cfg, feature)
            f_cfg.update(f_cfg_tmp.get_config_dict())
        # get hist setting for bkg/sig plot
        bkg_args = {"label": "background"}
        sig_args = {"label": "signal"}
        if is_raw:
            bkg_args["range"] = f_cfg.range_raw
            bkg_args["weights"] = bkg_wt * f_cfg.bkg_scale_raw
            sig_args["range"] = f_cfg.range_raw
            sig_args["weights"] = sig_wt * f_cfg.sig_scale_raw
        else:
            bkg_args["range"] = f_cfg.range_processed
            bkg_args["weights"] = bkg_wt * f_cfg.bkg_scale_processed
            sig_args["range"] = f_cfg.range_processed
            sig_args["weights"] = sig_wt * f_cfg.sig_scale_processed
        if f_cfg.logbin and bkg_args["range"]:
            plot_bins = np.logspace(
                np.log10(bkg_args["range"][0]),
                np.log10(bkg_args["range"][1]),
                f_cfg.bins,
            )
        else:
            plot_bins = f_cfg.bins
        bkg_args["bins"] = plot_bins
        sig_args["bins"] = plot_bins
        # overwrite settings with sub-level configs
        bkg_args.update(f_cfg.hist_kwargs_bkg.get_config_dict())
        sig_args.update(f_cfg.hist_kwargs_sig.get_config_dict())
        # make plot
        fig, ax = plt.subplots()
        f_bkg = bkg_df[feature]
        f_sig = sig_df[feature]
        ax.hist(f_bkg, **(bkg_args))
        ax.set_title(feature)
        # decide wether plot signal in same place as background
        if plot_cfg.separate_bkg_sig:
            modify_hist(ax, f_cfg)
            update_atlas_label(ax, ac)
            fig.savefig(f"{save_dir}/{feature}_bkg.{plot_cfg.save_format}")
            plt.close()
            # prepare a new ax
            fig, ax = plt.subplots()
        # sig
        ax.hist(f_sig, **(sig_args))
        ax.set_title(feature)
        modify_hist(ax, f_cfg)
        update_atlas_label(ax, ac)
        if plot_cfg.separate_bkg_sig:
            fig.savefig(f"{save_dir}/{feature}_sig.{plot_cfg.save_format}")
        else:
            fig.savefig(f"{save_dir}/{feature}.{plot_cfg.save_format}")
        plt.close()


def plot_input_dnn(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    job_config: ht.config,
    dnn_cut_down: float = None,
    dnn_cut_up: float = 1,
    multi_class_cut_branch: int = 0,
    save_dir: ht.pathlike = None,
):
    """Plots input distributions comparision plots with DNN cuts applied"""
    logger.info(
        f"Plotting input distributions with DNN cuts [{dnn_cut_down}, {dnn_cut_up}] applied."
    )
    # prepare
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_cut_kine_study
    # get sig/bkg DataFrame and weights
    bkg_df_raw = df_raw.loc[df_raw["sample_name"].isin(ic.bkg_list)]
    sig_df_raw = df_raw.loc[df_raw["sample_name"].isin(ic.sig_list)]
    bkg_wt = bkg_df_raw["weight"].to_numpy("float32")
    sig_wt = sig_df_raw["weight"].to_numpy("float32")
    # get predictions
    bkg_pred = df.loc[df["sample_name"].isin(ic.bkg_list)][
        ["y_pred_0"]
    ].to_numpy("float32")
    sig_pred = df.loc[df["sample_name"].isin(ic.sig_list)][
        ["y_pred_0"]
    ].to_numpy("float32")
    # normalize
    if plot_cfg.density:
        bkg_wt = bkg_wt / np.sum(bkg_wt)
        sig_wt = sig_wt / np.sum(sig_wt)
    # plot kinematics with dnn cuts
    if dnn_cut_down < 0 or dnn_cut_down > 1:
        logger.error(f"DNN cut {dnn_cut_down} is out of range [0, 1]!")
        return
    if dnn_cut_up < 0 or dnn_cut_up > 1:
        logger.error(f"DNN cut {dnn_cut_down} is out of range [0, 1]!")
        return
    if dnn_cut_down > dnn_cut_up:
        logger.error(
            f"DNN cut lower cut {dnn_cut_down} has higher value than {dnn_cut_up}!"
        )
        return
    # get signal weights with dnn applied
    sig_pred = sig_pred[:, multi_class_cut_branch]
    sig_cut_id = np.argwhere(
        (sig_pred < dnn_cut_down) | (sig_pred > dnn_cut_up)
    )
    sig_wt_dnn = sig_wt.copy()
    sig_wt_dnn[sig_cut_id] = 0
    # get background weights with dnn applied
    bkg_pred = bkg_pred[:, multi_class_cut_branch]
    bkg_cut_id = np.argwhere(
        (bkg_pred < dnn_cut_down) | (bkg_pred > dnn_cut_up)
    )
    bkg_wt_dnn = bkg_wt.copy()
    bkg_wt_dnn[bkg_cut_id] = 0
    # normalize weights for density plots
    if plot_cfg.density:
        bkg_wt_dnn = bkg_wt_dnn / np.sum(bkg_wt)
        sig_wt_dnn = sig_wt_dnn / np.sum(sig_wt)
    # plot
    plot_feature_list = ic.selected_features + ic.validation_features
    for feature in plot_feature_list:
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_array = bkg_df_raw[feature].to_numpy("float32")
        sig_array = sig_df_raw[feature].to_numpy("float32")

        note = f"DNN cut: [{dnn_cut_down}, {dnn_cut_up}]"
        # plot sig
        plot_input_dnn_single(
            job_config,
            feature,
            "sig",
            sig_array,
            sig_wt,
            sig_wt_dnn,
            save_dir,
            note=note,
        )
        # plot bkg
        plot_input_dnn_single(
            job_config,
            feature,
            "bkg",
            bkg_array,
            bkg_wt,
            bkg_wt_dnn,
            save_dir,
            note=note,
        )


def plot_input_dnn_single(
    job_config: ht.config,
    feature,
    event_type,
    kine,
    wt,
    wt_dnn,
    save_dir,
    note="",
):
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_cut_kine_study
    feature_cfg = plot_cfg.clone()
    if feature in plot_cfg.__dict__.keys():
        feature_cfg_tmp = getattr(plot_cfg, feature)
        feature_cfg.update(feature_cfg_tmp.get_config_dict())
    plot_range = feature_cfg.range_processed
    if event_type == "sig":
        color = feature_cfg.sig_color
    elif event_type == "bkg":
        color = feature_cfg.bkg_color
    else:
        color = None

    # plot with/without dnn cuts
    fig, main_ax, ratio_ax = ampl.ratio_axes()
    main_ax.hist(
        kine,
        bins=feature_cfg.bins,
        range=plot_range,
        weights=wt,
        histtype="step",
        color=color,
        label="before DNN cut",
    )
    main_ax.hist(
        kine,
        bins=feature_cfg.bins,
        range=plot_range,
        weights=wt_dnn,
        histtype="stepfilled",
        color=color,
        label="after DNN cut",
    )
    hist_bins, edges = np.histogram(
        kine,
        bins=feature_cfg.bins,
        weights=wt,
    )
    hist_bins_dnn, _ = np.histogram(
        kine,
        bins=feature_cfg.bins,
        weights=wt_dnn,
    )
    ampl.plot.plot_ratio(
        edges,
        hist_bins_dnn,
        np.zeros(len(hist_bins)),
        hist_bins,
        np.zeros(len(hist_bins)),
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
    main_ax.text(
        1,
        1,
        note,
        color="royalblue",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=main_ax.transAxes,
    )
    fig.suptitle(f"{feature} ({event_type})")
    fig.savefig(f"{save_dir}/{feature}({event_type}).{plot_cfg.save_format}")
    plt.close()


def modify_hist(ax: ht.ax, feature_cfg: ht.sub_config):
    if feature_cfg.x_label:
        ax.set_xlabel(feature_cfg.x_label)
    if feature_cfg.y_label:
        ax.set_ylabel(feature_cfg.y_label)
    log_minor_locator = LogLocator(
        base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    if feature_cfg.logx:
        ax.set_xscale("symlog")
        ax.xaxis.set_minor_locator(log_minor_locator)
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


def update_atlas_label(ax: ht.ax, apply_config: ht.sub_config):
    if apply_config.plot_atlas_label:
        ampl.plot.draw_atlas_label(
            0.05, 0.95, ax=ax, **(apply_config.atlas_label.get_config_dict())
        )
