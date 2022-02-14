import itertools
import logging

import atlas_mpl_style as ampl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hepynet.common.hepy_type as ht

logger = logging.getLogger("hepynet")


def plot_mva_scores(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    job_config: ht.config,
    save_dir: ht.pathlike,
    file_name: str = "mva_scores",
):
    # initialize
    logger.info("Plotting MVA scores")
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = ac.cfg_mva_scores_data_mc.clone()
    # prepare signal
    sig_scores_dict = {}
    sig_weights_dict = {}
    for sig_item in plot_config.sig_list:
        item_info = sig_item.split(":")
        item_info = [x.strip() for x in item_info]
        sig_samples = item_info[0].split("+")
        sig_samples = [x.strip() for x in sig_samples]
        if len(item_info) > 1:
            sig_name = item_info[1]
        else:
            sig_name = item_info[0]
        sig_score = df.loc[
            df["sample_name"].isin(sig_samples), "y_pred"
        ].values
        if sig_score.ndim == 1:
            sig_score = sig_score.reshape((-1, 1))
        sig_scores_dict[sig_name] = sig_score
        sig_weight = df_raw.loc[
            df["sample_name"].isin(sig_samples), "weight"
        ].values
        sig_weights_dict[sig_name] = sig_weight
    # prepare background
    bkg_scores_dict = {}
    bkg_weights_dict = {}
    for bkg_item in plot_config.bkg_list:
        item_info = bkg_item.split(":")
        item_info = [x.strip() for x in item_info]
        bkg_samples = item_info[0].split("+")
        bkg_samples = [x.strip() for x in bkg_samples]
        if len(item_info) > 1:
            bkg_name = item_info[1]
        else:
            bkg_name = item_info[0]
        bkg_score = df.loc[
            df["sample_name"].isin(bkg_samples), "y_pred"
        ].values
        if bkg_score.ndim == 1:
            bkg_score = bkg_score.reshape((-1, 1))
        bkg_scores_dict[bkg_name] = bkg_score
        bkg_weight = df_raw.loc[
            df["sample_name"].isin(bkg_samples), "weight"
        ].values
        bkg_weights_dict[bkg_name] = bkg_weight
    # prepare data
    if plot_config.apply_data:
        data_key = plot_config.data_key
        data_scores = df.loc[df["sample_name"] == data_key, "y_pred"].values
        if data_scores.ndim == 1:
            data_scores = data_scores.reshape((-1, 1))
        data_weights = df_raw.loc[
            df["sample_name"] == data_key, "weight"
        ].values

    # make plots
    all_nodes = ["sig"] + tc.output_bkg_node_names
    for node_id, node in enumerate(all_nodes):
        if plot_config.show_ratio:
            if plot_config.fig_size:
                fig = plt.figure(figsize=plot_config.fig_size)
            else:
                fig = plt.figure(figsize=(50 / 3, 50 / 3))
            gs = mpl.gridspec.GridSpec(4, 1, hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(gs[0:3])
            ax.tick_params(labelbottom=False)
            ratio_ax = fig.add_subplot(gs[3], sharex=ax)
            # ratio_ax.yaxis.set_major_locator(
            #    mpl.ticker.MaxNLocator(
            #        symmetric=True, prune="both", min_n_ticks=5, nbins=4
            #    )
            # )
            ratio_ax.autoscale(axis="x", tight=True)
            plt.sca(ax)
        else:
            if plot_config.fig_size:
                fig, ax = plt.subplots(figsize=plot_config.fig_size)
            else:
                fig, ax = plt.subplots(figsize=(50 / 3, 100 / 9))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = itertools.cycle(colors)
        # plot bkg
        bkg_collect = list()
        bkg_scores_all = None
        bkg_weights_all = None
        for key, value in bkg_scores_dict.items():
            node_score = value[:, node_id].flatten()
            node_weight = bkg_weights_dict[key] * plot_config.bkg_scale
            bkg_bins, _ = np.histogram(
                node_score,
                bins=plot_config.bins,
                range=plot_config.range,
                weights=node_weight,
                density=plot_config.density,
            )
            if plot_config.bkg_scale != 1:
                bkg_label = f"{key} x{plot_config.bkg_scale}"
            else:
                bkg_label = key
            bkg = ampl.plot.Background(
                bkg_label, bkg_bins, color=next(color_cycle)
            )
            bkg_collect.append(bkg)
            if bkg_scores_all is None:
                bkg_scores_all = node_score
                bkg_weights_all = node_weight
            else:
                bkg_scores_all = np.concatenate((bkg_scores_all, node_score))
                bkg_weights_all = np.concatenate(
                    (bkg_weights_all, node_weight)
                )
        bkg_all_bins, bkg_edges = np.histogram(
            bkg_scores_all,
            bins=plot_config.bins,
            range=plot_config.range,
            weights=bkg_weights_all,
            density=plot_config.density,
        )
        sumw2, _ = np.histogram(
            bkg_scores_all,
            bins=plot_config.bins,
            range=plot_config.range,
            weights=np.power(bkg_weights_all, 2),
        )
        bkg_stats_errs = np.sqrt(sumw2)
        if plot_config.density:
            norm_sum = np.sum(bkg_weights_all) * (1 / plot_config.bins)
            bkg_stats_errs /= norm_sum
        ampl.plot.plot_backgrounds(bkg_collect, bkg_edges, ax=ax)
        # plot sig
        for key, value in sig_scores_dict.items():
            sig_bins, sig_edges = np.histogram(
                value[:, node_id].flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=sig_weights_dict[key] * plot_config.sig_scale,
                density=plot_config.density,
            )
            if plot_config.sig_scale != 1:
                sig_label = f"{key} x{plot_config.sig_scale}"
            else:
                sig_label = key
            ampl.plot.plot_signal(
                sig_label, sig_edges, sig_bins, color=next(color_cycle), ax=ax
            )
        # plot data
        if plot_config.apply_data:
            data_bins, data_edges = np.histogram(
                data_scores[:, node_id].flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=data_weights * plot_config.data_scale,
                density=plot_config.density,
            )
            sumw2, _ = np.histogram(
                data_scores[:, node_id].flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=np.power(data_weights * plot_config.data_scale, 2),
            )
            data_stats_errs = np.sqrt(sumw2)
            if plot_config.density:
                norm_sum = np.sum(data_weights * plot_config.data_scale) * (
                    1 / plot_config.bins
                )
                data_stats_errs /= norm_sum
            if plot_config.data_scale != 1:
                data_label = f"Data x{plot_config.data_scale}"
            else:
                data_label = "Data"
            ampl.plot.plot_data(
                data_edges,
                data_bins,
                stat_errs=data_stats_errs,
                label=data_label,
                ax=ax,
            )
            # plot ratio
            if plot_config.show_ratio:
                ampl.plot.plot_ratio(
                    data_edges,
                    data_bins,
                    data_stats_errs,
                    bkg_all_bins,
                    bkg_stats_errs,
                    ratio_ax,
                    plottype="raw",
                    offscale_errs=True,  # TODO: add as an option?
                )
                ratio_ax.set_ylim(0, 2)
        ax.set_xlim(plot_config.range[0], plot_config.range[1])

        # reorder legends, data on top, background at bottom
        n_bkg = len(bkg_scores_dict)
        n_sig = len(sig_scores_dict)
        if plot_config.apply_data:
            order = [-1] + list(range(n_bkg, n_bkg + n_sig)) + list(range(n_bkg))
        else:
            order = list(range(n_bkg, n_bkg + n_sig)) + list(range(n_bkg))
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        ax.legend(
            handles, labels, **(plot_config.legend_paras.get_config_dict())
        )

        if ac.plot_atlas_label:
            if plot_config.density:
                desc = "Density Plot"
            else:
                desc = None
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict()), desc=desc
            )
        ax.set_xlabel("DNN score")
        # Save lin/log plots
        _, y_max = ax.get_ylim()
        ## save lin
        ax.set_ylim(0, y_max * 1.4)
        fig.savefig(
            f"{save_dir}/{file_name}_node_{node}_lin.{plot_config.save_format}"
        )
        ## save log
        ax.set_yscale("log")
        ax.set_ylim(
            plot_config.logy_min,
            y_max * np.power(10, np.log10(y_max / plot_config.logy_min) * 0.8),
        )
        fig.savefig(
            f"{save_dir}/{file_name}_node_{node}_log.{plot_config.save_format}"
        )

    return 0  # success run


def plot_train_test_compare(
    df: pd.DataFrame, job_config: ht.config, save_dir: ht.pathlike
):
    """Plots train/test datasets' cores distribution comparison"""
    # initialize
    logger.info("Plotting train/test scores.")
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = job_config.apply.cfg_train_test_compare
    all_nodes = ["sig"] + tc.output_bkg_node_names
    # get inputs
    train_index = df["is_train"] == True
    test_index = df["is_train"] == False
    sig_index = (df["is_sig"] == True) & (df["is_mc"] == True)
    bkg_index = (df["is_sig"] == False) & (df["is_mc"] == True)
    xs_train_scores = df.loc[sig_index & train_index, ["y_pred"]].values
    xs_test_scores = df.loc[sig_index & test_index, ["y_pred"]].values
    xs_train_weight = df.loc[sig_index & train_index, ["weight"]].values
    xs_test_weight = df.loc[sig_index & test_index, ["weight"]].values
    xb_train_scores = df.loc[bkg_index & train_index, ["y_pred"]].values
    xb_test_scores = df.loc[bkg_index & test_index, ["y_pred"]].values
    xb_train_weight = df.loc[bkg_index & train_index, ["weight"]].values
    xb_test_weight = df.loc[bkg_index & test_index, ["weight"]].values

    # plot for each nodes
    num_nodes = len(all_nodes)
    for node_num in range(num_nodes):
        fig, ax = plt.subplots()
        # plot test scores
        bkg_bins, bkg_edges = np.histogram(
            xb_test_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=xb_test_weight,
            density=plot_config.density,
        )
        bkg = ampl.plot.Background(
            "background (test)", bkg_bins, color=plot_config.bkg_color
        )
        ampl.plot.plot_backgrounds([bkg], bkg_edges, ax=ax)
        sig_bins, sig_edges = np.histogram(
            xs_test_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=xs_test_weight,
            density=plot_config.density,
        )
        ampl.plot.plot_signal(
            "signal (test)", sig_edges, sig_bins, color=plot_config.sig_color
        )
        # plot train scores
        ## bkg
        bkg_bins, bkg_edges = np.histogram(
            xb_train_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=xb_train_weight,
            density=plot_config.density,
        )
        sumw2, _ = np.histogram(
            xb_train_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=np.power(xb_train_weight, 2),
        )
        bkg_stats_errs = np.sqrt(sumw2)
        if plot_config.density:
            norm_sum = np.sum(xb_train_weight) * (1 / plot_config.bins)
            bkg_stats_errs /= norm_sum
        err_x = 0.5 * (bkg_edges[:-1] + bkg_edges[1:])
        ax.errorbar(
            err_x,
            bkg_bins,
            bkg_stats_errs,
            0.5 / plot_config.bins,
            fmt=".k",
            mfc=plot_config.bkg_color,
            ms=10,
            label="background (train)",
        )
        ## sig
        sig_bins, sig_edges = np.histogram(
            xs_train_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=xs_train_weight,
            density=plot_config.density,
        )
        sumw2, _ = np.histogram(
            xs_train_scores,
            bins=plot_config.bins,
            range=(0, 1),
            weights=np.power(xs_train_weight, 2),
        )
        sig_stats_errs = np.sqrt(sumw2)
        if plot_config.density:
            norm_sum = np.sum(xs_train_weight) * (1 / plot_config.bins)
            sig_stats_errs /= norm_sum
        err_x = 0.5 * (sig_edges[:-1] + sig_edges[1:])
        ax.errorbar(
            err_x,
            sig_bins,
            sig_stats_errs,
            0.5 / plot_config.bins,
            fmt=".k",
            mfc=plot_config.sig_color,
            ms=10,
            label="signal (train)",
        )
        # final adjustments
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            if plot_config.density:
                desc = "Density Plot"
            else:
                desc = None
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict()), desc=desc
            )
        ax.set_xlabel("DNN score")
        # Save lin/log plots
        file_name = f"mva_scores_{all_nodes[node_num]}"
        _, y_max = ax.get_ylim()
        ## save lin
        ax.set_ylim(0, y_max * 1.4)
        ax.set_xlabel("DNN score")
        fig.savefig(save_dir / f"{file_name}_lin.{plot_config.save_format}")
        ## save log
        ax.set_yscale("log")
        ax.set_ylim(
            plot_config.logy_min, y_max * np.power(10, np.log10(y_max) / 2)
        )
        fig.savefig(save_dir / f"{file_name}_log.{plot_config.save_format}")
