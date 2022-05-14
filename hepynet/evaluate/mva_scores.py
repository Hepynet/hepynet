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
):
    # Initialize
    logger.info("Plotting MVA scores")
    ic = job_config.input
    tc = job_config.train
    ac = job_config.apply
    plot_config = ac.cfg_mva_scores_data_mc.clone()
    # Plot for each nodes
    if tc.use_multi_label:
        all_nodes = list(ic.multi_label.keys())
    else:
        all_nodes = [1]
    num_nodes = len(all_nodes)
    for node_num in range(num_nodes):
        # Prepare inputs
        sig_scores_dict = {}
        sig_weights_dict = {}
        for sig_process in plot_config.sig_list:
            load_input(
                df_raw,
                df,
                node_num,
                sig_scores_dict,
                sig_weights_dict,
                sig_process,
                plot_config.sig_scale,
            )
        bkg_scores_dict = {}
        bkg_weights_dict = {}
        for bkg_process in plot_config.bkg_list:
            load_input(
                df_raw,
                df,
                node_num,
                bkg_scores_dict,
                bkg_weights_dict,
                bkg_process,
                plot_config.bkg_scale,
            )
        if plot_config.apply_data:
            data_key = plot_config.data_key
            data_scores = df.loc[
                df["sample_name"] == data_key, f"y_pred_{node_num}"
            ].values
            if data_scores.ndim == 1:
                data_scores = data_scores.reshape((-1, 1))
            data_weights = (
                df_raw.loc[df["sample_name"] == data_key, "weight"].values
                * plot_config.data_scale
            )
        # Make plots
        if plot_config.show_ratio:
            if plot_config.fig_size:
                fig = plt.figure(figsize=plot_config.fig_size)
            else:
                fig = plt.figure(figsize=(50 / 3, 50 / 3))
            gs = mpl.gridspec.GridSpec(4, 1, hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(gs[0:3])
            ax.tick_params(labelbottom=False)
            ratio_ax = fig.add_subplot(gs[3], sharex=ax)
            ratio_ax.autoscale(axis="x", tight=True)
            plt.sca(ax)
        else:
            if plot_config.fig_size:
                fig, ax = plt.subplots(figsize=plot_config.fig_size)
            else:
                fig, ax = plt.subplots(figsize=(50 / 3, 100 / 9))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = itertools.cycle(colors)
        # Plot bkg
        bkg_collect = list()
        bkg_scores_all = None
        bkg_weights_all = None
        for key, node_score in bkg_scores_dict.items():
            node_weight = bkg_weights_dict[key]
            bkg_bins, _ = np.histogram(
                node_score.flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=node_weight.flatten(),
            )
            bkg = ampl.plot.Background(key, bkg_bins, color=next(color_cycle))
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
            bkg_scores_all.flatten(),
            bins=plot_config.bins,
            range=plot_config.range,
            weights=bkg_weights_all.flatten(),
        )
        sumw2, _ = np.histogram(
            bkg_scores_all.flatten(),
            bins=plot_config.bins,
            range=plot_config.range,
            weights=np.power(bkg_weights_all, 2).flatten(),
        )
        bkg_stats_errs = np.sqrt(sumw2)
        ampl.plot.plot_backgrounds(bkg_collect, bkg_edges, ax=ax)
        # Plot sig
        for key, value in sig_scores_dict.items():
            sig_bins, sig_edges = np.histogram(
                value.flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=(sig_weights_dict[key]).flatten(),
            )
            ampl.plot.plot_signal(
                key, sig_edges, sig_bins, color=next(color_cycle), ax=ax
            )
        # Plot data
        if plot_config.apply_data:
            data_bins, data_edges = np.histogram(
                data_scores.flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=(data_weights * plot_config.data_scale).flatten(),
            )
            sumw2, _ = np.histogram(
                data_scores.flatten(),
                bins=plot_config.bins,
                range=plot_config.range,
                weights=np.power(
                    data_weights * plot_config.data_scale, 2
                ).flatten(),
            )
            data_stats_errs = np.sqrt(sumw2)
            if plot_config.label_scale and plot_config.data_scale != 1:
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
        # Reorder legends, data on top, background at bottom
        n_bkg = len(bkg_scores_dict)
        n_sig = len(sig_scores_dict)
        if plot_config.apply_data:
            order = (
                [-1] + list(range(n_bkg, n_bkg + n_sig)) + list(range(n_bkg))
            )
        else:
            order = list(range(n_bkg, n_bkg + n_sig)) + list(range(n_bkg))
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        ax.legend(
            handles, labels, **(plot_config.legend_paras.get_config_dict())
        )
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05,
                0.95,
                ax=ax,
                **(ac.atlas_label.get_config_dict()),
            )
        # Plot patch
        for func, args in plot_config.plot_patch.get_config_dict().items():
            getattr(ax, func)(**args)

        if plot_config.show_ratio:
            ratio_ax.set_xlabel("DNN score")
        else:
            ax.set_xlabel("DNN score")
        # Save lin/log plots
        _, y_max = ax.get_ylim()
        formats = plot_config.save_format
        if not isinstance(formats, list):
            formats = [formats]
        file_prefix = f"mva_scores_data_mc_{node_num}"
        # save with lin scale
        ax.set_ylim(0, y_max * 1.4)
        for fm in formats:
            fig.savefig(f"{save_dir}/{file_prefix}_lin.{fm}")
        # save with log scale
        ax.set_yscale("log")
        ax.set_ylim(
            plot_config.logy_min,
            y_max * np.power(10, np.log10(y_max / plot_config.logy_min) * 0.8),
        )
        for fm in formats:
            fig.savefig(f"{save_dir}/{file_prefix}_log.{fm}")
        if plot_config.symlog_x:
            ax.set_xscale(
                "symlog",
                linthresh=plot_config.linthresh,
                linscale=plot_config.linscale,
            )
            ## save lin
            ax.set_yscale("linear")
            ax.set_ylim(0, y_max * 1.4)
            for fm in formats:
                fig.savefig(f"{save_dir}/{file_prefix}_lin_symlog_x.{fm}")
            ## save log
            ax.set_yscale("log")
            ax.set_ylim(
                plot_config.logy_min,
                y_max
                * np.power(10, np.log10(y_max / plot_config.logy_min) * 0.8),
            )
            for fm in formats:
                fig.savefig(f"{save_dir}/{file_prefix}_log_symlog_x.{fm}")


def load_input(
    df_raw,
    df,
    node_num,
    scores_dict,
    weights_dict,
    process_def,
    scale=1,
):
    """Load input data for a node."""
    item_info = process_def.split(":")
    item_info = [x.strip() for x in item_info]
    samples = item_info[0].split("+")
    samples = [x.strip() for x in samples]
    if len(item_info) > 1:
        name = item_info[1]
    else:
        name = item_info[0]
    # Load score
    score = df.loc[
        df["sample_name"].isin(samples), f"y_pred_{node_num}"
    ].values
    if score.ndim == 1:
        score = score.reshape((-1, 1))
    scores_dict[name] = score
    # Load weights
    sub_df = df_raw.loc[
        df_raw["sample_name"].isin(samples), ["sample_name", "weight"]
    ]
    if type(scale) == int or type(scale) == float:
        sub_df["weight"] *= scale
    else:
        for samp, k in scale.items():
            sub_df.loc[sub_df["sample_name"] == samp, "weight"] *= k
    weights_dict[name] = sub_df["weight"].values


def plot_train_test_compare(
    df: pd.DataFrame, job_config: ht.config, save_dir: ht.pathlike
):
    """Plots train/test datasets' cores distribution comparison"""
    # Initialize
    logger.info("Plotting train/test scores.")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = job_config.apply.cfg_train_test_compare
    if tc.use_multi_label:
        all_nodes = list(ic.multi_label.keys())
    else:
        all_nodes = [1]
    # Get inputs
    train_index = df["is_train"] == True
    test_index = df["is_train"] == False
    sig_index = (df["is_sig"] == True) & (df["is_mc"] == True)
    bkg_index = (df["is_sig"] == False) & (df["is_mc"] == True)
    # Get weights
    xs_train_weight = df.loc[sig_index & train_index, ["weight"]].values
    xs_test_weight = df.loc[sig_index & test_index, ["weight"]].values
    xb_train_weight = df.loc[bkg_index & train_index, ["weight"]].values
    xb_test_weight = df.loc[bkg_index & test_index, ["weight"]].values

    # Plot for each nodes
    num_nodes = len(all_nodes)
    for node_num in range(num_nodes):
        # Get scores
        xs_train_scores = df.loc[
            sig_index & train_index, [f"y_pred_{node_num}"]
        ].values
        xs_test_scores = df.loc[
            sig_index & test_index, [f"y_pred_{node_num}"]
        ].values
        xb_train_scores = df.loc[
            bkg_index & train_index, [f"y_pred_{node_num}"]
        ].values
        xb_test_scores = df.loc[
            bkg_index & test_index, [f"y_pred_{node_num}"]
        ].values
        # Plot
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
        # Plot train scores
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
            ampl.plot.draw_atlas_label(
                0.05,
                0.95,
                ax=ax,
                **(ac.atlas_label.get_config_dict()),
            )
        ax.set_xlabel("DNN score")
        # Save lin/log plots
        file_name = f"mva_scores_{all_nodes[node_num]}"
        _, y_max = ax.get_ylim()
        # save lin
        ax.set_ylim(0, y_max * 1.4)
        ax.set_xlabel("DNN score")
        fig.savefig(save_dir / f"{file_name}_lin.{plot_config.save_format}")
        # save log
        ax.set_yscale("log")
        ax.set_ylim(
            plot_config.logy_min, y_max * np.power(10, np.log10(y_max) / 2)
        )
        fig.savefig(save_dir / f"{file_name}_log.{plot_config.save_format}")
