import logging

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hepynet.evaluate import evaluate_utils
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def plot_mva_scores(
    model_wrapper: hep_model.Model_Base,
    df: pd.DataFrame,
    job_config,
    save_dir,
    file_name="mva_scores",
):
    # initialize
    logger.info("Plotting MVA scores")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = ac.cfg_mva_scores_data_mc.clone()
    model = model_wrapper.get_model()
    # prepare signal
    sig_scores_dict = {}
    sig_weights_dict = {}
    for sig_key in plot_config.sig_list:
        sig_df = df.loc[df["sample_name"] == sig_key, :]
        x = sig_df[ic.selected_features].values
        sig_score, _, _ = evaluate_utils.k_folds_predict(model, x)
        if sig_score.ndim == 1:
            sig_score = sig_score.reshape((-1, 1))
        sig_scores_dict[sig_key] = sig_score
        sig_weights_dict[sig_key] = sig_df["weight"].values
    # prepare background
    bkg_scores_dict = {}
    bkg_weights_dict = {}
    for bkg_key in plot_config.bkg_list:
        bkg_df = df.loc[df["sample_name"] == bkg_key, :]
        x = bkg_df[ic.selected_features].values
        bkg_score, _, _ = evaluate_utils.k_folds_predict(model, x)
        if bkg_score.ndim == 1:
            bkg_score = bkg_score.reshape((-1, 1))
        bkg_scores_dict[bkg_key] = bkg_score
        bkg_weights_dict[bkg_key] = bkg_df["weight"].values
    # prepare data
    # TODO: support data plots
    data_scores = None
    data_weights = None
    if plot_config.apply_data:
        data_key = plot_config.data_key
        data_df = df.loc[df["sample_name"] == data_key, :]
        x = data_df[ic.selected_features].values
        data_scores, _, _ = evaluate_utils.k_folds_predict(model, x)
        data_weights = data_df["weight"].values
    # make plots
    all_nodes = ["sig"] + tc.output_bkg_node_names
    for node_id, node in enumerate(all_nodes):
        fig, ax = plt.subplots(figsize=(16.667, 16.667))
        # plot bkg
        bkg_collect = list()
        bkg_edges = None
        for key, value in bkg_scores_dict.items():
            bkg_bins, bkg_edges = np.histogram(
                value[:, node_id].flatten(),
                bins=plot_config.bins,
                range=(0, 1),
                weights=bkg_weights_dict[key],
                density=plot_config.density,
            )
            bkg = ampl.plot.Background(key, bkg_bins)
            bkg_collect.append(bkg)
        ampl.plot.plot_backgrounds(bkg_collect, bkg_edges, ax=ax)
        # plot sig
        for key, value in sig_scores_dict.items():
            sig_bins, sig_edges = np.histogram(
                value[:, node_id].flatten(),
                bins=plot_config.bins,
                range=(0, 1),
                weights=sig_weights_dict[key],
                density=plot_config.density,
            )
            ampl.plot.plot_signal(key, sig_edges, sig_bins)
        ax.set_xlim(0, 1)
        if plot_config.log:
            ax.set_yscale("log")
            _, y_max = ax.get_ylim()
            ax.set_ylim(
                plot_config.logy_min,
                y_max * np.power(10, np.log10(y_max / plot_config.logy_min) / 2),
            )
        else:
            _, y_max = ax.get_ylim()
            ax.set_ylim(0, y_max * 1.4)
        ax.legend(loc="upper right", ncol=2)
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        fig.savefig(f"{save_dir}/{file_name}_node_{node}.{plot_config.save_format}")

    return 0  # success run


def plot_train_test_compare(
    model_wrapper: hep_model.Model_Base, df, job_config, save_dir
):
    """Plots train/test scores distribution to check overtrain"""
    # initialize
    logger.info("Plotting train/test scores.")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = job_config.apply.cfg_train_test_compare
    model = model_wrapper.get_model()
    all_nodes = ["sig"] + tc.output_bkg_node_names
    # get inputs
    cols = ic.selected_features
    train_index = df["is_train"] == True
    test_index = df["is_train"] == False
    sig_index = (df["is_sig"] == True) & (df["is_mc"] == True)
    bkg_index = (df["is_sig"] == False) & (df["is_mc"] == True)
    xs_train = df.loc[sig_index & train_index, cols].values
    xs_test = df.loc[sig_index & test_index, cols].values
    xs_train_weight = df.loc[sig_index & train_index, ["weight"]].values
    xs_test_weight = df.loc[sig_index & test_index, ["weight"]].values
    xb_train = df.loc[bkg_index & train_index, cols].values
    xb_test = df.loc[bkg_index & test_index, cols].values
    xb_train_weight = df.loc[bkg_index & train_index, ["weight"]].values
    xb_test_weight = df.loc[bkg_index & test_index, ["weight"]].values

    # plot for each nodes
    num_nodes = len(all_nodes)
    xb_train_scores, _, _ = evaluate_utils.k_folds_predict(model, xb_train)
    xs_train_scores, _, _ = evaluate_utils.k_folds_predict(model, xs_train)
    xb_test_scores, _, _ = evaluate_utils.k_folds_predict(model, xb_test)
    xs_test_scores, _, _ = evaluate_utils.k_folds_predict(model, xs_test)
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
            fmt="ok",
            label="background (train)",
            markerfacecolor=plot_config.bkg_color,
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
            fmt="ok",
            label="signal (train)",
            markerfacecolor=plot_config.sig_color,
        )
        # final adjustments
        ax.set_xlim(0, 1)
        if plot_config.log:
            ax.set_yscale("log")
            _, y_max = ax.get_ylim()
            ax.set_ylim(plot_config.logy_min, y_max * np.power(10, np.log10(y_max) / 2))
        else:
            _, y_max = ax.get_ylim()
            ax.set_ylim(0, y_max * 1.4)
        ax.legend(loc="upper right")
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        # Make and show plots
        file_name = f"mva_scores_{all_nodes[node_num]}"
        fig.savefig(save_dir / f"{file_name}.{plot_config.save_format}")
