# -*- coding: utf-8 -*-
"""Functions for making plots"""

import copy
import csv
import json
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import NullFormatter
from scipy.special import softmax
from sklearn.metrics import auc, roc_auc_score, roc_curve

import ROOT
from easy_atlas_plot.plot_utils import plot_utils, th1_tools
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.data_io import feed_box, root_io
from lfv_pdnn.train import train_utils


def calculate_auc(x_plot, y_plot, weights, model, shuffle_col=None, rm_last_two=False):
    """Returns auc of given sig/bkg array."""
    auc_value = []
    if shuffle_col is not None:
        # randomize x values but don't change overall distribution
        x_plot = array_utils.reset_col(x_plot, x_plot, weights, col=shuffle_col)
    y_pred = model.predict(x_plot)
    if y_plot.ndim == 2:
        num_nodes = y_plot.shape[1]
        for node_id in range(num_nodes):
            fpr_dm, tpr_dm, _ = roc_curve(
                y_plot[:, node_id], y_pred[:, node_id], sample_weight=weights
            )
            sort_index = fpr_dm.argsort()
            fpr_dm = fpr_dm[sort_index]
            tpr_dm = tpr_dm[sort_index]
            auc_value.append(auc(fpr_dm, tpr_dm))
    else:
        auc_value = [roc_auc_score(y_plot, y_pred, sample_weight=weights)]
    return auc_value


def get_significances(
    model_wrapper, significance_algo="asimov", multi_class_cut_branch=0
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
    model_meta = model_wrapper.model_meta
    # prepare signal
    sig_key = model_meta["sig_key"]
    sig_arr_temp = feedbox.get_array("xs", "raw", array_key=sig_key)
    sig_arr_temp[:, 0:-2] = train_utils.norarray(
        sig_arr_temp[:, 0:-2],
        average=np.array(model_meta["norm_average"]),
        variance=np.array(model_meta["norm_variance"]),
    )
    sig_selected_arr = train_utils.get_valid_feature(sig_arr_temp)
    sig_predictions = model_wrapper.get_model().predict(sig_selected_arr)
    if sig_predictions.ndim == 2:
        sig_predictions = sig_predictions[:, multi_class_cut_branch]
    sig_predictions_weights = np.reshape(
        feedbox.get_array("xs", "reshape", array_key=sig_key)[:, -1], (-1, 1)
    )
    # prepare background
    bkg_key = model_meta["bkg_key"]
    bkg_arr_temp = feedbox.get_array("xb", "raw", array_key=bkg_key)
    bkg_arr_temp[:, 0:-2] = train_utils.norarray(
        bkg_arr_temp[:, 0:-2],
        average=np.array(model_meta["norm_average"]),
        variance=np.array(model_meta["norm_variance"]),
    )
    bkg_selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
    bkg_predictions = model_wrapper.get_model().predict(bkg_selected_arr)
    if bkg_predictions.ndim == 2:
        bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
    bkg_predictions_weights = np.reshape(
        feedbox.get_array("xb", "reshape", array_key=bkg_key)[:, -1], (-1, 1)
    )
    # prepare thresholds
    bin_array = np.array(range(-1000, 1000))
    thresholds = 1.0 / (1.0 + 1.0 / np.exp(bin_array * 0.02))
    thresholds = np.insert(thresholds, 0, 0)
    # scan
    significances = []
    plot_thresholds = []
    sig_above_threshold = []
    bkg_above_threshold = []
    total_sig_weight = np.sum(sig_predictions_weights)
    total_bkg_weight = np.sum(bkg_predictions_weights)
    for dnn_cut in thresholds:
        sig_ids_passed = sig_predictions > dnn_cut
        total_sig_weights_passed = np.sum(sig_predictions_weights[sig_ids_passed])
        bkg_ids_passed = bkg_predictions > dnn_cut
        total_bkg_weights_passed = np.sum(bkg_predictions_weights[bkg_ids_passed])
        if total_bkg_weights_passed > 0 and total_sig_weights_passed > 0:
            plot_thresholds.append(dnn_cut)
            current_significance = train_utils.calculate_significance(
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
    total_sig_weight = np.sum(sig_predictions_weights)
    total_bkg_weight = np.sum(bkg_predictions_weights)
    return (plot_thresholds, significances, sig_above_threshold, bkg_above_threshold)


def plot_accuracy(
    accuracy_list: list,
    val_accuracy_list: list,
    figsize: tuple = (8, 6),
    show_fig: bool = False,
    save_dir: str = None,
) -> None:
    """Plots accuracy vs training epoch."""
    print("Plotting accuracy curve.")
    fig, ax = plt.subplots(figsize=figsize)
    # Plot
    ax.plot(accuracy_list)
    ax.plot(val_accuracy_list)
    # Config
    ax.set_title("model accuracy")
    ax.set_ylabel("accuracy")
    # ax.set_ylim((0, 1))
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"], loc="lower left")
    ax.grid()
    if show_fig:
        plt.show()
    if save_dir is not None:
        fig.savefig(save_dir + "/eva_accuracy.png")


def plot_auc_text(ax, titles, auc_values):
    """Plots auc information on roc curve."""
    auc_text = "auc values:\n"
    for (title, auc_value) in zip(titles, auc_values):
        auc_text = auc_text + title + ": " + str(auc_value) + "\n"
    auc_text = auc_text[:-1]
    ax.text(
        0.5,
        0.02,
        auc_text,
        bbox={"facecolor": "antiquewhite", "alpha": 0.3},
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
    )


def plot_correlation_matrix(ax, corr_matrix_dict, matrix_key="bkg"):
    # Get matrix
    corr_matrix = corr_matrix_dict[matrix_key]
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
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )


def plot_feature_importance(
    model_wrapper, save_dir, identifier="", log=True, max_feature=16
):
    """Calculates importance of features and sort the feature.

    Definition of feature importance used here can be found in:
    https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data

    """
    print("Plotting feature importance.")
    # Prepare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    num_feature = len(feedbox.selected_features)
    selected_feature_names = np.array(feedbox.selected_features)
    train_test_dict = feedbox.get_train_test_arrays(
        sig_key=model_wrapper.model_meta["sig_key"],
        bkg_key=model_wrapper.model_meta["bkg_key"],
        multi_class_bkgs=model_wrapper.model_hypers["output_bkg_node_names"],
        reset_mass=False,
        output_keys=["x_test", "y_test", "wt_test"],
    )
    x_test = train_test_dict["x_test"]
    y_test = train_test_dict["y_test"]
    weight_test = train_test_dict["wt_test"]
    all_nodes = []
    if y_test.ndim == 2:
        all_nodes = ["sig"] + model_wrapper.model_hypers["output_bkg_node_names"]
    else:
        all_nodes = ["sig"]
    # Make plots
    fig_save_pattern = save_dir + "/importance_" + identifier + "_{}.png"
    if num_feature > 16:
        canvas_height = 16
    else:
        canvas_height = num_feature
    fig_save_path = save_dir + "/importance_" + identifier + "_{}.png"
    base_auc = calculate_auc(x_test, y_test, weight_test, model, rm_last_two=True)
    # Calculate importance
    feature_auc = []
    for num, feature_name in enumerate(selected_feature_names):
        current_auc = calculate_auc(
            x_test, y_test, weight_test, model, shuffle_col=num, rm_last_two=True
        )
        feature_auc.append(current_auc)
    for node_id, node in enumerate(all_nodes):
        print("making importance plot for node:", node)
        fig_save_path = fig_save_pattern.format(node)
        fig, ax = plt.subplots(figsize=(9, canvas_height))
        print("base auc:", base_auc[node_id])
        feature_importance = np.zeros(num_feature)
        for num, feature_name in enumerate(selected_feature_names):
            current_auc = feature_auc[num][node_id]
            feature_importance[num] = (1 - current_auc) / (1 - base_auc[node_id])
            print(feature_name, ":", feature_importance[num])

        # Sort
        sort_list = np.flip(np.argsort(feature_importance))
        sorted_importance = feature_importance[sort_list]
        sorted_names = selected_feature_names[sort_list]
        print("feature importance rank:", sorted_names)
        # Plot
        if num_feature > max_feature:
            num_show = max_feature
        else:
            num_show = num_feature
        ax.barh(
            np.flip(np.arange(num_show)),
            sorted_importance[:num_show],
            align="center",
            alpha=0.5,
            log=log,
        )
        ax.axvline(x=1, ls="--", color="r")
        ax.set_title("feature importance")
        ax.set_yticks(np.arange(num_show))
        ax.set_yticklabels(sorted_names[:num_show])
        fig.savefig(fig_save_path)


def plot_input_distributions(
    model_wrapper,
    apply_data=False,
    figsize=(8, 6),
    style_cfg_path=None,
    show_reshaped=False,
    dnn_cut=None,
    multi_class_cut_branch=0,
    compare_cut_sb_separated=False,
    plot_density=True,
    save_dir=None,
    save_format="png",
):
    """Plots input distributions comparision plots for sig/bkg/data"""
    print("Plotting input distributions.")
    config = {}
    if style_cfg_path is not None:
        with open(style_cfg_path) as plot_config_file:
            config = json.load(plot_config_file)

    model_meta = model_wrapper.model_meta
    sig_key = model_meta["sig_key"]
    bkg_key = model_meta["bkg_key"]
    feedbox = model_wrapper.feedbox
    if show_reshaped:  # validation features not supported in get_reshape yet
        plot_feature_list = model_wrapper.selected_features
        bkg_array, bkg_fill_weights = feedbox.get_reshape("xb", array_key=bkg_key)
        sig_array, sig_fill_weights = feedbox.get_reshape("xs", array_key=sig_key)
    else:
        plot_feature_list = (
            model_wrapper.selected_features + model_wrapper.validation_features
        )
        bkg_array, bkg_fill_weights = feedbox.get_raw(
            "xb", array_key=bkg_key, add_validation_features=True
        )
        sig_array, sig_fill_weights = feedbox.get_raw(
            "xs", array_key=sig_key, add_validation_features=True
        )
    if plot_density:
        bkg_fill_weights = bkg_fill_weights / np.sum(bkg_fill_weights)
        sig_fill_weights = sig_fill_weights / np.sum(sig_fill_weights)
    # get fill weights with dnn cut
    if dnn_cut is not None:
        assert dnn_cut >= 0 and dnn_cut <= 1, "dnn_cut out or range."
        model_meta = model_wrapper.model_meta
        # prepare signal
        sig_selected_arr, _ = feedbox.get_reweight(
            "xs", array_key=sig_key, reset_mass=False
        )
        sig_predictions = model_wrapper.get_model().predict(sig_selected_arr)
        if sig_predictions.ndim == 2:
            sig_predictions = sig_predictions[:, multi_class_cut_branch]
        sig_cut_index = array_utils.get_cut_index(sig_predictions, [dnn_cut], ["<"])
        sig_fill_weights_dnn = sig_fill_weights.copy()
        sig_fill_weights_dnn[sig_cut_index] = 0
        # prepare background
        bkg_selected_arr, _ = feedbox.get_reweight(
            "xb", array_key=bkg_key, reset_mass=False
        )
        bkg_predictions = model_wrapper.get_model().predict(bkg_selected_arr)
        if bkg_predictions.ndim == 2:
            bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
        bkg_cut_index = array_utils.get_cut_index(bkg_predictions, [dnn_cut], ["<"])
        bkg_fill_weights_dnn = bkg_fill_weights.copy()
        bkg_fill_weights_dnn[bkg_cut_index] = 0
        # normalize weights for density plots
        if plot_density:
            bkg_fill_weights_dnn = bkg_fill_weights_dnn / np.sum(bkg_fill_weights_dnn)
            sig_fill_weights_dnn = sig_fill_weights_dnn / np.sum(sig_fill_weights_dnn)
    # prepare thresholds
    for feature_id, feature in enumerate(plot_feature_list):
        bkg_fill_array = np.reshape(bkg_array[:, feature_id], (-1, 1))
        sig_fill_array = np.reshape(sig_array[:, feature_id], (-1, 1))
        # prepare background histogram
        hist_bkg = th1_tools.TH1FTool(
            feature + "_bkg", "bkg", nbin=100, xlow=-20, xup=20
        )
        hist_bkg.reinitial_hist_with_fill_array(bkg_fill_array)
        hist_bkg.fill_hist(bkg_fill_array, bkg_fill_weights)
        hist_bkg.set_config(config)
        hist_bkg.update_config("hist", "SetLineColor", 4)
        hist_bkg.update_config("hist", "SetFillStyle", 3354)
        hist_bkg.update_config("hist", "SetFillColor", ROOT.kBlue)
        hist_bkg.update_config("x_axis", "SetTitle", feature)
        hist_bkg.apply_config()
        # prepare signal histogram
        hist_sig = th1_tools.TH1FTool(
            feature + "_sig", "sig", nbin=100, xlow=-20, xup=20
        )
        hist_sig.reinitial_hist_with_fill_array(sig_fill_array)
        hist_sig.fill_hist(sig_fill_array, sig_fill_weights)
        hist_sig.set_config(config)
        hist_sig.update_config("hist", "SetLineColor", 2)
        hist_sig.update_config("hist", "SetFillStyle", 3354)
        hist_sig.update_config("hist", "SetFillColor", ROOT.kRed)
        hist_sig.update_config("x_axis", "SetTitle", feature)
        hist_sig.apply_config()
        # prepare bkg/sig histograms with dnn cut
        if dnn_cut is not None:
            hist_bkg_dnn = th1_tools.TH1FTool(
                feature + "_bkg_cut_dnn", "bkg_cut_dnn", nbin=100, xlow=-20, xup=20
            )
            hist_bkg_dnn.reinitial_hist_with_fill_array(bkg_fill_array)
            hist_bkg_dnn.fill_hist(bkg_fill_array, bkg_fill_weights_dnn)
            hist_bkg_dnn.set_config(config)
            hist_bkg_dnn.update_config("hist", "SetLineColor", 4)
            hist_bkg_dnn.update_config("hist", "SetFillStyle", 3001)
            hist_bkg_dnn.update_config("hist", "SetFillColor", ROOT.kBlue)
            hist_bkg_dnn.update_config("x_axis", "SetTitle", feature)
            hist_bkg_dnn.apply_config()
            hist_sig_dnn = th1_tools.TH1FTool(
                feature + "_sig_cut_dnn", "sig_cut_dnn", nbin=100, xlow=-20, xup=20
            )
            hist_sig_dnn.reinitial_hist_with_fill_array(sig_fill_array)
            hist_sig_dnn.fill_hist(sig_fill_array, sig_fill_weights_dnn)
            hist_sig_dnn.set_config(config)
            hist_sig_dnn.update_config("hist", "SetLineColor", 2)
            hist_sig_dnn.update_config("hist", "SetFillStyle", 3001)
            hist_sig_dnn.update_config("hist", "SetFillColor", ROOT.kRed)
            hist_sig_dnn.update_config("x_axis", "SetTitle", feature)
            hist_sig_dnn.apply_config()
        # combined histograms
        if not compare_cut_sb_separated:
            if dnn_cut is not None:
                hist_col = th1_tools.HistCollection(
                    [hist_bkg_dnn, hist_sig_dnn],
                    name=feature,
                    title="input var: " + feature,
                )
            else:
                hist_col = th1_tools.HistCollection(
                    [hist_bkg, hist_sig], name=feature, title="input var: " + feature
                )
            hist_col.draw(
                draw_options="hist", legend_title="legend", draw_norm=plot_density,
            )
            hist_col.save(
                save_dir=save_dir, save_file_name=feature, save_format=save_format
            )
        else:
            # bkg
            plot_title = "input var: " + feature + "_bkg"
            plot_canvas = ROOT.TCanvas(plot_title, plot_title, 800, 800)
            plot_pad_compare = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
            plot_pad_compare.SetBottomMargin(0)
            plot_pad_compare.SetGridx()
            plot_pad_compare.Draw()
            plot_pad_compare.cd()
            hist_col_bkg = th1_tools.HistCollection(
                [hist_bkg_dnn, hist_bkg],
                name=feature + "_bkg",
                title="input var: " + feature,
                canvas=plot_pad_compare,
            )
            x_min, x_max = hist_col_bkg.draw(  ##>> hot fix
                draw_options="hist",
                legend_title="legend",
                draw_norm=False,
                auto_xrange_method="major",
            )
            plot_canvas.cd()
            plot_pad_ratio = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.25)
            plot_pad_ratio.SetTopMargin(0)
            plot_pad_ratio.SetGridx()
            plot_pad_ratio.Draw()
            ratio_plot = th1_tools.RatioPlot(
                hist_bkg_dnn,
                hist_bkg,
                x_title="DNN Score",
                y_title="bkg(cut-dnn)/bkg",
                canvas=plot_pad_ratio,
            )
            ratio_plot.update_style_cfg("hist", "SetMinimum", 0)
            ratio_plot.update_style_cfg("hist", "SetMaximum", 1)
            ratio_plot.update_style_cfg(
                "x_axis", "SetRange", [x_min, x_max]
            )  ## >> hot fix
            ratio_plot.draw(draw_err=False, draw_base_line=False)
            hist_col_bkg.save(
                save_dir=save_dir,
                save_file_name=feature + "_bkg",
                save_format=save_format,
            )
            plot_canvas.SaveAs(save_dir + "/" + feature + "_bkg." + save_format)

            # sig
            plot_title = "input var: " + feature + "_sig"
            plot_canvas = ROOT.TCanvas(plot_title, plot_title, 800, 800)
            plot_pad_compare = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
            plot_pad_compare.SetBottomMargin(0)
            plot_pad_compare.SetGridx()
            plot_pad_compare.Draw()
            plot_pad_compare.cd()
            hist_col_sig = th1_tools.HistCollection(
                [hist_sig_dnn, hist_sig],
                name=feature + "_sig",
                title="input var: " + feature,
                canvas=plot_pad_compare,
            )
            x_min, x_max = hist_col_sig.draw(  ##>> hot fix
                draw_options="hist",
                legend_title="legend",
                draw_norm=False,
                auto_xrange_method="major",
            )
            plot_canvas.cd()
            plot_pad_ratio = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.25)
            plot_pad_ratio.SetTopMargin(0)
            plot_pad_ratio.SetGridx()
            plot_pad_ratio.Draw()
            ratio_plot = th1_tools.RatioPlot(
                hist_sig_dnn,
                hist_sig,
                x_title="DNN Score",
                y_title="sig(cut-dnn)/sig",
                canvas=plot_pad_ratio,
            )
            ratio_plot.update_style_cfg("hist", "SetMinimum", 0)
            ratio_plot.update_style_cfg("hist", "SetMaximum", 1)
            ratio_plot.update_style_cfg(
                "x_axis", "SetRange", [x_min, x_max]
            )  ## >> hot fix
            ratio_plot.draw(draw_err=False, draw_base_line=False)
            plot_canvas.SaveAs(save_dir + "/" + feature + "_sig." + save_format)


def plot_overtrain_check(
    model_wrapper,
    figsize: tuple = (8, 6),
    save_dir: str = None,
    bins: int = 50,
    x_range: tuple = (-0.1, 1.1),
    log: bool = True,
    reset_mass: bool = False,
):
    """Plots train/test scores distribution to check overtrain"""
    print("Plotting train/test scores (original mass).")
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta
    sig_key = model_meta["sig_key"]
    bkg_key = model_meta["bkg_key"]
    all_nodes = ["sig"] + model_wrapper.model_hypers["output_bkg_node_names"]
    train_test_dict = feedbox.get_train_test_arrays(
        sig_key=sig_key,
        bkg_key=bkg_key,
        multi_class_bkgs=model_wrapper.model_hypers["output_bkg_node_names"],
        reset_mass=reset_mass,
        output_keys=[
            "xs_train",
            "xs_test",
            "wts_train",
            "wts_test",
            "xb_train",
            "xb_test",
            "wtb_train",
            "wtb_test",
        ],
    )
    xs_train = train_test_dict["xs_train"]
    xs_test = train_test_dict["xs_test"]
    xs_train_weight = train_test_dict["wts_train"]
    xs_test_weight = train_test_dict["wts_test"]
    xb_train = train_test_dict["xb_train"]
    xb_test = train_test_dict["xb_test"]
    xb_train_weight = train_test_dict["wtb_train"]
    xb_test_weight = train_test_dict["wtb_test"]
    # plot for each nodes
    num_nodes = len(all_nodes)
    xb_train_scores = model.predict(xb_train)
    xs_train_scores = model.predict(xs_train)
    xb_test_scores = model.predict(xb_test)
    xs_test_scores = model.predict(xs_test)
    for node_num in range(num_nodes):
        fig, ax = plt.subplots(figsize=figsize)
        # plot scores
        plot_scores(
            ax,
            xb_test_scores[:, node_num],
            xb_test_weight,
            xs_test_scores[:, node_num],
            xs_test_weight,
            apply_data=False,
            title="over training check",
            bkg_label="b-test",
            sig_label="s-test",
            bins=bins,
            range=x_range,
            density=True,
            log=log,
        )
        make_bar_plot(
            ax,
            xb_train_scores[:, node_num],
            "b-train",
            weights=xb_train_weight,
            bins=bins,
            range=x_range,
            density=True,
            use_error=True,
            color="darkblue",
            fmt=".",
        )
        make_bar_plot(
            ax,
            xs_train_scores[:, node_num],
            "s-train",
            weights=xs_train_weight,
            bins=bins,
            range=x_range,
            density=True,
            use_error=True,
            color="maroon",
            fmt=".",
        )
        # Make and show plots
        if save_dir is not None:
            if reset_mass:
                fig_name = "/eva_overtrain_original_mass_{}.png".format(
                    all_nodes[node_num]
                )
            else:
                fig_name = "/eva_overtrain_reset_mass_{}.png".format(
                    all_nodes[node_num]
                )
            fig.savefig(save_dir + "/" + fig_name)


def plot_loss(
    loss_list: list,
    val_loss_list: list,
    figsize: tuple = (12, 9),
    show_fig: bool = False,
    save_dir: str = None,
) -> None:
    """Plots loss vs training epoch."""
    print("Plotting loss curve.")
    fig, ax = plt.subplots(figsize=figsize)
    # Plot
    ax.plot(loss_list)
    ax.plot(val_loss_list)
    # Config
    ax.set_title("model loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(["train", "val"], loc="lower left")
    ax.grid()
    if show_fig:
        plt.show()
    if save_dir is not None:
        fig.savefig(save_dir + "/eva_loss.png")


def plot_roc(
    ax,
    x,
    y,
    weights,
    model,
    node_num=0,
    color="blue",
    linestyle="solid",
    yscal="linear",
    ylim=(0, 1),
):
    """Plots roc curve on given axes."""
    # Get data
    y_pred = model.predict(x)
    fpr_dm, tpr_dm, _ = roc_curve(
        y[:, node_num], y_pred[:, node_num], sample_weight=weights
    )
    # Make plots
    ax.plot(fpr_dm, tpr_dm, color=color, linestyle=linestyle)
    ax.set_title("roc curve")
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_yscale(yscal)
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Calculate auc and return parameters
    auc_value = roc_auc_score(y[:, node_num], y_pred[:, node_num])
    return auc_value, fpr_dm, tpr_dm


def plot_scores(
    ax,
    bkg_scores,
    bkg_weight,
    sig_scores,
    sig_weight,
    data_scores=None,
    data_weight=None,
    apply_data=False,
    title="scores",
    bkg_label="bkg",
    sig_label="sig",
    bins=50,
    range=(-0.25, 1.25),
    density=True,
    log=False,
):
    """Plots score distribution for signal and background."""
    ax.hist(
        bkg_scores,
        weights=bkg_weight,
        bins=bins,
        range=range,
        histtype="step",
        label=bkg_label,
        density=density,
        log=log,
        facecolor="blue",
        edgecolor="darkblue",
        alpha=0.5,
        fill=True,
    )
    ax.hist(
        sig_scores,
        weights=sig_weight,
        bins=bins,
        range=range,
        histtype="step",
        label=sig_label,
        density=density,
        log=log,
        facecolor="red",
        edgecolor="maroon",
        hatch="///",
        alpha=1,
        fill=False,
    )
    if apply_data:
        make_bar_plot(
            ax,
            data_scores,
            "data",
            weights=np.reshape(data_weight, (-1, 1)),
            bins=bins,
            range=range,
            density=density,
            use_error=False,
        )
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()


# def plot_scores_separate(
#    ax,
#    model_wrapper,
#    bkg_dict,
#    bkg_plot_key_list=None,
#    sig_arr=None,
#    sig_weights=None,
#    apply_data=False,
#    data_arr=None,
#    data_weight=None,
#    plot_title="all input scores",
#    bins=50,
#    range=(-0.25, 1.25),
#    density=True,
#    log=False,
# ):
#    """Plots training score distribution for different background with matplotlib.
#
#    Note:
#        bkg_plot_key_list can be used to adjust order of background sample
#        stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
#        'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top
#
#    """
#    print("Plotting scores with bkg separated.")
#    predict_arr_list = []
#    predict_arr_weight_list = []
#    model = model_wrapper.get_model()
#    feedbox = model_wrapper.feedbox
#    model_meta = model_wrapper.model_meta
#    # plot background
#    if (type(bkg_plot_key_list) is not list) or len(bkg_plot_key_list) == 0:
#        # prepare plot key list sort with total weight by default
#        original_keys = list(bkg_dict.keys())
#        total_weight_list = []
#        for key in original_keys:
#            total_weight = np.sum((bkg_dict[key])[:, -1])
#            total_weight_list.append(total_weight)
#        sort_indexes = np.argsort(np.array(total_weight_list))
#        bkg_plot_key_list = [original_keys[index] for index in sort_indexes]
#    for arr_key in bkg_plot_key_list:
#        bkg_arr_temp = bkg_dict[arr_key].copy()
#        bkg_arr_temp[:, 0:-2] = train_utils.norarray(
#            bkg_arr_temp[:, 0:-2],
#            average=np.array(model_meta["norm_average"]),
#            variance=np.array(model_meta["norm_variance"]),
#        )
#        selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
#        predict_arr_list.append(np.array(model.predict(selected_arr)))
#        predict_arr_weight_list.append(bkg_arr_temp[:, -1])
#    try:
#        ax.hist(
#            np.transpose(predict_arr_list),
#            bins=bins,
#            range=range,
#            weights=np.transpose(predict_arr_weight_list),
#            histtype="bar",
#            label=bkg_plot_key_list,
#            density=density,
#            stacked=True,
#        )
#    except:
#        ax.hist(
#            predict_arr_list[0],
#            bins=bins,
#            range=range,
#            weights=predict_arr_weight_list[0],
#            histtype="bar",
#            label=bkg_plot_key_list,
#            density=density,
#            stacked=True,
#        )
#    # plot signal
#    if sig_arr is None:
#        sig_key = model_meta["sig_key"]
#        xs_reshape = feedbox.get_array("xs", "reshape", array_key=sig_key)
#        selected_arr = train_utils.get_valid_feature(xs_reshape)
#        predict_arr = model.predict(selected_arr)
#        predict_weight_arr = xs_reshape[:, -1]
#    else:
#        sig_arr_temp = sig_arr.copy()
#        sig_arr_temp[:, 0:-2] = train_utils.norarray(
#            sig_arr[:, 0:-2],
#            average=np.array(model_meta["norm_average"]),
#            variance=np.array(model_meta["norm_variance"]),
#        )
#        selected_arr = train_utils.get_valid_feature(sig_arr_temp)
#        predict_arr = np.array(model.predict(selected_arr))
#        predict_weight_arr = sig_arr_temp[:, -1]
#    ax.hist(
#        predict_arr,
#        bins=bins,
#        range=range,
#        weights=predict_weight_arr,
#        histtype="step",
#        label="sig",
#        density=density,
#    )
#    # plot data
#    if apply_data:
#        data_key = model_meta["data_key"]
#        if data_arr is None:
#            xd = feedbox.get_array("xd", "raw", array_key=data_key)
#            xd_selected = feedbox.get_array(
#                "xd", "selected", array_key=data_key, reset_mass=False
#            )
#            data_arr = xd_selected
#            data_weight = xd[:, -1]
#        make_bar_plot(
#            ax,
#            model.predict(data_arr),
#            "data",
#            weights=np.reshape(data_weight, (-1, 1)),
#            bins=bins,
#            range=range,
#            density=density,
#            use_error=False,
#        )
#    ax.set_title(plot_title)
#    ax.legend(loc="upper right")
#    ax.set_xlabel("Output score")
#    ax.set_ylabel("arb. unit")
#    ax.grid()
#    if log is True:
#        ax.set_yscale("log")
#        ax.set_title(plot_title + "(log)")
#    else:
#        ax.set_title(plot_title + "(lin)")


def plot_scores_separate_root(
    model_wrapper,
    bkg_plot_key_list,
    apply_data=False,
    apply_data_range=None,
    plot_title="all input scores",
    bins=50,
    x_range=(-0.25, 1.25),
    scale_sig=False,
    density=True,
    save_plot=False,
    save_dir=None,
    save_file_name=None,
):
    """Plots training score distribution for different background with ROOT

    Note:
        bkg_plot_key_list can be used to adjust order of background sample 
        stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
        'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top

    """
    print("Plotting scores with bkg separated with ROOT.")
    all_nodes = ["sig"] + model_wrapper.model_hypers["output_bkg_node_names"]
    num_nodes = len(all_nodes)

    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta
    bkg_dict = feedbox.xb_dict

    # prepare background
    if (type(bkg_plot_key_list) is not list) or len(bkg_plot_key_list) == 0:
        # prepare plot key list sort with total weight by default
        original_keys = list(bkg_dict.keys())
        total_weight_list = []
        for key in original_keys:
            _, weights = feedbox.get_raw("xb", array_key=key)
            total_weight = np.sum(weights)
            total_weight_list.append(total_weight)
        sort_indexes = np.argsort(np.array(total_weight_list))
        bkg_plot_key_list = [original_keys[index] for index in sort_indexes]
    bkg_predict_dict = {}
    bkg_weight_dict = {}
    for arr_key in bkg_plot_key_list:
        predict_arr, predict_weight_arr = feedbox.get_reshape("xb", array_key=arr_key)
        bkg_predict_dict[arr_key] = model.predict(predict_arr)
        bkg_weight_dict[arr_key] = predict_weight_arr
    # prepare signal
    sig_key = model_meta["sig_key"]
    sig_arr_temp, sig_weight = feedbox.get_reshape("xs", array_key=sig_key)
    sig_predict = model.predict(sig_arr_temp)
    # prepare data
    if apply_data:
        data_key = model_meta["data_key"]
        data_arr_temp, data_weight = feedbox.get_reshape("xd", array_key=data_key)
        data_predict = model.predict(data_arr_temp)

    # plot
    for node_num in range(num_nodes):
        plot_canvas = ROOT.TCanvas(plot_title, plot_title, 600, 600)
        plot_pad_score = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        plot_pad_score.SetBottomMargin(0)
        plot_pad_score.SetGridx()
        plot_pad_score.Draw()
        plot_pad_score.cd()
        hist_list = []
        # plot background
        for arr_key in bkg_plot_key_list:
            th1_temp = th1_tools.TH1FTool(
                arr_key, arr_key, nbin=bins, xlow=x_range[0], xup=x_range[1]
            )
            th1_temp.fill_hist(
                bkg_predict_dict[arr_key][:, node_num], bkg_weight_dict[arr_key]
            )
            hist_list.append(th1_temp)
        hist_stacked_bkgs = th1_tools.THStackTool(
            "bkg stack plot", plot_title, hist_list, canvas=plot_pad_score
        )
        hist_stacked_bkgs.set_palette("kPastel")
        hist_stacked_bkgs.draw("pfc hist")
        hist_stacked_bkgs.get_hstack().GetYaxis().SetTitle("events/bin")
        hist_stacked_bkgs.get_hstack().GetXaxis().SetTitle("dnn score")
        hist_bkg_total = hist_stacked_bkgs.get_added_hist()
        total_weight_bkg = hist_bkg_total.get_hist().GetSumOfWeights()
        # plot signal
        if scale_sig:
            sig_title = "sig-scaled"
        else:
            sig_title = "sig"
        hist_sig = th1_tools.TH1FTool(
            "sig added",
            sig_title,
            nbin=bins,
            xlow=x_range[0],
            xup=x_range[1],
            canvas=plot_pad_score,
        )
        hist_sig.fill_hist(sig_predict[:, node_num], sig_weight)
        total_weight_sig = hist_sig.get_hist().GetSumOfWeights()
        if scale_sig:
            total_weight = hist_stacked_bkgs.get_total_weights()
            scale_factor = total_weight / hist_sig.get_hist().GetSumOfWeights()
            hist_sig.get_hist().Scale(scale_factor)
        hist_sig.update_config("hist", "SetLineColor", ROOT.kRed)
        # set proper y range
        maximum_y = max(
            plot_utils.get_highest_bin_value(hist_list),
            plot_utils.get_highest_bin_value(hist_sig),
        )
        hist_stacked_bkgs.get_hstack().SetMaximum(1.2 * maximum_y)
        hist_stacked_bkgs.get_hstack().SetMinimum(0.1)
        hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelFont(43)
        hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelSize(15)
        hist_sig.draw("same hist")
        # plot data if required
        total_weight_data = 0
        if apply_data:
            hist_data = th1_tools.TH1FTool(
                "data added",
                "data",
                nbin=bins,
                xlow=x_range[0],
                xup=x_range[1],
                canvas=plot_pad_score,
            )
            hist_data.fill_hist(data_predict[:, node_num], data_weight)
            hist_data.update_config("hist", "SetMarkerStyle", ROOT.kFullCircle)
            hist_data.update_config("hist", "SetMarkerColor", ROOT.kBlack)
            hist_data.update_config("hist", "SetMarkerSize", 0.8)
            if apply_data_range is not None:
                hist_data.get_hist().GetXaxis().SetRangeUser(
                    apply_data_range[0], apply_data_range[1]
                )
            hist_data.draw("same e1")
            total_weight_data = hist_data.get_hist().GetSumOfWeights()
            hist_data.build_legend(0.4, 0.7, 0.6, 0.9)
        else:
            total_weight_data = 0

        # ratio plot
        plot_canvas.cd()
        plot_pad_ratio = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.25)
        plot_pad_ratio.SetTopMargin(0)
        plot_pad_ratio.SetGridx()
        plot_pad_ratio.Draw()
        if apply_data:
            hist_numerator = hist_data
            hist_denominator = hist_bkg_total
        else:
            hist_numerator = hist_bkg_total
            hist_denominator = hist_bkg_total
        ratio_plot = th1_tools.RatioPlot(
            hist_numerator,
            hist_denominator,
            x_title="DNN Score",
            y_title="data/bkg",
            canvas=plot_pad_ratio,
        )
        ratio_plot.draw()

        # show & save total weight info
        model_wrapper.total_weight_sig = total_weight_sig
        print("sig total weight:", total_weight_sig)
        model_wrapper.total_weight_bkg = total_weight_bkg
        print("bkg total weight:", total_weight_bkg)
        model_wrapper.total_weight_data = total_weight_data
        print("data total weight:", total_weight_data)
        # save plot
        if save_plot:
            plot_canvas.SaveAs(
                save_dir + "/" + save_file_name + "_" + all_nodes[node_num] + "_lin.png"
            )
            plot_pad_score.SetLogy(2)
            plot_canvas.SaveAs(
                save_dir + "/" + save_file_name + "_" + all_nodes[node_num] + "_log.png"
            )


def plot_significance_scan(
    ax, model_wrapper, save_dir=".", significance_algo="asimov", suffix=""
) -> None:
    """Shows significance change with threshold.

    Note:
        significance is calculated by s/sqrt(b)
    """
    print("Plotting significance scan.")

    (
        plot_thresholds,
        significances,
        sig_above_threshold,
        bkg_above_threshold,
    ) = get_significances(model_wrapper, significance_algo=significance_algo)

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
    original_significance = train_utils.calculate_significance(
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


def plot_multi_class_roc(
    model_wrapper, figsize: tuple = (8, 6), save_dir: str = None,
):
    """Plots roc curve."""
    print("Plotting train/test roc curve.")
    fig, ax = plt.subplots(figsize=figsize)

    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta
    sig_key = model_meta["sig_key"]
    bkg_key = model_meta["bkg_key"]
    all_nodes = ["sig"] + model_wrapper.model_hypers["output_bkg_node_names"]
    train_test_dict = feedbox.get_train_test_arrays(
        sig_key=sig_key,
        bkg_key=bkg_key,
        multi_class_bkgs=model_wrapper.model_hypers["output_bkg_node_names"],
        reset_mass=False,
        output_keys=["x_train", "x_test", "y_train", "y_test", "wt_train", "wt_test",],
    )
    x_train_original_mass = train_test_dict["x_train"]
    x_test_original_mass = train_test_dict["x_test"]
    y_train_original_mass = train_test_dict["y_train"]
    y_test_original_mass = train_test_dict["y_test"]
    wt_train_original_mass = train_test_dict["wt_train"]
    wt_test_original_mass = train_test_dict["wt_test"]
    num_nodes = len(all_nodes)
    color_map = plt.get_cmap("Pastel1")
    auc_labels = []
    auc_contents = []
    for node_num in range(num_nodes):
        color = color_map(float(node_num) / num_nodes)
        # plot roc for train dataset without reseting mass
        auc_train_original, _, _ = plot_roc(
            ax,
            x_train_original_mass,
            y_train_original_mass,
            wt_train_original_mass,
            model,
            node_num=node_num,
            color=color,
            linestyle="dashed",
        )
        # plot roc for test dataset without reseting mass
        auc_test_original, _, _ = plot_roc(
            ax,
            x_test_original_mass,
            y_test_original_mass,
            wt_test_original_mass,
            model,
            node_num=node_num,
            color=color,
            linestyle="solid",
        )
        auc_labels += ["tr_" + all_nodes[node_num], "te_" + all_nodes[node_num]]
        auc_contents += [round(auc_train_original, 5), round(auc_test_original, 5)]

    # Show auc value:
    plot_auc_text(ax, auc_labels, auc_contents)
    # Extra plot config
    ax.legend(auc_labels, loc="lower right")
    ax.grid()
    # Collect meta data
    auc_dict = {}
    auc_dict["auc_train_original"] = auc_train_original
    auc_dict["auc_test_original"] = auc_test_original
    # Make plots
    if save_dir is not None:
        ax.set_ylim(0, 1)
        ax.set_yscale("linear")
        fig.savefig(save_dir + "/eva_roc_linear.png")
        ax.set_ylim(0.1, 1 - 1e-4)
        ax.set_yscale("logit")
        fig.savefig(save_dir + "/eva_roc_logit.png")
    return auc_dict


def make_bar_plot(
    ax,
    datas,
    labels: list,
    weights,
    bins: int,
    range: tuple,
    title: str = None,
    x_lable: str = None,
    y_lable: str = None,
    x_unit: str = None,
    x_scale: float = None,
    density: bool = False,
    use_error: bool = False,
    color: str = "black",
    fmt: str = ".k",
) -> None:
    """Plot with verticle bar, can be used for data display.

        Note:
        According to ROOT:
        "The error per bin will be computed as sqrt(sum of squares of weight) for each bin."

    """
    plt.ioff()
    # Check input
    data_1dim = np.array([])
    weight_1dim = np.array([])
    if isinstance(datas, np.ndarray):
        datas = [datas]
        weights = [weights]
    for data, weight in zip(datas, weights):
        assert isinstance(data, np.ndarray), "datas element should be numpy array."
        assert isinstance(weight, np.ndarray), "weights element should be numpy array."
        assert (
            data.shape == weight.shape
        ), "Input weights should be None or have same type as arrays."
        if len(data_1dim) == 0:
            data_1dim = data
            weight_1dim = weight
        else:
            data_1dim = np.concatenate((data_1dim, data))
            weight_1dim = np.concatenate((weight_1dim, weight))

    # Scale x axis
    if x_scale is not None:
        data_1dim = data_1dim * x_scale
    # Make bar plot
    # get bin error and edges
    plot_ys, _ = np.histogram(
        data_1dim, bins=bins, range=range, weights=weight_1dim, density=density
    )
    sum_weight_squares, bin_edges = np.histogram(
        data_1dim, bins=bins, range=range, weights=np.power(weight_1dim, 2)
    )
    if density:
        error_scale = 1 / (np.sum(weight_1dim) * (range[1] - range[0]) / bins)
        errors = np.sqrt(sum_weight_squares) * error_scale
    else:
        errors = np.sqrt(sum_weight_squares)
    # Only plot ratio when bin is not 0.
    bin_centers = np.array([])
    bin_ys = np.array([])
    bin_yerrs = np.array([])
    for i, y1 in enumerate(plot_ys):
        if y1 != 0:
            ele_center = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1])])
            bin_centers = np.concatenate((bin_centers, ele_center))
            ele_y = np.array([y1])
            bin_ys = np.concatenate((bin_ys, ele_y))
            ele_yerr = np.array([errors[i]])
            bin_yerrs = np.concatenate((bin_yerrs, ele_yerr))
    # plot bar
    bin_size = bin_edges[1] - bin_edges[0]
    if use_error:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=bin_yerrs,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    else:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=None,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    # Config
    if title is not None:
        ax.set_title(title)
    if x_lable is not None:
        if x_unit is not None:
            ax.set_xlabel(x_lable + "/" + x_unit)
        else:
            ax.set_xlabel(x_lable)
    else:
        if x_unit is not None:
            ax.set_xlabel(x_unit)
    if y_lable is not None:
        ax.set_ylabel(y_lable)
    if range is not None:
        ax.axis(xmin=range[0], xmax=range[1])
    ax.legend(loc="upper right")


def plot_2d_density(
    job_wrapper, save_plot=False, save_dir=None, save_file_name="2d_density",
):
    """Plots 2D hist to see event distribution of signal and backgrounds events.

    x-axis will be dnn scores, y-axis will be mass parameter, z-axis is total
    weight shown by color

    """
    model_wrapper = job_wrapper.model_wrapper
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta
    # plot signal
    sig_key = model_meta["sig_key"]
    sig_arr_original = feedbox.get_array("xs", "raw", array_key=sig_key)
    sig_arr_temp = feedbox.get_array("xs", "raw", array_key=sig_key)
    sig_arr_temp[:, 0:-2] = train_utils.norarray(
        sig_arr_temp[:, 0:-2],
        average=np.array(model_meta["norm_average"]),
        variance=np.array(model_meta["norm_variance"]),
    )
    selected_arr = train_utils.get_valid_feature(sig_arr_temp)
    predict_arr = model_wrapper.get_model().predict(selected_arr)
    if predict_arr.ndim == 2:
        predict_arr = predict_arr[:, 0]
    mass_index = job_wrapper.selected_features.index(job_wrapper.reset_feature_name)
    x = predict_arr
    y = sig_arr_original[:, mass_index]
    w = sig_arr_temp[:, -1]
    ## make plot
    plot_canvas = ROOT.TCanvas("2d_density_sig", "2d_density_sig", 1200, 900)
    hist_sig = th1_tools.TH2FTool(
        "2d_density_sig",
        "2d_density_sig",
        nbinx=50,
        xlow=0,
        xup=1.0,
        nbiny=50,
        ylow=min(y),
        yup=max(y),
    )
    hist_sig.fill_hist(fill_array_x=x, fill_array_y=y, weight_array=w)
    hist_sig.set_canvas(plot_canvas)
    hist_sig.set_palette("kBird")
    hist_sig.update_config("hist", "SetStats", 0)
    hist_sig.update_config("x_axis", "SetTitle", "dnn score")
    hist_sig.update_config("y_axis", "SetTitle", "mass")
    hist_sig.draw("colz")
    hist_sig.save(save_dir=save_dir, save_file_name=save_file_name + "_sig")
    # plot background
    bkg_key = model_meta["bkg_key"]
    bkg_arr_original = feedbox.get_array("xb", "raw", array_key=bkg_key)
    bkg_arr_temp = feedbox.get_array("xb", "raw", array_key=bkg_key)
    bkg_arr_temp[:, 0:-2] = train_utils.norarray(
        bkg_arr_temp[:, 0:-2],
        average=np.array(model_meta["norm_average"]),
        variance=np.array(model_meta["norm_variance"]),
    )
    selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
    predict_arr = model_wrapper.get_model().predict(selected_arr)
    if predict_arr.ndim == 2:
        predict_arr = predict_arr[:, 0]
    mass_index = job_wrapper.selected_features.index(job_wrapper.reset_feature_name)
    x = predict_arr
    y = bkg_arr_original[:, mass_index]
    w = bkg_arr_temp[:, -1]
    ## make plot
    plot_canvas = ROOT.TCanvas("2d_density_bkg", "2d_density_bkg", 1200, 900)
    hist_bkg = th1_tools.TH2FTool(
        "2d_density_bkg",
        "2d_density_bkg",
        nbinx=50,
        xlow=0,
        xup=1.0,
        nbiny=50,
        ylow=min(y),
        yup=max(y),
    )
    hist_bkg.fill_hist(fill_array_x=x, fill_array_y=y, weight_array=w)
    hist_bkg.set_canvas(plot_canvas)
    hist_bkg.set_palette("kBird")
    hist_bkg.update_config("hist", "SetStats", 0)
    hist_bkg.update_config("x_axis", "SetTitle", "dnn score")
    hist_bkg.update_config("y_axis", "SetTitle", "mass")
    hist_bkg.draw("colz")
    hist_bkg.save(save_dir=save_dir, save_file_name=save_file_name + "_bkg")


def plot_2d_significance_scan(
    job_wrapper,
    save_plot=False,
    save_dir=None,
    save_file_name="2d_significance",
    cut_ranges_dn=None,
    cut_ranges_up=None,
):
    """Makes 2d map of significance"""
    dnn_cut_list = np.arange(0.8, 1.0, 0.02)
    w_inputs = []
    print("Making 2d significance scan.")
    sig_dict = root_io.get_npy_individuals(
        job_wrapper.npy_path,
        job_wrapper.campaign,
        job_wrapper.region,
        job_wrapper.channel,
        job_wrapper.sig_list,
        job_wrapper.selected_features,
        "sig",
        cut_features=job_wrapper.cut_features,
        cut_values=job_wrapper.cut_values,
        cut_types=job_wrapper.cut_types,
    )
    bkg_dict = root_io.get_npy_individuals(
        job_wrapper.npy_path,
        job_wrapper.campaign,
        job_wrapper.region,
        job_wrapper.channel,
        job_wrapper.bkg_list,
        job_wrapper.selected_features,
        "bkg",
        cut_features=job_wrapper.cut_features,
        cut_values=job_wrapper.cut_values,
        cut_types=job_wrapper.cut_types,
    )
    for sig_id, scan_sig_key in enumerate(job_wrapper.sig_list):
        xs = array_utils.modify_array(sig_dict[scan_sig_key], select_channel=True)
        m_cut_name = job_wrapper.reset_feature_name
        if cut_ranges_dn is None or len(cut_ranges_dn) == 0:
            means, variances = train_utils.get_mean_var(
                xs[:, 0:-2], axis=0, weights=xs[:, -1]
            )
            m_index = job_wrapper.selected_features.index(m_cut_name)
            m_cut_dn = means[m_index] - math.sqrt(variances[m_index])
            m_cut_up = means[m_index] + math.sqrt(variances[m_index])
        else:
            m_cut_dn = cut_ranges_dn[sig_id]
            m_cut_up = cut_ranges_up[sig_id]
        feedbox = feed_box.Feedbox(
            sig_dict,
            bkg_dict,
            selected_features=job_wrapper.selected_features,
            apply_data=False,
            reshape_array=job_wrapper.norm_array,
            reset_mass=job_wrapper.reset_feature,
            reset_mass_name=job_wrapper.reset_feature_name,
            remove_negative_weight=job_wrapper.rm_negative_weight_events,
            cut_features=[m_cut_name, m_cut_name],
            cut_values=[m_cut_dn, m_cut_up],
            cut_types=[">", "<"],
            sig_weight=job_wrapper.sig_sumofweight,
            bkg_weight=job_wrapper.bkg_sumofweight,
            data_weight=job_wrapper.data_sumofweight,
            test_rate=job_wrapper.test_rate,
            rdm_seed=None,
            model_meta=job_wrapper.model_wrapper.model_meta,
            verbose=job_wrapper.verbose,
        )
        job_wrapper.model_wrapper.set_inputs(feedbox, apply_data=job_wrapper.apply_data)
        (plot_thresholds, significances, _, _,) = get_significances(
            job_wrapper.model_wrapper, significance_algo=job_wrapper.significance_algo,
        )
        plot_significances = []
        for dnn_cut in dnn_cut_list:
            threshold_id = (np.abs(np.array(plot_thresholds) - dnn_cut)).argmin()
            plot_significances.append(significances[threshold_id])
        w_inputs.append(plot_significances)
    x = []
    y = []
    w = []
    for index, w_input in enumerate(w_inputs):
        if len(x) == 0:
            x = dnn_cut_list.tolist()
            y = [job_wrapper.sig_list[index]] * len(w_input)
            w = w_input
        else:
            x += dnn_cut_list.tolist()
            y += [job_wrapper.sig_list[index]] * len(w_input)
            w += w_input
    # make plot
    plot_canvas = ROOT.TCanvas("2d_significance_c", "2d_significance_c", 1200, 900)
    hist_sig = th1_tools.TH2FTool(
        "2d_significance",
        "2d_significance",
        nbinx=10,
        xlow=0.8,
        xup=1.0,
        nbiny=len(job_wrapper.sig_list),
        ylow=0,
        yup=len(job_wrapper.sig_list),
    )
    hist_sig_text = th1_tools.TH2FTool(
        "2d_significance_text",
        "2d_significance_text",
        nbinx=10,
        xlow=0.8,
        xup=1.0,
        nbiny=len(job_wrapper.sig_list),
        ylow=0,
        yup=len(job_wrapper.sig_list),
    )
    hist_sig.fill_hist(fill_array_x=x, fill_array_y=y, weight_array=w)
    hist_sig.set_canvas(plot_canvas)
    hist_sig.set_palette("kBird")
    hist_sig.update_config("hist", "SetStats", 0)
    hist_sig.update_config("x_axis", "SetTitle", "dnn_cut")
    hist_sig.update_config("y_axis", "SetTitle", "mass point")
    hist_sig.update_config("y_axis", "SetLabelOffset", 0.003)
    hist_sig.draw("colz")
    hist_sig_text.fill_hist(
        fill_array_x=x, fill_array_y=y, weight_array=np.array(w).round(decimals=2)
    )
    hist_sig_text.set_canvas(plot_canvas)
    hist_sig_text.update_config("hist", "SetMarkerSize", 1.8)
    hist_sig_text.draw("text same")
    if save_plot:
        plot_canvas.SaveAs(save_dir + "/" + save_file_name + ".png")
