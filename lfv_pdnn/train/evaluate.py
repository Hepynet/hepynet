# -*- coding: utf-8 -*-
"""Functions for making plots"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import NullFormatter
from sklearn.metrics import auc, roc_curve

import ROOT
from HEPTools.plot_utils import plot_utils, th1_tools
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.train import train_utils


def calculate_auc(xs, xb, model, shuffle_col=None, rm_last_two=False):
    """Returns auc of given sig/bkg array."""
    x_plot, y_plot, y_pred = process_array(
        xs, xb, model, shuffle_col=shuffle_col, rm_last_two=rm_last_two)
    fpr_dm, tpr_dm, _ = roc_curve(y_plot,
                                  y_pred,
                                  sample_weight=x_plot[:, -1])
    # Calculate auc and return
    auc_value = auc(fpr_dm, tpr_dm)
    return auc_value


def plot_accuracy(ax: plt.axes, accuracy_list: list, val_accuracy_list: list) -> None:
    """Plots accuracy vs training epoch."""
    print("Plotting accuracy curve.")
    # Plot
    ax.plot(accuracy_list)
    ax.plot(val_accuracy_list)
    # Config
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    # ax.set_ylim((0, 1))
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='lower left')
    ax.grid()


def plot_auc_text(ax, titles, auc_values):
    """Plots auc information on roc curve."""
    auc_text = 'auc values:\n'
    for (title, auc_value) in zip(titles, auc_values):
        auc_text = auc_text + title + ": " + str(auc_value) + '\n'
    auc_text = auc_text[:-1]
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax.text(0.5,
            0.6,
            auc_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props)


def plot_correlation_matrix(ax, corr_matrix_dict, matrix_key="bkg"):
    # Get matrix
    corr_matrix = corr_matrix_dict[matrix_key]
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=.3,
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5},
                ax=ax)


def plot_feature_importance(ax, model_wrapper, log=True, max_feature=8):
    """Calculates importance of features and sort the feature.

    Definition of feature importance used here can be found in:
    https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data

    """
    print("Plotting feature importance.")
    # Prepare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    num_feature = len(feedbox["selected_features"])
    selected_feature_names = np.array(feedbox["selected_features"])
    feature_importance = np.zeros(num_feature)
    xs = feedbox["xs_test_original_mass"]
    xb = feedbox["xb_test_original_mass"]
    base_auc = calculate_auc(xs, xb, model, rm_last_two=True)
    print("base auc:", base_auc)
    # Calculate importance
    for num, feature_name in enumerate(selected_feature_names):
        current_auc = calculate_auc(
            xs, xb, model, shuffle_col=num, rm_last_two=True)
        feature_importance[num] = (1 - current_auc) / (1 - base_auc)
        print(feature_name, ":", feature_importance[num])
    # Sort
    sort_list = np.flip(np.argsort(feature_importance))
    sorted_importance = feature_importance[sort_list]
    sorted_names = selected_feature_names[sort_list]
    print("Feature importance rank:", sorted_names)
    # Plot
    if num_feature > max_feature:
        num_show = max_feature
    else:
        num_show = num_feature
    ax.bar(np.arange(num_show),
           sorted_importance[:num_show],
           align='center',
           alpha=0.5,
           log=log)
    ax.axhline(1, ls='--', color='r')
    ax.set_title("feature importance")
    ax.set_xticks(np.arange(num_show))
    ax.set_xticklabels(sorted_names[:num_show])


def plot_input_distributions(
        model_wrapper,
        apply_data=False,
        figsize=(8, 6),
        style_cfg_path=None,
        save_fig=False,
        save_dir=None,
        save_format="png"):
    """Plots input distributions comparision plots for sig/bkg/data"""
    print("Plotting input distributions.")
    config = {}
    if style_cfg_path is not None:
        with open(style_cfg_path) as plot_config_file:
            config = json.load(plot_config_file)

    for feature_id, feature in enumerate(model_wrapper.selected_features):
        # prepare background histogram
        hist_bkg = th1_tools.TH1FTool(
            feature + "_bkg",
            "bkg",
            nbin=100,
            xlow=-20,
            xup=20)
        fill_array = np.reshape(
            model_wrapper.feedbox["xb_raw"][:, feature_id], (-1, 1))
        fill_weights = np.reshape(
            model_wrapper.feedbox["xb_raw"][:, -1], (-1, 1))
        hist_bkg.reinitial_hist_with_fill_array(fill_array)
        hist_bkg.fill_hist(fill_array, fill_weights)
        hist_bkg.set_config(config)
        hist_bkg.update_config("hist", "SetLineColor", 4)
        hist_bkg.update_config("hist", "SetFillStyle", 3001)
        hist_bkg.update_config("hist", "SetFillColor", ROOT.kBlue)
        hist_bkg.update_config('x_axis', 'SetTitle', feature)
        hist_bkg.apply_config()
        # hist_bkg.draw()
        # hist_bkg.save(save_dir=save_dir,
        #              save_file_name=feature + "_bkg",
        #              save_format=save_format)
        # prepare signal histogram
        hist_sig = th1_tools.TH1FTool(
            feature + "_sig",
            "sig",
            nbin=100,
            xlow=-20,
            xup=20)
        fill_array = np.reshape(
            model_wrapper.feedbox["xs_raw"][:, feature_id], (-1, 1))
        fill_weights = np.reshape(
            model_wrapper.feedbox["xs_raw"][:, -1], (-1, 1))
        hist_sig.reinitial_hist_with_fill_array(fill_array)
        hist_sig.fill_hist(fill_array, fill_weights)
        hist_sig.set_config(config)
        hist_sig.update_config("hist", "SetLineColor", 2)
        hist_sig.update_config("hist", "SetFillStyle", 3001)
        hist_sig.update_config("hist", "SetFillColor", ROOT.kRed)
        hist_sig.update_config('x_axis', 'SetTitle', feature)
        hist_sig.apply_config()
        hist_col = th1_tools.HistCollection(
            [hist_bkg, hist_sig],
            name=feature,
            title="input var: " + feature)
        hist_col.draw(
            draw_options="hist",
            legend_title="legend",
            draw_norm=True,
            remove_empty_ends=True)
        hist_col.save(
            save_dir=save_dir,
            save_file_name=feature,
            save_format=save_format)


def plot_overtrain_check(ax, model_wrapper, bins=50, range=(-0.25, 1.25), log=True):
    """Plots train/test scores distribution to check overtrain"""
    print("Plotting train/test scores.")
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    # plot test scores
    plot_scores(
        ax,
        model,
        feedbox["xb_test_selected"],
        feedbox["xb_test"][:, -1],
        feedbox["xs_test_selected"],
        feedbox["xs_test"][:, -1],
        apply_data=False,
        title="over training check",
        bkg_label="b-test",
        sig_label="s-test",
        bins=bins,
        range=range,
        density=True,
        log=log)
    # plot train scores
    make_bar_plot(
        ax,
        model.predict(feedbox["xb_train_selected"]),
        "b-train",
        weights=np.reshape(feedbox["xb_train"][:, -1], (-1, 1)),
        bins=bins,
        range=range,
        density=True,
        use_error=True,
        color="darkblue",
        fmt=".")
    make_bar_plot(
        ax,
        model.predict(feedbox["xs_train_selected"]),
        "s-train",
        weights=np.reshape(feedbox["xs_train"][:, -1], (-1, 1)),
        bins=bins,
        range=range,
        density=True,
        use_error=True,
        color="maroon",
        fmt=".")


def plot_overtrain_check_original_mass(
        ax,
        model_wrapper,
        bins=50,
        range=(-0.25, 1.25),
        log=True):
    """Plots train/test scores distribution to check overtrain"""
    print("Plotting train/test scores (original mass).")
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    # plot test scores
    plot_scores(
        ax,
        model,
        feedbox["xb_test_selected_original_mass"],
        feedbox["xb_test_original_mass"][:, -1],
        feedbox["xs_test_selected_original_mass"],
        feedbox["xs_test_original_mass"][:, -1],
        apply_data=False,
        title="over training check",
        bkg_label="b-test",
        sig_label="s-test",
        bins=bins,
        range=range,
        density=True,
        log=log)
    # plot train scores
    make_bar_plot(
        ax,
        model.predict(feedbox["xb_train_selected_original_mass"]),
        "b-train",
        weights=np.reshape(feedbox["xb_train_original_mass"][:, -1], (-1, 1)),
        bins=bins,
        range=range,
        density=True,
        use_error=True,
        color="darkblue",
        fmt=".")
    make_bar_plot(
        ax,
        model.predict(feedbox["xs_train_selected_original_mass"]),
        "s-train",
        weights=np.reshape(feedbox["xs_train_original_mass"][:, -1], (-1, 1)),
        bins=bins,
        range=range,
        density=True,
        use_error=True,
        color="maroon",
        fmt=".")


def plot_loss(ax: plt.axes, loss_list: list, val_loss_list: list) -> None:
    """Plots loss vs training epoch."""
    print("Plotting loss curve.")
    # Plot
    ax.plot(loss_list)
    ax.plot(val_loss_list)
    # Config
    ax.set_title('model loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['train', 'val'], loc='lower left')
    ax.grid()


def plot_roc(ax, xs, xb, model, yscal="logit", ylim=(0.1, 1-1e-4)):
    """Plots roc curve on given axes."""
    # Get data
    x_plot, y_plot, y_pred = process_array(xs, xb, model, rm_last_two=True)
    fpr_dm, tpr_dm, _ = roc_curve(y_plot, y_pred, sample_weight=x_plot[:, -1])
    # Make plots
    ax.plot(fpr_dm, tpr_dm)
    ax.set_title("roc curve")
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_yscale(yscal)
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Calculate auc and return parameters
    auc_value = auc(fpr_dm, tpr_dm)
    return auc_value, fpr_dm, tpr_dm


def plot_scores(
        ax,
        model,
        selected_bkg,
        bkg_weight,
        selected_sig,
        sig_weight,
        selected_data=None,
        data_weight=None,
        apply_data=False,
        title="scores",
        bkg_label="bkg",
        sig_label="sig",
        bins=50,
        range=(-0.25, 1.25),
        density=True,
        log=False):
    """Plots score distribution for siganl and background."""
    ax.hist(model.predict(selected_bkg),
            weights=bkg_weight,
            bins=bins,
            range=range,
            histtype='step',
            label=bkg_label,
            density=density,
            log=log,
            facecolor='blue',
            edgecolor='darkblue',
            alpha=0.5,
            fill=True)
    ax.hist(model.predict(selected_sig),
            weights=sig_weight,
            bins=bins,
            range=range,
            histtype='step',
            label=sig_label,
            density=density,
            log=log,
            facecolor='red',
            edgecolor='maroon',
            hatch='///',
            alpha=1,
            fill=False)
    if apply_data:
        make_bar_plot(ax,
                      model.predict(selected_data),
                      "data",
                      weights=np.reshape(data_weight, (-1, 1)),
                      bins=bins,
                      range=range,
                      density=density,
                      use_error=False)
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()


def plot_scores_separate(ax,
                         model_wrapper,
                         bkg_dict,
                         bkg_plot_key_list=None,
                         sig_arr=None,
                         sig_weights=None,
                         apply_data=False,
                         data_arr=None,
                         data_weight=None,
                         plot_title='all input scores',
                         bins=50,
                         range=(-0.25, 1.25),
                         density=True,
                         log=False):
    """Plots training score distribution for different background with matplotlib.

    Note:
        bkg_plot_key_list can be used to adjust order of background sample 
        stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
        'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top

    """
    print("Plotting scores with bkg separated.")
    predict_arr_list = []
    predict_arr_weight_list = []
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    # plot background
    if (type(bkg_plot_key_list) is
            not list) or len(bkg_plot_key_list) == 0:
        # prepare plot key list sort with total weight by default
        original_keys = list(bkg_dict.keys())
        total_weight_list = []
        for key in original_keys:
            total_weight = np.sum((bkg_dict[key])[:, -1])
            total_weight_list.append(total_weight)
        sort_indexes = np.argsort(np.array(total_weight_list))
        bkg_plot_key_list = [
            original_keys[index] for index in sort_indexes
        ]
    for arr_key in bkg_plot_key_list:
        bkg_arr_temp = bkg_dict[arr_key].copy()
        bkg_arr_temp[:, 0:-2] = train_utils.norarray(
            bkg_arr_temp[:, 0:-2],
            average=feedbox["norm_average"],
            variance=feedbox["norm_variance"])
        selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
        predict_arr_list.append(
            np.array(model.predict(selected_arr)))
        predict_arr_weight_list.append(bkg_arr_temp[:, -1])
    try:
        ax.hist(np.transpose(predict_arr_list),
                bins=bins,
                range=range,
                weights=np.transpose(predict_arr_weight_list),
                histtype='bar',
                label=bkg_plot_key_list,
                density=density,
                stacked=True)
    except:
        ax.hist(predict_arr_list[0],
                bins=bins,
                range=range,
                weights=predict_arr_weight_list[0],
                histtype='bar',
                label=bkg_plot_key_list,
                density=density,
                stacked=True)
    # plot signal
    if sig_arr is None:
        selected_arr = train_utils.get_valid_feature(
            feedbox["xs_reshape"])
        predict_arr = model.predict(selected_arr)
        predict_weight_arr = feedbox["xs_reshape"][:, -1]
    else:
        sig_arr_temp = sig_arr.copy()
        sig_arr_temp[:, 0:-2] = train_utils.norarray(
            sig_arr[:, 0:-2],
            average=feedbox["norm_average"],
            variance=feedbox["norm_variance"])
        selected_arr = train_utils.get_valid_feature(sig_arr_temp)
        predict_arr = np.array(model.predict(selected_arr))
        predict_weight_arr = sig_arr_temp[:, -1]
    ax.hist(predict_arr,
            bins=bins,
            range=range,
            weights=predict_weight_arr,
            histtype='step',
            label='sig',
            density=density)
    # plot data
    if apply_data:
        if data_arr is None:
            data_arr = feedbox["xd_selected_original_mass"].copy()
            data_weight = feedbox["xd"][:, -1]
        make_bar_plot(ax,
                      model.predict(data_arr),
                      "data",
                      weights=np.reshape(data_weight, (-1, 1)),
                      bins=bins,
                      range=range,
                      density=density,
                      use_error=False)
    ax.set_title(plot_title)
    ax.legend(loc='upper right')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    if log is True:
        ax.set_yscale('log')
        ax.set_title(plot_title + "(log)")
    else:
        ax.set_title(plot_title + "(lin)")


def plot_scores_separate_root(model_wrapper,
                              bkg_dict,
                              bkg_plot_key_list,
                              sig_arr=None,
                              apply_data=False,
                              apply_data_range=None,
                              data_arr=None,
                              plot_title='all input scores',
                              bins=50,
                              range=(-0.25, 1.25),
                              scale_sig=False,
                              density=True,
                              log_scale=False,
                              save_plot=False,
                              save_dir=None,
                              save_file_name=None):
    """Plots training score distribution for different background with ROOT

    Note:
        bkg_plot_key_list can be used to adjust order of background sample 
        stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
        'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top

    """
    print("Plotting scores with bkg separated with ROOT.")
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    plot_canvas = ROOT.TCanvas(plot_title, plot_title, 800, 800)
    plot_pad_score = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    plot_pad_score.SetBottomMargin(0)
    plot_pad_score.SetGridx()
    plot_pad_score.Draw()
    plot_pad_score.cd()
    hist_list = []
    # plot background
    if (type(bkg_plot_key_list) is
            not list) or len(bkg_plot_key_list) == 0:
        # prepare plot key list sort with total weight by default
        original_keys = list(bkg_dict.keys())
        total_weight_list = []
        for key in original_keys:
            total_weight = np.sum((bkg_dict[key])[:, -1])
            total_weight_list.append(total_weight)
        sort_indexes = np.argsort(np.array(total_weight_list))
        bkg_plot_key_list = [
            original_keys[index] for index in sort_indexes
        ]
    for arr_key in bkg_plot_key_list:
        bkg_arr_temp = bkg_dict[arr_key].copy()
        bkg_arr_temp[:, 0:-2] = train_utils.norarray(
            bkg_arr_temp[:, 0:-2],
            average=feedbox["norm_average"],
            variance=feedbox["norm_variance"])
        selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
        predict_arr = np.array(model.predict(selected_arr))
        predict_weight_arr = bkg_arr_temp[:, -1]

        th1_temp = th1_tools.TH1FTool(arr_key,
                                      arr_key,
                                      nbin=bins,
                                      xlow=range[0],
                                      xup=range[1])
        th1_temp.fill_hist(predict_arr, predict_weight_arr)
        hist_list.append(th1_temp)
    hist_stacked_bkgs = th1_tools.THStackTool("bkg stack plot",
                                              plot_title,
                                              hist_list,
                                              canvas=plot_pad_score)
    hist_stacked_bkgs.draw("pfc hist", log_scale=log_scale)
    hist_stacked_bkgs.get_hstack().GetYaxis().SetTitle("events/bin")
    hist_bkg_total = hist_stacked_bkgs.get_added_hist()
    total_weight_bkg = hist_bkg_total.get_hist().GetSumOfWeights()
    # plot signal
    if sig_arr is None:
        selected_arr = train_utils.get_valid_feature(
            feedbox["xs_reshape"])
        predict_arr = model.predict(selected_arr)
        predict_weight_arr = feedbox["xs_reshape"][:, -1]
    else:
        sig_arr_temp = sig_arr.copy()
        sig_arr_temp[:, 0:-2] = train_utils.norarray(
            sig_arr[:, 0:-2],
            average=feedbox["norm_average"],
            variance=feedbox["norm_variance"])
        selected_arr = train_utils.get_valid_feature(sig_arr_temp)
        predict_arr = np.array(model.predict(selected_arr))
        predict_weight_arr = sig_arr_temp[:, -1]
    if scale_sig:
        sig_title = "sig-scaled"
    else:
        sig_title = "sig"
    hist_sig = th1_tools.TH1FTool("sig added",
                                  sig_title,
                                  nbin=bins,
                                  xlow=range[0],
                                  xup=range[1],
                                  canvas=plot_pad_score)
    hist_sig.fill_hist(predict_arr, predict_weight_arr)
    total_weight_sig = hist_sig.get_hist().GetSumOfWeights()
    if scale_sig:
        total_weight = hist_stacked_bkgs.get_total_weights()
        scale_factor = total_weight / hist_sig.get_hist().GetSumOfWeights()
        hist_sig.get_hist().Scale(scale_factor)
    hist_sig.update_config("hist", "SetLineColor", ROOT.kRed)
    # set proper y range
    maximum_y = max(plot_utils.get_highest_bin_value(hist_list),
                    plot_utils.get_highest_bin_value(hist_sig))
    hist_stacked_bkgs.get_hstack().SetMaximum(1.2 * maximum_y)
    hist_stacked_bkgs.get_hstack().SetMinimum(0.1)
    hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelFont(43)
    hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelSize(15)
    hist_sig.draw("same hist")
    # plot data if required
    total_weight_data = 0
    if apply_data:
        if data_arr is None:
            selected_arr = feedbox["xd_selected_original_mass"].copy(
            )
            predict_arr = model.predict(selected_arr)
            predict_weight_arr = feedbox["xd"][:, -1]
        hist_data = th1_tools.TH1FTool("data added",
                                       "data",
                                       nbin=bins,
                                       xlow=range[0],
                                       xup=range[1],
                                       canvas=plot_pad_score)
        hist_data.fill_hist(predict_arr, predict_weight_arr)
        hist_data.update_config("hist", "SetMarkerStyle", ROOT.kFullCircle)
        hist_data.update_config("hist", "SetMarkerColor", ROOT.kBlack)
        hist_data.update_config("hist", "SetMarkerSize", 0.8)
        if apply_data_range is not None:
            hist_data.get_hist().GetXaxis().SetRangeUser(
                apply_data_range[0], apply_data_range[1])
        hist_data.draw("same e1", log_scale=log_scale)
        total_weight_data = hist_data.get_hist().GetSumOfWeights()
    else:
        hist_data = hist_sig
        total_weight_data = 0
    hist_data.build_legend(0.4, 0.7, 0.6, 0.9)

    # ratio plot
    if apply_data:
        plot_canvas.cd()
        plot_pad_ratio = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
        plot_pad_ratio.SetTopMargin(0)
        plot_pad_ratio.SetGridx()
        plot_pad_ratio.Draw()
        ratio_plot = th1_tools.RatioPlot(
            hist_data,
            hist_bkg_total,
            x_title="DNN Score",
            y_title="data/bkg",
            canvas=plot_pad_ratio)
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
        plot_canvas.SaveAs(save_dir + "/" + save_file_name + ".png")


def plot_significance_scan(ax, model_wrapper) -> None:
    """Shows significance change with threshould.

    Note:
        significance is calculated by s/sqrt(b)
    """
    print("Plotting significance scan.")
    feedbox = model_wrapper.feedbox
    sig_predictions = model_wrapper.get_model().predict(
        feedbox["xs_reshape_selected"])
    sig_predictions_weights = np.reshape(
        feedbox["xs_reshape"][:, -1], (-1, 1))
    bkg_predictions = model_wrapper.get_model().predict(
        feedbox["xb_reshape_selected"])
    bkg_predictions_weights = np.reshape(
        feedbox["xb_reshape"][:, -1], (-1, 1))
    # prepare threshoulds
    bin_array = np.array(range(-1000, 1000))
    threshoulds = 1. / (1. + 1. / np.exp(bin_array * 0.02))
    # scan
    significances = []
    plot_threshoulds = []
    sig_above_threshould = []
    bkg_above_threshould = []
    for dnn_cut in threshoulds:
        sig_ids_passed = sig_predictions > dnn_cut
        total_sig_weights_passed = np.sum(
            sig_predictions_weights[sig_ids_passed])
        bkg_ids_passed = bkg_predictions > dnn_cut
        total_bkg_weights_passed = np.sum(
            bkg_predictions_weights[bkg_ids_passed])
        if total_bkg_weights_passed > 0 and total_sig_weights_passed > 0:
            plot_threshoulds.append(dnn_cut)
            current_significance = train_utils.calculate_asimove(
                total_sig_weights_passed, total_bkg_weights_passed)
            # current_significance = total_sig_weights_passed / total_bkg_weights_passed
            significances.append(current_significance)
            sig_above_threshould.append(total_sig_weights_passed)
            bkg_above_threshould.append(total_bkg_weights_passed)
    total_sig_weight = np.sum(sig_predictions_weights)
    total_bkg_weight = np.sum(bkg_predictions_weights)
    significances_no_nan = np.nan_to_num(significances)
    max_significance = np.amax(significances_no_nan)
    index = np.argmax(significances_no_nan)
    max_significance_threshould = plot_threshoulds[index]
    max_significance_sig_total = sig_above_threshould[index]
    max_significance_bkg_total = bkg_above_threshould[index]
    # make plots
    # plot original significance
    original_significance = train_utils.calculate_asimove(
        total_sig_weight, total_bkg_weight)
    ax.axhline(y=original_significance, color="grey", linestyle="--")
    # significance scan curve
    ax.plot(plot_threshoulds, significances_no_nan,
            color='r', label="asimov")
    # signal/background events scan curve
    ax2 = ax.twinx()
    max_sig_events = sig_above_threshould[0]
    max_bkg_events = bkg_above_threshould[0]
    sig_events_above_threshould = np.array(
        sig_above_threshould) / max_sig_events
    bkg_events_above_threshould = np.array(
        bkg_above_threshould) / max_bkg_events
    ax2.plot(plot_threshoulds,
             sig_events_above_threshould,
             color="orange",
             label="sig")
    ax2.plot(plot_threshoulds,
             bkg_events_above_threshould,
             color="blue",
             label="bkg")
    ax2.set_ylabel('sig(bkg) ratio after cut')
    # reference threshould
    ax.axvline(x=max_significance_threshould,
               color='green',
               linestyle="-.")
    # more infomation
    content = "max asimov:" + str(
        common_utils.get_significant_digits(max_significance, 6)
    ) + "\nbest threshould:" + str(
        common_utils.get_significant_digits(
            max_significance_threshould, 6)) + "\nbase asimov:" + str(
                common_utils.get_significant_digits(
                    original_significance,
                    6)) + "\nsig events above threshould:" + str(
                        common_utils.get_significant_digits(
                            max_significance_sig_total,
                            6)) + "\nbkg events above threshould:" + str(
                                common_utils.get_significant_digits(
                                    max_significance_bkg_total, 6))
    ax.text(0.05,
            0.9,
            content,
            verticalalignment='top',
            horizontalalignment='left',
            transform=ax.transAxes,
            color='green',
            fontsize=12)
    # set up plot
    ax.set_title("significance scan")
    ax.set_xscale("logit")
    ax.set_xlabel('DNN score threshould')
    ax.set_ylabel('asimov')
    ax.set_ylim(bottom=0)
    ax.locator_params(nbins=10, axis='x')
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='center left')
    ax2.legend(loc='center right')
    # ax2.set_yscale("log")
    # collect meta data
    model_wrapper.original_significance = original_significance
    model_wrapper.max_significance = max_significance
    model_wrapper.max_significance_threshould = max_significance_threshould


def plot_train_test_roc(ax, model_wrapper, yscal="logit", ylim=(0.1, 1-1e-4)):
    """Plots roc curve."""
    print("Plotting train/test roc curve.")
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    # First plot roc for train dataset
    auc_train, _, _ = plot_roc(
        ax, feedbox["xs_train"], feedbox["xb_train"], model)
    # Then plot roc for test dataset
    auc_test, _, _ = plot_roc(
        ax, feedbox["xs_test"], feedbox["xb_test"], model)
    # Then plot roc for train dataset without reseting mass
    auc_train_original, _, _ = plot_roc(
        ax,
        feedbox["xs_train_original_mass"],
        feedbox["xb_train_original_mass"],
        model, 
        yscal=yscal, 
        ylim=ylim)
    # Lastly, plot roc for test dataset without reseting mass
    auc_test_original, _, _ = plot_roc(
        ax,
        feedbox["xs_test_original_mass"],
        feedbox["xb_test_original_mass"],
        model, 
        yscal=yscal, 
        ylim=ylim)
    # Show auc value:
    plot_auc_text(
        ax, ['TV ', 'TE ', 'TVO', 'TEO'],
        [auc_train, auc_test, auc_train_original, auc_test_original])
    # Extra plot config
    ax.legend([
        'TV (train+val)', 'TE (test)', 'TVO (train+val original)',
        'TEO (test original)'
    ],
        loc='lower right')
    ax.grid()
    # Collect meta data
    auc_dict = {}
    auc_dict["auc_train"] = auc_train
    auc_dict["auc_test"] = auc_test
    auc_dict["auc_train_original"] = auc_train_original
    auc_dict["auc_test_original"] = auc_test_original
    return auc_dict


def process_array(xs, xb, model, shuffle_col=None, rm_last_two=False):
    """Process sig/bkg arrays in the same way for training arrays."""
    # Get data
    xs_proc = xs.copy()
    xb_proc = xb.copy()
    x_proc = np.concatenate((xs_proc, xb_proc))
    if shuffle_col is not None:
        x_proc = array_utils.reset_col(x_proc, x_proc, shuffle_col)
    if rm_last_two:
        x_proc_selected = train_utils.get_valid_feature(x_proc)
    else:
        x_proc_selected = x_proc
    y_proc = np.concatenate(
        (np.ones(xs_proc.shape[0]), np.zeros(xb_proc.shape[0])))
    y_pred = model.predict(x_proc_selected)
    return x_proc, y_proc, y_pred


def make_bar_plot(ax,
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
                  fmt: str = ".k") -> None:
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
        assert isinstance(
            data, np.ndarray), "datas element should be numpy array."
        assert isinstance(
            weight, np.ndarray), "weights element should be numpy array."
        assert data.shape == weight.shape, "Input weights should be None or have same type as arrays."
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
    plot_ys, _ = np.histogram(data_1dim,
                              bins=bins,
                              range=range,
                              weights=weight_1dim,
                              density=density)
    sum_weight_squares, bin_edges = np.histogram(data_1dim,
                                                 bins=bins,
                                                 range=range,
                                                 weights=np.power(
                                                     weight_1dim, 2))
    if density:
        error_scale = 1 / (np.sum(weight_1dim) *
                           (range[1] - range[0]) / bins)
        errors = np.sqrt(sum_weight_squares) * error_scale
    else:
        errors = np.sqrt(sum_weight_squares)
    # Only plot ratio when bin is not 0.
    bin_centers = np.array([])
    bin_ys = np.array([])
    bin_yerrs = np.array([])
    for i, y1 in enumerate(plot_ys):
        if y1 != 0:
            ele_center = np.array(
                [0.5 * (bin_edges[i] + bin_edges[i + 1])])
            bin_centers = np.concatenate((bin_centers, ele_center))
            ele_y = np.array([y1])
            bin_ys = np.concatenate((bin_ys, ele_y))
            ele_yerr = np.array([errors[i]])
            bin_yerrs = np.concatenate((bin_yerrs, ele_yerr))
    # plot bar
    bin_size = bin_edges[1] - bin_edges[0]
    if use_error:
        ax.errorbar(bin_centers,
                    bin_ys,
                    xerr=bin_size / 2.,
                    yerr=bin_yerrs,
                    fmt=fmt,
                    label=labels,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color)
    else:
        ax.errorbar(bin_centers,
                    bin_ys,
                    xerr=bin_size / 2.,
                    yerr=None,
                    fmt=fmt,
                    label=labels,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color)
    # Config
    if title is not None:
        ax.set_title(title)
    if x_lable is not None:
        if x_unit is not None:
            ax.set_xlabel(x_lable + '/' + x_unit)
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
