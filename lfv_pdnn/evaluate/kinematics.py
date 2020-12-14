import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from easy_atlas_plot.plot_utils import plot_utils_root, th1_tools
from lfv_pdnn.common import array_utils, config_utils

logger = logging.getLogger("lfv_pdnn")


def plot_input_distributions(
    model_wrapper,
    sig_key,
    bkg_key,
    apply_data=False,
    figsize=(8, 6),
    style_cfg_path=None,
    show_reshaped=False,
    dnn_cut=None,
    multi_class_cut_branch=0,
    compare_cut_sb_separated=False,
    plot_density=True,
    plot_cfg={},
    save_dir=None,
    save_format="png",
    print_ratio_table=False,
):
    """Plots input distributions comparision plots for sig/bkg/data"""
    print("Plotting input distributions.")
    config = plot_cfg.clone()
    # config = {}
    # if style_cfg_path is not None:
    #    with open(style_cfg_path) as plot_config_file:
    #        config = json.load(plot_config_file)

    model_meta = model_wrapper.model_meta
    # sig_key = model_meta["sig_key"]
    # bkg_key = model_meta["bkg_key"]
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
    else:
        bkg_fill_weights_dnn = None
        sig_fill_weights_dnn = None
    # plot
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for feature_id, feature in enumerate(plot_feature_list):
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_fill_array = np.reshape(bkg_array[:, feature_id], (-1, 1))
        sig_fill_array = np.reshape(sig_array[:, feature_id], (-1, 1))
        if plot_utils_root.HAS_ROOT:
            plot_range = None
            if feature in config.__dict__.keys():
                feature_cfg = getattr(config, feature)
                plot_range = feature_cfg.range
            plot_input_distributions_root(
                feature,
                bkg_fill_array,
                bkg_fill_weights,
                sig_fill_array,
                sig_fill_weights,
                cut_dnn=dnn_cut,
                bkg_fill_weights_dnn=bkg_fill_weights_dnn,
                sig_fill_weights_dnn=sig_fill_weights_dnn,
                config={},
                compare_cut_sb_separated=compare_cut_sb_separated,
                plot_range=plot_range,
                plot_density=plot_density,
                save_dir=save_dir,
                save_format=save_format,
                print_ratio_table=print_ratio_table,
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            feature_cfg = config.clone()
            if feature in config.__dict__.keys():
                feature_cfg_tmp = getattr(config, feature)
                feature_cfg.update(feature_cfg_tmp.get_config_dict())
            # plot sig
            feature_cfg.update({"color": "tomato", "label": "signal"})
            plot_inputs_plt(
                ax, sig_fill_array, sig_fill_weights, feature_cfg,
            )
            # plot bkg
            feature_cfg.update({"color": "royalblue", "label": "background"})
            plot_inputs_plt(
                ax, bkg_fill_array, bkg_fill_weights, feature_cfg,
            )
            ax.legend(loc="upper right")
            fig.suptitle(feature, fontsize=16)
            fig.savefig(f"{save_dir}/{feature}.{config.save_format}")
            plt.close()


def plot_inputs_plt(
    ax, values, weights, plot_cfg,
):
    config = plot_cfg.clone()
    ax.hist(
        values,
        bins=config.bins,
        range=config.range,
        density=config.density,
        weights=weights,
        histtype=config.histtype,
        color=config.color,
        label=config.label,
        alpha=config.alpha,
        hatch=config.hatch,
    )


def plot_input_distributions_root(
    feature,
    bkg_fill_array,
    bkg_fill_weights,
    sig_fill_array,
    sig_fill_weights,
    cut_dnn=False,
    bkg_fill_weights_dnn=None,
    sig_fill_weights_dnn=None,
    config={},
    compare_cut_sb_separated=False,
    plot_range=None,
    plot_density=True,
    save_dir=".",
    save_format="png",
    print_ratio_table=False,
):
    # prepare background histogram
    hist_bkg = th1_tools.TH1FTool(feature + "_bkg", "bkg", nbin=100, xlow=-20, xup=20)
    hist_bkg.reinitial_hist_with_fill_array(bkg_fill_array)
    hist_bkg.fill_hist(bkg_fill_array, bkg_fill_weights)
    hist_bkg.set_config(config)
    hist_bkg.update_config("hist", "SetLineColor", 4)
    hist_bkg.update_config("hist", "SetFillStyle", 3354)
    hist_bkg.update_config("hist", "SetFillColor", ROOT.kBlue)
    hist_bkg.update_config("x_axis", "SetTitle", feature)
    hist_bkg.apply_config()
    # prepare signal histogram
    hist_sig = th1_tools.TH1FTool(feature + "_sig", "sig", nbin=100, xlow=-20, xup=20)
    hist_sig.reinitial_hist_with_fill_array(sig_fill_array)
    hist_sig.fill_hist(sig_fill_array, sig_fill_weights)
    hist_sig.set_config(config)
    hist_sig.update_config("hist", "SetLineColor", 2)
    hist_sig.update_config("hist", "SetFillStyle", 3354)
    hist_sig.update_config("hist", "SetFillColor", ROOT.kRed)
    hist_sig.update_config("x_axis", "SetTitle", feature)
    hist_sig.apply_config()
    # prepare bkg/sig histograms with dnn cut
    if cut_dnn:
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
        if cut_dnn:
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
            draw_options="hist",
            legend_title="legend",
            draw_norm=plot_density,
            x_range=plot_range,
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
        if plot_range is None:
            plot_range_bkg = [x_min, x_max]
        else:
            plot_range_bkg = plot_range
        ratio_plot.update_style_cfg("x_axis", "SetRange", plot_range_bkg)  ## >> hot fix
        ratio_plot.draw(draw_err=False, draw_base_line=False)
        hist_col_bkg.save(
            save_dir=save_dir, save_file_name=feature + "_bkg", save_format=save_format,
        )
        plot_canvas.SaveAs(save_dir + "/" + feature + "_bkg." + save_format)
        if print_ratio_table:
            ratio_plot.print_ratio(
                save_path=f"{save_dir}/{feature}_bkg_ratio_table.txt"
            )

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
        if plot_range is None:
            plot_range_sig = [x_min, x_max]
        else:
            plot_range_sig = plot_range
        ratio_plot.update_style_cfg("x_axis", "SetRange", plot_range_sig)  ## >> hot fix
        ratio_plot.draw(draw_err=False, draw_base_line=False)
        plot_canvas.SaveAs(save_dir + "/" + feature + "_sig." + save_format)
        if print_ratio_table:
            ratio_plot.print_ratio(
                save_path=f"{save_dir}/{feature}_sig_ratio_table.txt"
            )

