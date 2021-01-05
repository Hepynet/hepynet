import logging
import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hepynet.common import array_utils, config_utils

logger = logging.getLogger("hepynet")

try:
    import ROOT
    root_available = True
    from easy_atlas_plot.plot_utils import plot_utils_root, th1_tools
except ImportError:
    root_available = False


def plot_correlation_matrix(model_wrapper, save_dir="."):
    save_dir = pathlib.Path(save_dir)
    save_dir = save_dir.joinpath("kinematics")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
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


def plot_input_distributions(
    model_wrapper,
    job_config,
    dnn_cut=None,
    multi_class_cut_branch=0,
    save_dir=None,
    show_reshaped=False,
    compare_cut_sb_separated=False,
    use_root=False,
):
    """Plots input distributions comparision plots for sig/bkg/data"""
    logger.info("Plotting input distributions.")
    # setup config
    ic = job_config.input.clone()
    ac = job_config.apply.clone()
    plot_cfg = ac.cfg_kine_study
    # prepare
    feedbox = model_wrapper.feedbox
    if show_reshaped:  # validation features not supported in get_reshape yet
        plot_feature_list = ic.selected_features
        bkg_array, bkg_fill_weights = feedbox.get_reshape("xb", array_key=ic.bkg_key)
        sig_array, sig_fill_weights = feedbox.get_reshape("xs", array_key=ic.sig_key)
    else:
        plot_feature_list = (
            ic.selected_features + ic.validation_features
        )
        bkg_array, bkg_fill_weights = feedbox.get_raw(
            "xb", array_key=ic.bkg_key, add_validation_features=True
        )
        sig_array, sig_fill_weights = feedbox.get_raw(
            "xs", array_key=ic.sig_key, add_validation_features=True
        )
    if plot_cfg.density:
        bkg_fill_weights = bkg_fill_weights / np.sum(bkg_fill_weights)
        sig_fill_weights = sig_fill_weights / np.sum(sig_fill_weights)

    # get fill weights with dnn cut
    if dnn_cut is not None:
        assert dnn_cut >= 0 and dnn_cut <= 1, "dnn_cut out or range."
        # prepare signal
        sig_selected_arr, _ = feedbox.get_reweight(
            "xs", array_key=ic.sig_key, reset_mass=False
        )
        sig_predictions = model_wrapper.get_model().predict(sig_selected_arr)
        if sig_predictions.ndim == 2:
            sig_predictions = sig_predictions[:, multi_class_cut_branch]
        sig_cut_index = array_utils.get_cut_index(sig_predictions, [dnn_cut], ["<"])
        sig_fill_weights_dnn = sig_fill_weights.copy()
        sig_fill_weights_dnn[sig_cut_index] = 0
        # prepare background
        bkg_selected_arr, _ = feedbox.get_reweight(
            "xb", array_key=ic.bkg_key, reset_mass=False
        )
        bkg_predictions = model_wrapper.get_model().predict(bkg_selected_arr)
        if bkg_predictions.ndim == 2:
            bkg_predictions = bkg_predictions[:, multi_class_cut_branch]
        bkg_cut_index = array_utils.get_cut_index(bkg_predictions, [dnn_cut], ["<"])
        bkg_fill_weights_dnn = bkg_fill_weights.copy()
        bkg_fill_weights_dnn[bkg_cut_index] = 0
        # normalize weights for density plots
        if plot_cfg.density:
            bkg_fill_weights_dnn = bkg_fill_weights_dnn / np.sum(bkg_fill_weights)
            sig_fill_weights_dnn = sig_fill_weights_dnn / np.sum(sig_fill_weights)
    else:
        bkg_fill_weights_dnn = None
        sig_fill_weights_dnn = None

    # remove 0 weight events
    bkg_non_0_ids = np.argwhere(bkg_fill_weights != 0).flatten()
    bkg_array = bkg_array[bkg_non_0_ids]
    bkg_fill_weights = bkg_fill_weights[bkg_non_0_ids].reshape((-1, 1))
    sig_non_0_ids = np.argwhere(sig_fill_weights != 0).flatten()
    sig_array = sig_array[sig_non_0_ids]
    sig_fill_weights = sig_fill_weights[sig_non_0_ids].reshape((-1, 1))
    if dnn_cut is not None:
        bkg_fill_weights_dnn = bkg_fill_weights_dnn[bkg_non_0_ids].reshape((-1, 1))
        sig_fill_weights_dnn = sig_fill_weights_dnn[sig_non_0_ids].reshape((-1, 1))

    # plot
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for feature_id, feature in enumerate(plot_feature_list):
        logger.debug(f"Plotting kinematics for {feature}")
        bkg_fill_array = np.reshape(bkg_array[:, feature_id], (-1, 1))
        sig_fill_array = np.reshape(sig_array[:, feature_id], (-1, 1))
        if use_root and plot_utils_root.HAS_ROOT:
            plot_range = None
            if feature in plot_cfg.__dict__.keys():
                feature_cfg = getattr(plot_cfg, feature)
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
                plot_density=plot_cfg.density,
                save_dir=save_dir,
                save_format="png",
                print_ratio_table=plot_cfg.save_ratio_table,
            )
        else:
            if dnn_cut is None:
                plot_input_distributions_plt(
                    feature,
                    sig_fill_array,
                    sig_fill_weights,
                    bkg_fill_array,
                    bkg_fill_weights,
                    plot_config=plot_cfg,
                    save_dir=save_dir,
                )
            else:
                plot_input_distributions_cut_dnn_plt(
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
        histtype=plot_cfg.histtype,
        facecolor=plot_cfg.facecolor,
        edgecolor=plot_cfg.edgecolor,
        label=plot_cfg.label,
        alpha=plot_cfg.alpha,
        hatch=plot_cfg.hatch,
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
        weights=numerator_weights,
    )
    denominator_ys, _ = np.histogram(
        denominator_values,
        bins=plot_cfg.bins,
        range=plot_cfg.range,
        weights=denominator_weights,
    )
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
    ax.set_ylim([0, 1])
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


def plot_input_distributions_plt(
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
        # plot sig
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_cfg.update(
            {"facecolor": plot_config.sig_color, "edgecolor": "none", "label": "signal"}
        )
        plot_hist_plt(
            ax, sig_fill_array, sig_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature, fontsize=16)
        fig.savefig(f"{save_dir}/{feature}_sig.{plot_config.save_format}")
        plt.close()
        # plot bkg
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_cfg.update(
            {
                "facecolor": plot_config.bkg_color,
                "edgecolor": "none",
                "label": "background",
            }
        )
        plot_hist_plt(
            ax, bkg_fill_array, bkg_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature, fontsize=16)
        fig.savefig(f"{save_dir}/{feature}_bkg.{plot_config.save_format}")
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot sig
        feature_cfg.update(
            {"facecolor": plot_config.sig_color, "edgecolor": "none", "label": "signal"}
        )
        plot_hist_plt(
            ax, sig_fill_array, sig_fill_weights, feature_cfg,
        )
        # plot bkg
        feature_cfg.update(
            {
                "facecolor": plot_config.bkg_color,
                "edgecolor": "none",
                "label": "background",
            }
        )
        plot_hist_plt(
            ax, bkg_fill_array, bkg_fill_weights, feature_cfg,
        )
        ax.legend(loc="upper right")
        fig.suptitle(feature, fontsize=16)
        fig.savefig(f"{save_dir}/{feature}.{plot_config.save_format}")
        plt.close()


def plot_input_distributions_cut_dnn_plt(
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
    fig = plt.figure(figsize=(8, 8))
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
    fig = plt.figure(figsize=(8, 8))
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

