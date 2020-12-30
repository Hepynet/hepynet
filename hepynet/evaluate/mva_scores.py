import copy
import logging

import matplotlib.pyplot as plt
import numpy as np

from easy_atlas_plot.plot_utils import plot_utils_plt, plot_utils_root, th1_tools
from hepynet.evaluate import evaluate_utils

logger = logging.getLogger("hepynet")


def plot_mva_scores(model_wrapper, plot_config, save_dir, file_name="mva_scores"):
    # initialize
    logger.info("Plotting MVA scores")
    config = plot_config.clone()
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta

    # prepare signal
    sig_scores_dict = {}
    sig_weights_dict = {}
    for sig_key in config.sig_list:
        predict_arr, predict_weight_arr = feedbox.get_reshape("xs", array_key=sig_key)
        sig_scores_dict[sig_key] = model.predict(predict_arr).flatten()
        sig_weights_dict[sig_key] = predict_weight_arr.flatten()

    # prepare background
    bkg_scores_dict = {}
    bkg_weights_dict = {}
    for bkg_key in config.bkg_list:
        predict_arr, predict_weight_arr = feedbox.get_reshape("xb", array_key=bkg_key)
        bkg_scores_dict[bkg_key] = model.predict(predict_arr).flatten()
        bkg_weights_dict[bkg_key] = predict_weight_arr.flatten()

    # prepare data
    data_scores = np.array([])
    data_weights = np.array([])
    if config.apply_data:
        data_key = model_meta["data_key"]
        data_arr_temp, data_weights = feedbox.get_reshape("xd", array_key=data_key)
        data_scores = model.predict(data_arr_temp)
        data_weights = data_weights.flatten()

    # make plots
    if config.use_root:
        if plot_utils_root.HAS_ROOT:
            pass
        else:
            logger.error("Can't import ROOT, try to use matplotlib as backend.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_scores_plt(
            ax,
            config,
            sig_scores_dict,
            sig_weights_dict,
            bkg_scores_dict,
            bkg_weights_dict,
            data_scores,
            data_weights,
        )
        fig.savefig(f"{save_dir}/{file_name}.{config.save_format}")

    return 0  # success run


def plot_scores_plt(
    ax,
    plot_config,
    sig_scores_dict,
    sig_weights_dict,
    bkg_scores_dict,
    bkg_weights_dict,
    data_scores=None,
    data_weights=None,
):
    """Plots training score distribution for different background with matplotlib.
    """
    logger.debug("Plotting scores with matplotlib backend")
    config = plot_config.clone()
    if config.sig_list is None:
        config.sig_list = list(sig_scores_dict.keys())
    if config.bkg_list is None:
        config.bkg_list = list(bkg_scores_dict.keys())
    # plot background
    ax.hist(
        np.transpose(list(bkg_scores_dict.values())),
        bins=config.bins,
        range=config.range,
        weights=np.transpose(list(bkg_weights_dict.values())),
        histtype="bar",
        label=config.bkg_list,
        density=config.density,
        stacked=True,
    )
    # plot signal
    ax.hist(
        np.transpose(list(sig_scores_dict.values())),
        bins=config.bins,
        range=config.range,
        weights=np.transpose(list(sig_weights_dict.values())),
        histtype="step",
        label=config.sig_list,
        density=config.density,
    )
    # plot data
    if config.apply_data:
        evaluate_utils.paint_bars(
            ax,
            data_scores,
            "data",
            weights=data_weights,
            bins=config.bins,
            range=config.range,
            density=config.density,
            use_error=False,
        )
    ax.set_title(config.plot_title)
    ax.legend(loc="upper center")
    ax.set_xlabel("output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    if config.log:
        ax.set_yscale("log")
        ax.set_title(f"{config.plot_title}(log)")
    else:
        ax.set_title(f"{config.plot_title}(lin)")


def plot_scores_root(
    model_wrapper,
    sig_key,
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
    logger.info("Plotting scores with bkg separated with ROOT.")
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
    # sig_key = model_meta["sig_key"]
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
            plot_utils_root.get_highest_bin_value(hist_list),
            plot_utils_root.get_highest_bin_value(hist_sig),
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
        logger.debug(f"sig total weight: {total_weight_sig}")
        model_wrapper.total_weight_bkg = total_weight_bkg
        logger.debug(f"bkg total weight: {total_weight_bkg}")
        model_wrapper.total_weight_data = total_weight_data
        logger.debug(f"data total weight: {total_weight_data}")
        # save plot
        if save_plot:
            plot_canvas.SaveAs(
                save_dir + "/" + save_file_name + "_" + all_nodes[node_num] + "_lin.png"
            )
            plot_pad_score.SetLogy(2)
            plot_canvas.SaveAs(
                save_dir + "/" + save_file_name + "_" + all_nodes[node_num] + "_log.png"
            )


def plot_train_test_compare(
    model_wrapper, plot_config, save_dir=".", save_format="png"
):
    """Plots train/test scores distribution to check overtrain"""
    # initialize
    logger.info("Plotting train/test scores (original mass).")
    config = plot_config.clone()
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
        reset_mass=feedbox.reset_mass,
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
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot scores
        plot_scores_plt(
            ax,
            config,
            {"signal": xs_test_scores[:, node_num].flatten()},
            {"signal": xs_test_weight.flatten()},
            {"background": xb_test_scores[:].flatten()},
            {"background": xb_test_weight.flatten()},
        )
        evaluate_utils.paint_bars(
            ax,
            xb_train_scores[:, node_num],
            "b-train",
            weights=xb_train_weight,
            bins=config.bins,
            range=config.range,
            density=config.density,
            use_error=True,
            color="darkblue",
            fmt=".",
        )
        evaluate_utils.paint_bars(
            ax,
            xs_train_scores[:, node_num],
            "s-train",
            weights=xs_train_weight,
            bins=config.bins,
            range=config.range,
            density=config.density,
            use_error=True,
            color="maroon",
            fmt=".",
        )
        ax.legend(loc="upper center")
        # Make and show plots
        if feedbox.reset_mass:
            file_name = f"/mva_scores_overtrain_original_mass_{all_nodes[node_num]}"
        else:
            file_name = f"/mva_scores_overtrain_reset_mass_{all_nodes[node_num]}"
        fig.savefig(f"{save_dir}/{file_name}.{save_format}")
