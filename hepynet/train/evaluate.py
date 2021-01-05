# -*- coding: utf-8 -*-
"""Functions for making plots"""

import copy
import json
import logging
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import auc, roc_auc_score, roc_curve

from hepynet.common import array_utils, common_utils
from hepynet.data_io import feed_box, numpy_io
from hepynet.train import train_utils

try:
    import ROOT
    root_available = True
    from easy_atlas_plot.plot_utils import plot_utils_root, th1_tools
except ImportError:
    root_available = False

logger = logging.getLogger("hepynet")


def plot_2d_density(
    job_wrapper, save_plot=False, save_dir=None, save_file_name="2d_density",
):
    """Plots 2D hist to see event distribution of signal and backgrounds events.

    x-axis will be dnn scores, y-axis will be mass parameter, z-axis is total
    weight shown by color

    """
    ic = job_wrapper.job_config.input
    model_wrapper = job_wrapper.model_wrapper
    feedbox = model_wrapper.feedbox
    model_meta = model_wrapper.model_meta
    # plot signal
    sig_key = model_meta["sig_key"]
    sig_arr_raw, _ = feedbox.get_raw("xs", array_key=sig_key)
    sig_arr, sig_weights = feedbox.get_reshape("xs", array_key=sig_key)
    predict_arr = model_wrapper.get_model().predict(sig_arr)
    if predict_arr.ndim == 2:
        predict_arr = predict_arr[:, 0]
    mass_index = ic.selected_features.index(ic.reset_feature_name)
    x = predict_arr
    y = sig_arr_raw[:, mass_index]
    ## make plot
    plot_canvas = ROOT.TCanvas("2d_density_sig", "2d_density_sig", 800, 600)
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
    hist_sig.fill_hist(fill_array_x=x, fill_array_y=y, weight_array=sig_weights)
    hist_sig.set_canvas(plot_canvas)
    hist_sig.set_palette("kBird")
    hist_sig.update_config("hist", "SetStats", 0)
    hist_sig.update_config("x_axis", "SetTitle", "dnn score")
    hist_sig.update_config("y_axis", "SetTitle", "mass")
    hist_sig.draw("colz")
    hist_sig.save(save_dir=save_dir, save_file_name=save_file_name + "_sig")
    # plot background
    bkg_key = model_meta["bkg_key"]
    bkg_arr_raw, _ = feedbox.get_raw("xb", array_key=bkg_key)
    bkg_arr, bkg_weights = feedbox.get_reshape("xb", array_key=bkg_key)
    predict_arr = model_wrapper.get_model().predict(bkg_arr)
    if predict_arr.ndim == 2:
        predict_arr = predict_arr[:, 0]
    mass_index = ic.selected_features.index(ic.reset_feature_name)
    x = predict_arr
    y = bkg_arr_raw[:, mass_index]
    ## make plot
    plot_canvas = ROOT.TCanvas("2d_density_bkg", "2d_density_bkg", 800, 600)
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
    hist_bkg.fill_hist(fill_array_x=x, fill_array_y=y, weight_array=bkg_weights)
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
    dnn_cut_min=None,
    dnn_cut_max=None,
    dnn_cut_step=None,
):
    """Makes 2d map of significance"""
    if not (dnn_cut_min and dnn_cut_max and dnn_cut_step):
        logging.warn(
            "No complete dnn_cut specified, using default (min=0.5, max=1, step=0.05)"
        )
        dnn_cut_min = 0.5
        dnn_cut_max = 1
        dnn_cut_step = 0.05
    dnn_cut_list = np.arange(dnn_cut_min, dnn_cut_max, dnn_cut_step)
    w_inputs = []
    logger.info("Making 2d significance scan.")
    ic = job_wrapper.job_config.input
    tc = job_wrapper.job_config.train
    ac = job_wrapper.job_config.apply
    rc = job_wrapper.job_config.run
    bkg_dict = numpy_io.load_npy_arrays(
        rc.npy_path,
        ic.campaign,
        ic.region,
        ic.channel,
        ic.bkg_list,
        ic.selected_features,
        validation_features=ic.validation_features,
        cut_features=ic.cut_features,
        cut_values=ic.cut_values,
        cut_types=ic.cut_types,
    )
    sig_dict = numpy_io.load_npy_arrays(
        rc.npy_path,
        ic.campaign,
        ic.region,
        ic.channel,
        ic.sig_list,
        ic.selected_features,
        validation_features=ic.validation_features,
        cut_features=ic.cut_features,
        cut_values=ic.cut_values,
        cut_types=ic.cut_types,
    )
    for sig_id, scan_sig_key in enumerate(ic.sig_list):
        xs = sig_dict[scan_sig_key]
        m_cut_name = ic.reset_feature_name
        if cut_ranges_dn is None or len(cut_ranges_dn) == 0:
            means, variances = train_utils.get_mean_var(
                xs[m_cut_name], axis=0, weights=xs["weight"]
            )
            m_cut_dn = means[0] - math.sqrt(variances[0])
            m_cut_up = means[0] + math.sqrt(variances[0])
        else:
            m_cut_dn = cut_ranges_dn[sig_id]
            m_cut_up = cut_ranges_up[sig_id]
        feedbox = feed_box.Feedbox(
            sig_dict,
            bkg_dict,
            selected_features=ic.selected_features,
            apply_data=False,
            reshape_array=ic.norm_array,
            reset_mass=ic.reset_feature,
            reset_mass_name=ic.reset_feature_name,
            remove_negative_weight=ic.rm_negative_weight_events,
            cut_features=[m_cut_name, m_cut_name],
            cut_values=[m_cut_dn, m_cut_up],
            cut_types=[">", "<"],
            sig_weight=ic.sig_sumofweight,
            bkg_weight=ic.bkg_sumofweight,
            data_weight=ic.data_sumofweight,
            test_rate=ic.test_rate,
            rdm_seed=None,
            model_meta=job_wrapper.model_wrapper.model_meta,
            verbose=tc.verbose,
        )
        job_wrapper.model_wrapper.set_inputs(feedbox, apply_data=ac.apply_data)
        significance_algo = ac.cfg_2d_significance_scan["significance_algo"]
        (plot_thresholds, significances, _, _,) = get_significances(
            job_wrapper.model_wrapper,
            ic.sig_key,
            ic.bkg_key,
            significance_algo=significance_algo,
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
        new_x = (dnn_cut_list + dnn_cut_step / 2).tolist()
        new_y = [ic.sig_list[index]] * len(w_input)
        new_w = w_input
        logging.debug(f"new x: {new_x}")
        logging.debug(f"new y: {new_y}")
        logging.debug(f"new w: {new_w}")
        if len(x) == 0:
            x = new_x
            y = new_y
            w = new_w
        else:
            x += new_x
            y += new_y
            w += new_w
    # make plot
    plot_canvas = ROOT.TCanvas("2d_significance_c", "2d_significance_c", 800, 600)
    hist_sig = th1_tools.TH2FTool(
        "2d_significance",
        "2d_significance",
        nbinx=len(dnn_cut_list),
        xlow=dnn_cut_min,
        xup=dnn_cut_max,
        nbiny=len(ic.sig_list),
        ylow=0,
        yup=len(ic.sig_list),
    )
    hist_sig_text = th1_tools.TH2FTool(
        "2d_significance_text",
        "2d_significance_text",
        nbinx=len(dnn_cut_list),
        xlow=dnn_cut_min,
        xup=dnn_cut_max,
        nbiny=len(ic.sig_list),
        ylow=0,
        yup=len(ic.sig_list),
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
