import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from hepynet.evaluate import evaluate_utils
from hepynet.train import hep_model
from hepynet.common import common_utils

logger = logging.getLogger("hepynet")


# TODO: plots look strange, need to check the implementation with new data structure
# use with cautious
def plot_mva_scores(
    model_wrapper: hep_model.Model_Base, job_config, save_dir, file_name="mva_scores"
):
    # initialize
    logger.info("Plotting MVA scores")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    plot_config = ac.cfg_mva_scores_data_mc.clone()
    model = model_wrapper.get_model()
    feedbox = model_wrapper.get_feedbox()

    # prepare signal
    sig_scores_dict = {}
    sig_weights_dict = {}
    for sig_key in plot_config.sig_list:
        input_df = feedbox.get_reshape_merged("xs", array_key=sig_key)
        sig_score, _, _ = evaluate_utils.k_folds_predict(
            model, input_df[ic.selected_features].values
        )
        if sig_score.ndim == 1:
            sig_score = sig_score.reshape((-1, 1))
        sig_scores_dict[sig_key] = sig_score
        sig_weights_dict[sig_key] = input_df["weight"]

    # prepare background
    bkg_scores_dict = {}
    bkg_weights_dict = {}
    for bkg_key in plot_config.bkg_list:
        input_df = feedbox.get_reshape_merged("xb", array_key=bkg_key)
        bkg_score, _, _ = evaluate_utils.k_folds_predict(
            model, input_df[ic.selected_features].values
        )
        if bkg_score.ndim == 1:
            bkg_score = bkg_score.reshape((-1, 1))
        bkg_scores_dict[bkg_key] = bkg_score
        bkg_weights_dict[bkg_key] = input_df["weight"]

    # prepare data
    data_scores = np.array([])
    data_weights = np.array([])
    if plot_config.apply_data:
        data_key = plot_config.data_key
        input_df = feedbox.get_reshape_merged("xd", array_key=data_key)
        data_scores, _, _ = evaluate_utils.k_folds_predict(
            model, input_df[ic.selected_features].values
        )
        data_weights = input_df["weight"]

    # make plots
    all_nodes = ["sig"] + tc.output_bkg_node_names
    for node_id, node in enumerate(all_nodes):
        fig, ax = plt.subplots()
        sig_node_dict = {}
        for key, value in sig_scores_dict.items():
            sig_node_dict[key] = value[:, node_id]
        bkg_node_dict = {}
        for key, value in bkg_scores_dict.items():
            bkg_node_dict[key] = value[:, node_id]
        plot_scores_plt(
            ax,
            plot_config,
            sig_node_dict,
            sig_weights_dict,
            bkg_node_dict,
            bkg_weights_dict,
            data_scores,
            data_weights,
        )
        fig.savefig(f"{save_dir}/{file_name}_node_{node}.{plot_config.save_format}")

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


def plot_train_test_compare(model_wrapper: hep_model.Model_Base, job_config, save_dir):
    """Plots train/test scores distribution to check overtrain"""
    # initialize
    logger.info("Plotting train/test scores (original mass).")
    ic = job_config.input
    tc = job_config.train
    plot_config = job_config.apply.cfg_train_test_compare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.get_feedbox()
    sig_key = common_utils.get_default_if_none(plot_config.sig_key, ic.sig_key)
    bkg_key = common_utils.get_default_if_none(plot_config.bkg_key, ic.bkg_key)
    all_nodes = ["sig"] + tc.output_bkg_node_names

    input_df = feedbox.get_train_test_df(
        sig_key=sig_key,
        bkg_key=bkg_key,
        multi_class_bkgs=tc.output_bkg_node_names,
        reset_mass=feedbox.get_job_config().input.reset_mass,
    )
    cols = ic.selected_features
    train_index = input_df["is_train"] == True
    test_index = input_df["is_train"] == False
    sig_index = (input_df["is_sig"] == True) & (input_df["is_mc"] == True)
    bkg_index = (input_df["is_sig"] == False) & (input_df["is_mc"] == True)
    xs_train = input_df.loc[sig_index & train_index, cols].values
    xs_test = input_df.loc[sig_index & test_index, cols].values
    xs_train_weight = input_df.loc[sig_index & train_index, ["weight"]].values
    xs_test_weight = input_df.loc[sig_index & test_index, ["weight"]].values
    xb_train = input_df.loc[bkg_index & train_index, cols].values
    xb_test = input_df.loc[bkg_index & test_index, cols].values
    xb_train_weight = input_df.loc[bkg_index & train_index, ["weight"]].values
    xb_test_weight = input_df.loc[bkg_index & test_index, ["weight"]].values

    # plot for each nodes
    num_nodes = len(all_nodes)
    xb_train_scores, _, _ = evaluate_utils.k_folds_predict(model, xb_train)
    xs_train_scores, _, _ = evaluate_utils.k_folds_predict(model, xs_train)
    xb_test_scores, _, _ = evaluate_utils.k_folds_predict(model, xb_test)
    xs_test_scores, _, _ = evaluate_utils.k_folds_predict(model, xs_test)
    for node_num in range(num_nodes):
        fig, ax = plt.subplots()
        # plot scores
        ## plot train scores
        plot_scores_plt(
            ax,
            plot_config,
            {"s-test": xs_test_scores[:, node_num].flatten()},
            {"s-test": xs_test_weight.flatten()},
            {"b-test": xb_test_scores[:, node_num].flatten()},
            {"b-test": xb_test_weight.flatten()},
        )
        ## plot test scores
        evaluate_utils.paint_bars(
            ax,
            xb_train_scores[:, [node_num]].flatten(),
            "b-train",
            weights=xb_train_weight.flatten(),
            bins=plot_config.bins,
            range=plot_config.range,
            density=plot_config.density,
            use_error=True,
            color="green",
            fmt=".",
        )
        evaluate_utils.paint_bars(
            ax,
            xs_train_scores[:, [node_num]].flatten(),
            "s-train",
            weights=xs_train_weight.flatten(),
            bins=plot_config.bins,
            range=plot_config.range,
            density=plot_config.density,
            use_error=True,
            color="pink",
            fmt=".",
        )
        ax.legend(loc="upper center")
        # Make and show plots
        if feedbox.get_job_config().input.reset_mass:
            file_name = f"mva_scores_{all_nodes[node_num]}_original_mass"
        else:
            file_name = f"mva_scores_{all_nodes[node_num]}_reset_mass"
        fig.savefig(save_dir / f"{file_name}.{plot_config.save_format}")
