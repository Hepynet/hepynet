import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from hepynet.common.common_utils import get_default_if_none
from hepynet.evaluate import evaluate_utils
from hepynet.train import hep_model

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
        sig_scores_dict[sig_key], _, _ = evaluate_utils.k_folds_predict(
            model, input_df[ic.selected_features].values
        )
        sig_weights_dict[sig_key] = input_df["weight"]

    # prepare background
    bkg_scores_dict = {}
    bkg_weights_dict = {}
    for bkg_key in plot_config.bkg_list:
        input_df = feedbox.get_reshape_merged("xb", array_key=bkg_key)
        bkg_scores_dict[bkg_key], _, _ = evaluate_utils.k_folds_predict(
            model, input_df[ic.selected_features].values
        )
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
    tc = job_config.train.clone()
    rc = job_config.run.clone()
    plot_config = job_config.apply.cfg_train_test_compare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.get_feedbox()
    all_nodes = ["sig"] + tc.output_bkg_node_names

    input_dir = pathlib.Path(rc.save_sub_dir) / "input"
    x_train = np.load(input_dir / "x_train.npy")
    x_test = np.load(input_dir / "x_test.npy")
    y_train = np.load(input_dir / "y_train.npy")
    y_test = np.load(input_dir / "y_test.npy")
    wt_train = np.load(input_dir / "wt_train.npy")
    wt_test = np.load(input_dir / "wt_test.npy")

    xs_train = x_train[y_train == 1]
    xs_test = x_test[y_test == 1]
    xs_train_weight = wt_train[y_train == 1]
    xs_test_weight = wt_test[y_test ==  1]
    xb_train = x_train[y_train == 0]
    xb_test = x_test[y_test == 0]
    xb_train_weight = wt_train[y_train == 0]
    xb_test_weight = wt_test[y_test == 0]



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
            xb_train_scores[:, [node_num]],
            "b-train",
            weights=xb_train_weight,
            bins=plot_config.bins,
            range=plot_config.range,
            density=plot_config.density,
            use_error=True,
            # color="darkblue",
            fmt=".",
        )
        evaluate_utils.paint_bars(
            ax,
            xs_train_scores[:, [node_num]],
            "s-train",
            weights=xs_train_weight,
            bins=plot_config.bins,
            range=plot_config.range,
            density=plot_config.density,
            use_error=True,
            # color="maroon",
            fmt=".",
        )
        ax.legend(loc="upper center")
        # Make and show plots
        if feedbox.get_job_config().input.reset_mass:
            file_name = f"mva_scores_{all_nodes[node_num]}_original_mass"
        else:
            file_name = f"mva_scores_{all_nodes[node_num]}_reset_mass"
        fig.savefig(save_dir / f"{file_name}.{plot_config.save_format}")
