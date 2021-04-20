import itertools
import logging
import pathlib

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import hepynet.common.hepy_type as ht

logger = logging.getLogger("hepynet")


def plot_multi_class_roc(
    df: pd.DataFrame, job_config: ht.config, save_dir: ht.pathlike,
):
    """Plots roc curve."""
    logger.info("Plotting train/test roc curve.")
    # setup config
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    # prepare
    output_bkg_node_names = tc.output_bkg_node_names
    output_bkg_node_names = tc.output_bkg_node_names
    all_nodes = ["sig"] + output_bkg_node_names
    if df.shape[0] > 1000000:
        logger.warn(
            f"Too large input detected ({df.shape[0]} rows), randomly sampling 1000000 rows for roc calculation"
        )
        df = df.sample(n=1000000)
    train_index = df["is_train"] == True
    test_index = df["is_train"] == False
    y_train = df.loc[train_index, ["y"]].values
    y_train_pred = df.loc[train_index, ["y_pred"]].values
    y_test = df.loc[test_index, ["y"]].values
    y_test_pred = df.loc[test_index, ["y_pred"]].values
    wt_train = df.loc[train_index, "weight"].values
    wt_test = df.loc[test_index, "weight"].values
    # plot node by node
    num_nodes = len(all_nodes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = itertools.cycle(colors)
    auc_labels = []
    auc_contents = []
    fig, ax = plt.subplots()
    for node_num in range(num_nodes):
        color = next(color_cycle)
        # plot roc for train dataset without reseting mass
        auc_train, _, _ = plot_roc(
            ax,
            y_train,
            y_train_pred,
            wt_train,
            node_num=node_num,
            color=color,
            linestyle="dashed",
        )
        # plot roc for test dataset without reseting mass
        auc_test, _, _ = plot_roc(
            ax,
            y_test,
            y_test_pred,
            wt_test,
            node_num=node_num,
            color=color,
            linestyle="solid",
        )
        auc_labels += [
            f"tr-{all_nodes[node_num]} (AUC: {round(auc_train, 5)})",
            f"te-{all_nodes[node_num]} (AUC: {round(auc_test, 5)})",
        ]
        auc_contents += [round(auc_train, 5), round(auc_test, 5)]
    # extra plot config
    ax.legend(auc_labels, loc="lower right")
    ax.grid()
    # collect meta data
    auc_dict = {}
    auc_dict["auc_train_original"] = auc_train
    auc_dict["auc_test_original"] = auc_test
    # make plots
    ax.set_ylim(0, 1.4)
    ax.set_yscale("linear")
    if ac.plot_atlas_label:
        ampl.plot.draw_atlas_label(
            0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
        )
    ## save linear scale plot
    save_dir = pathlib.Path(save_dir)
    fig.savefig(save_dir / "roc_linear.png")
    ## log scale x
    ax.set_xlim(1e-5, 1)
    ax.set_xscale("log")
    fig.savefig(save_dir / "roc_logx.png")
    return auc_dict


def plot_roc(
    ax: ht.ax,
    y: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    node_num: int = 0,
    color: str = "blue",
    linestyle: str = "solid",
    yscal: str = "linear",
    ylim: ht.bound = (0, 1),
):
    """Plots roc curve on given axes."""
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    # Make plots
    fpr_dm, tpr_dm, _ = roc_curve(
        y[:, node_num], y_pred[:, node_num], sample_weight=weights
    )
    ax.plot
    ax.plot(fpr_dm, tpr_dm, color=color, linestyle=linestyle)
    ax.set_title("ROC Curve")
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_yscale(yscal)
    auc_value = my_roc_auc(y[:, node_num], y_pred[:, node_num], weights)
    return auc_value, fpr_dm, tpr_dm


# code from: https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py
def my_roc_auc(
    classes: np.ndarray, predictions: np.ndarray, weights: np.ndarray = None
) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    """

    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
        shape=len(classes),
        dtype=[("c", classes.dtype), ("p", predictions.dtype), ("w", weights.dtype)],
    )
    data["c"], data["p"], data["w"] = classes, predictions, weights

    data = data[np.argsort(data["c"])]
    data = data[
        np.argsort(data["p"], kind="mergesort")
    ]  # here we're relying on stability as we need class orders preserved

    correction = 0.0
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data["p"][1:] == data["p"][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        (ids,) = mask2.nonzero()
        correction = (
            sum(
                [
                    ((dsplit["c"] == class0) * dsplit["w"] * msplit).sum()
                    * ((dsplit["c"] == class1) * dsplit["w"] * msplit).sum()
                    for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))
                ]
            )
            * 0.5
        )

    weights_0 = data["w"] * (data["c"] == class0)
    weights_1 = data["w"] * (data["c"] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (
        weights_1.sum() * cumsum_0[-1]
    )
