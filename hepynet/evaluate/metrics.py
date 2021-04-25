import itertools
import logging
import pathlib
from typing import Tuple

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import save
import pandas as pd
from seaborn.matrix import heatmap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import hepynet.common.hepy_type as ht

logger = logging.getLogger("hepynet")


def make_metrics_plot(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    job_config: ht.config,
    save_dir: ht.pathlike,
):
    """Plots PR curve."""
    ac = job_config.apply.clone()
    if df.shape[0] > 1000000:
        logger.warn(
            f"Too large input detected ({df.shape[0]} rows), randomly sampling 1000000 rows for metrics calculation"
        )
        df = df.sample(n=1000000)
    train_index = df["is_train"] == True
    test_index = df["is_train"] == False
    y_train = df.loc[train_index, ["y"]].values
    y_train_pred = df.loc[train_index, ["y_pred"]].values
    y_test = df.loc[test_index, ["y"]].values
    y_test_pred = df.loc[test_index, ["y_pred"]].values
    wt_raw_train = df_raw.loc[train_index, "weight"].values  # need raw weights
    wt_raw_test = df_raw.loc[test_index, "weight"].values
    train_inputs = (y_train, y_train_pred, wt_raw_train)
    test_inputs = (y_test, y_test_pred, wt_raw_test)

    save_dir = pathlib.Path(save_dir) / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    if ac.book_confusion_matrix:
        make_confusion_matrix_plot(
            train_inputs, test_inputs, job_config, save_dir
        )
    if ac.book_pr:
        make_pr_curve_plot(train_inputs, test_inputs, job_config, save_dir)
    if ac.book_roc:
        make_roc_curve_plot(train_inputs, test_inputs, job_config, save_dir)


def make_confusion_matrix_plot(
    train_inputs: Tuple[np.ndarray],
    test_inputs: Tuple[np.ndarray],
    job_config: ht.config,
    save_dir: ht.pathlike,
):
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    score_cut = ac.cfg_confusion_matrix.dnn_cut
    save_dir = pathlib.Path(save_dir) / f"confusion_matrix"
    save_dir.mkdir(exist_ok=True, parents=True)
    # prepare
    output_bkg_node_names = tc.output_bkg_node_names
    output_bkg_node_names = tc.output_bkg_node_names
    all_nodes = ["sig"] + output_bkg_node_names
    y_train, y_train_pred, wt_train = train_inputs
    y_test, y_test_pred, wt_test = test_inputs
    # plot node by node
    row_label = ["Negative Class", "Positive Class"]
    col_label = ["Negative Prediction", "Positive Prediction"]
    for node_num, node in enumerate(all_nodes):
        # train dataset
        con_matrix_train = confusion_matrix(
            y_train[:, node_num],
            y_train_pred[:, node_num] > score_cut,
            sample_weight=wt_train,
        )
        matrix_df = pd.DataFrame(con_matrix_train)
        matrix_df.columns = col_label
        matrix_df.index = row_label
        matrix_df["Total"] = matrix_df.sum(axis=1, numeric_only=True)
        matrix_df.loc["Total"] = matrix_df.sum(numeric_only=True)
        matrix_df.to_csv(save_dir / f"node_{node}_cut_{score_cut}_train.csv")
        fig, ax = plt.subplots()
        heatmap(con_matrix_train, cmap="coolwarm_r", annot=True, ax=ax)
        ax.set_title(f"node {node} - cut {score_cut} - train")
        ax.set_xlabel("Predicted Classes")
        ax.set_ylabel("Real Classes")
        fig.savefig(save_dir / f"node_{node}_cut_{score_cut}_train.png")
        with open(
            save_dir / f"node_{node}_cut_{score_cut}_train_report.txt"
        ) as report_file:
            report = classification_report(
                y_train, y_train_pred[:, node_num] > score_cut
            )
            print(report, file=report_file)
        # test dataset
        con_matrix_test = confusion_matrix(
            y_test[:, node_num],
            y_test_pred[:, node_num] > score_cut,
            sample_weight=wt_test,
        )
        matrix_df = pd.DataFrame(con_matrix_test)
        matrix_df.columns = col_label
        matrix_df.index = row_label
        matrix_df["Total"] = matrix_df.sum(axis=1, numeric_only=True)
        matrix_df.loc["Total"] = matrix_df.sum(numeric_only=True)
        matrix_df.to_csv(save_dir / f"node_{node}_cut_{score_cut}_test.csv")
        fig, ax = plt.subplots()
        heatmap(con_matrix_test, cmap="coolwarm_r", annot=True, ax=ax)
        ax.set_title(f"node {node} - cut {score_cut} - test")
        ax.set_xlabel("Predicted Classes")
        ax.set_ylabel("Real Classes")
        fig.savefig(save_dir / f"node_{node}_cut_{score_cut}_test.png")
        with open(
            save_dir / f"node_{node}_cut_{score_cut}_test_report.txt"
        ) as report_file:
            report = classification_report(
                y_test, y_test_pred[:, node_num] > score_cut
            )
            print(report, file=report_file)


def make_pr_curve_plot(
    train_inputs: Tuple[np.ndarray],
    test_inputs: Tuple[np.ndarray],
    job_config: ht.config,
    save_dir: ht.pathlike,
):
    """Plots PR curve."""
    logger.info("Plotting train/test PR curve.")
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    # prepare
    output_bkg_node_names = tc.output_bkg_node_names
    output_bkg_node_names = tc.output_bkg_node_names
    all_nodes = ["sig"] + output_bkg_node_names
    y_train, y_train_pred, wt_train = train_inputs
    y_test, y_test_pred, wt_test = test_inputs
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
        auc_train, _, _ = plot_single_pr(
            ax,
            y_train[:, node_num],
            y_train_pred[:, node_num],
            wt_train,
            node_num=node_num,
            color=color,
            linestyle="dashed",
        )
        # plot roc for test dataset without reseting mass
        auc_test, _, _ = plot_single_pr(
            ax,
            y_test[:, node_num],
            y_test_pred[:, node_num],
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
    fig.savefig(save_dir / "pr_linear.png")
    ## log scale x
    ax.set_xlim(1e-5, 1)
    ax.set_xscale("log")
    fig.savefig(save_dir / "pr_logx.png")
    return auc_dict


def make_roc_curve_plot(
    train_inputs: Tuple[np.ndarray],
    test_inputs: Tuple[np.ndarray],
    job_config: ht.config,
    save_dir: ht.pathlike,
):
    """Plots ROC curve."""
    logger.info("Plotting train/test ROC curve.")
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    # prepare
    output_bkg_node_names = tc.output_bkg_node_names
    output_bkg_node_names = tc.output_bkg_node_names
    all_nodes = ["sig"] + output_bkg_node_names
    y_train, y_train_pred, wt_train = train_inputs
    y_test, y_test_pred, wt_test = test_inputs
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
        auc_train, _, _ = plot_single_roc(
            ax,
            y_train[:, node_num],
            y_train_pred[:, node_num],
            wt_train,
            node_num=node_num,
            color=color,
            linestyle="dashed",
        )
        # plot roc for test dataset without reseting mass
        auc_test, _, _ = plot_single_roc(
            ax,
            y_test[:, node_num],
            y_test_pred[:, node_num],
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
    fig.savefig(save_dir / "roc_linear.png")
    ## log scale x
    ax.set_xlim(1e-5, 1)
    ax.set_xscale("log")
    fig.savefig(save_dir / "roc_logx.png")
    return auc_dict


def plot_single_pr(
    ax: ht.ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    node_num: int = 0,
    color: str = "blue",
    linestyle: str = "solid",
    yscal: str = "linear",
    ylim: ht.bound = (0, 1),
):
    """Plots PR (Precision-Recall) curve on given axes."""
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    # Make plots
    precision, recall, _ = precision_recall_curve(
        y_true[:, node_num], y_pred[:, node_num], sample_weight=weights
    )
    # precision = np.concatenate(([0], precision, [1]))
    # recall = np.concatenate(([1], recall, [0]))
    ax.plot
    ax.plot(recall, precision, color=color, linestyle=linestyle)
    ax.set_title("PR Curve")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_yscale(yscal)
    # auc_value = my_roc_auc(y_true[:, node_num], y_pred[:, node_num], weights)
    auc_value = np.trapz(
        y=precision, x=recall
    )  # use Trapezium rule for simplicity
    return auc_value, precision, recall


def plot_single_roc(
    ax: ht.ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    node_num: int = 0,
    color: str = "blue",
    linestyle: str = "solid",
    yscal: str = "linear",
    ylim: ht.bound = (0, 1),
):
    """Plots ROC (receiver operating characteristic) curve on given axes."""
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    # Make plots
    fpr_dm, tpr_dm, _ = roc_curve(
        y_true[:, node_num], y_pred[:, node_num], sample_weight=weights
    )
    fpr_dm = np.concatenate(([0], fpr_dm, [1]))
    tpr_dm = np.concatenate(([0], tpr_dm, [1]))
    ax.plot
    ax.plot(fpr_dm, tpr_dm, color=color, linestyle=linestyle)
    ax.set_title("ROC Curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_yscale(yscal)
    # auc_value = my_roc_auc(y_true[:, node_num], y_pred[:, node_num], weights)
    auc_value = np.trapz(
        y=tpr_dm, x=fpr_dm
    )  # use Trapezium rule for simplicity
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
        dtype=[
            ("c", classes.dtype),
            ("p", predictions.dtype),
            ("w", weights.dtype),
        ],
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
                    for dsplit, msplit in zip(
                        np.split(data, ids), np.split(mask1, ids)
                    )
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
