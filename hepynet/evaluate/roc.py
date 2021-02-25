import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from sklearn.metrics import auc, roc_auc_score, roc_curve

from hepynet.common import array_utils, common_utils

logger = logging.getLogger("hepynet")


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


def plot_multi_class_roc(model_wrapper, job_config):
    """Plots roc curve."""
    logger.info("Plotting train/test roc curve.")
    # setup config
    rc = job_config.run
    ic = job_config.input
    tc = job_config.train
    # prepare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.feedbox
    output_bkg_node_names = tc.output_bkg_node_names
    all_nodes = ["sig"] + output_bkg_node_names
    train_test_dict = feedbox.get_train_test_arrays(
        sig_key=ic.sig_key,
        bkg_key=ic.bkg_key,
        multi_class_bkgs=output_bkg_node_names,
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
    fig, ax = plt.subplots(figsize=(8, 6))
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
        auc_labels += [f"tr_{all_nodes[node_num]}", f"te_{all_nodes[node_num]}"]
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
    if rc.save_dir is not None:
        ax.set_ylim(0, 1)
        ax.set_yscale("linear")
        fig.savefig(f"{rc.save_dir}/roc_linear.png")
        ax.set_ylim(0.1, 1 - 1e-4)
        ax.set_yscale("logit")
        fig.savefig(f"{rc.save_dir}/roc_logit.png")
    return auc_dict


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
    sort_ids = np.argsort(fpr_dm)
    # auc_value = roc_auc_score(y[:, node_num], y_pred[:, node_num], sample_weight=weights)
    # auc_value = auc(fpr_dm[sort_ids], tpr_dm[sort_ids])
    auc_value = my_roc_auc(y[:, node_num], y_pred[:, node_num], weights)
    return auc_value, fpr_dm, tpr_dm


# code from: https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py
def my_roc_auc(classes : np.ndarray,
               predictions : np.ndarray,
               weights : np.ndarray = None) -> float:
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
            dtype=[('c', classes.dtype),
                   ('p', predictions.dtype),
                   ('w', weights.dtype)]
        )
    data['c'], data['p'], data['w'] = classes, predictions, weights

    data = data[np.argsort(data['c'])]
    data = data[np.argsort(data['p'], kind='mergesort')] # here we're relying on stability as we need class orders preserved

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data['p'][1:] == data['p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit['c'] == class0) * dsplit['w'] * msplit).sum() * 
                          ((dsplit['c'] == class1) * dsplit['w'] * msplit).sum()
                          for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))]) * 0.5
 
    weights_0 = data['w'] * (data['c'] == class0)
    weights_1 = data['w'] * (data['c'] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (weights_1.sum() * cumsum_0[-1])