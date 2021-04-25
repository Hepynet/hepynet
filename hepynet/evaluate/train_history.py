import logging

import matplotlib.pyplot as plt

from hepynet.common.common_utils import get_default_if_none
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


def get_metrics_band(train_history, metric_name, num_folds):
    # different folds can have different length of metrics
    folds_lengths = list()
    train_folds = list()
    val_folds = list()
    for fold_num in range(num_folds):
        train_fold = train_history[fold_num][metric_name]
        folds_lengths.append(len(train_fold))
        train_folds.append(train_fold)
        val_fold = train_history[fold_num]["val_" + metric_name]
        val_folds.append(val_fold)

    max_len = max(folds_lengths)

    # process train metrics
    (
        train_mean,
        train_low,
        train_high,
    ) = train_utils.merge_unequal_length_arrays(train_folds)
    val_mean, val_low, val_high = train_utils.merge_unequal_length_arrays(
        val_folds
    )

    return {
        "epoch": range(max_len),
        "mean": train_mean,
        "low": train_low,
        "high": train_high,
        "val_mean": val_mean,
        "val_low": val_low,
        "val_high": val_high,
    }


def plot_history(model_wrapper, job_config, save_dir=None):
    """Evaluates training result.

        Args:
            figsize: tuple
                Defines plot size.

        """
    logger.info("Plotting training history")
    plot_config = job_config.apply.cfg_history
    train_history = model_wrapper._train_history
    num_folds = model_wrapper._num_folds
    for metric_key in train_history[0].keys():
        if not metric_key.startswith("val_"):
            plot_metrics(
                metric_key,
                train_history,
                plot_config,
                num_folds=num_folds,
                save_dir=save_dir,
            )


def plot_metrics(
    metric_name,
    train_history,
    plot_config,
    num_folds: int = 1,
    save_dir: str = ".",
) -> None:
    logger.info(f"> Plotting {metric_name}")
    plot_dict = get_metrics_band(train_history, metric_name, num_folds)
    # plot
    fig, ax = plt.subplots()
    ax.plot(plot_dict["epoch"], plot_dict["mean"], label="train")
    if num_folds > 1:
        ax.fill_between(
            plot_dict["epoch"],
            plot_dict["low"],
            plot_dict["high"],
            alpha=0.2,
            label="k-folds train",
        )
    ax.plot(plot_dict["epoch"], plot_dict["val_mean"], label="val")
    if num_folds > 1:
        ax.fill_between(
            plot_dict["epoch"],
            plot_dict["val_low"],
            plot_dict["val_high"],
            alpha=0.2,
            label="k-folds val",
        )
    # config
    plot_title = get_default_if_none(plot_config.plot_title, metric_name)
    ax.set_title(plot_title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid()
    save_format = get_default_if_none(plot_config.save_format, "png")
    fig.savefig(f"{save_dir}/history_{metric_name}.{save_format}")
