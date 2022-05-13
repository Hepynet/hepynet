import logging

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import yaml

from hepynet.common.common_utils import get_default_if_none
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


def get_metrics_band(train_history, metric_name, num_folds):
    # Different folds can have different length of metrics
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
    # Process train metrics
    (
        train_mean,
        train_low,
        train_high,
    ) = train_utils.merge_unequal_length_arrays(train_folds)
    val_mean, val_low, val_high = train_utils.merge_unequal_length_arrays(
        val_folds
    )
    return {
        "epoch": list(range(max_len)),
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
    train_history = model_wrapper._train_history
    num_folds = model_wrapper._num_folds
    for metric_key in train_history[0].keys():
        if not metric_key.startswith("val_"):
            plot_metrics(
                metric_key,
                train_history,
                job_config,
                num_folds=num_folds,
                save_dir=save_dir,
            )


def plot_metrics(
    metric_name,
    train_history,
    job_config,
    num_folds: int = 1,
    save_dir: str = ".",
) -> None:
    logger.info(f"> Plotting {metric_name}")
    ac = job_config.apply
    plot_dict = get_metrics_band(train_history, metric_name, num_folds)
    # Plot
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
    if ac.plot_atlas_label:
        ampl.plot.draw_atlas_label(
            0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
        )
    # Config
    ax.set_title(f"{metric_name} vs epoch")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("epoch")
    y_lim = getattr(ac.cfg_history.y_lim, metric_name, None)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend(loc="upper right")
    ax.grid()
    save_format = get_default_if_none(ac.cfg_history.save_format, ["png"])
    try:
        for fmt in save_format:
            fig.savefig(f"{save_dir}/history_{metric_name}.{fmt}")
    except:
        fig.savefig(f"{save_dir}/history_{metric_name}.{save_format}")
    # Save yaml data
    with open(f"{save_dir}/history_{metric_name}.yaml", "w") as f:
        yaml.dump(plot_dict, f, default_flow_style=False)
