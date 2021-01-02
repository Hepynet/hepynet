import logging

import matplotlib.pyplot as plt

from hepynet.common.common_utils import get_default_if_none

logger = logging.getLogger("hepynet")


def plot_history(
    model_wrapper, job_config, save_dir=None,
):
    """Evaluates training result.

        Args:
            figsize: tuple
                Defines plot size.

        """
    logger.info("Plotting training history")
    plot_config = job_config.apply.cfg_history
    train_history = model_wrapper._train_history
    for metric_key in train_history.keys():
        if not metric_key.startswith("val_"):
            plot_metrics(metric_key, train_history, plot_config, save_dir=save_dir)


def plot_metrics(metric_name, train_history, plot_config, save_dir: str = ".",) -> None:
    logger.info(f"Plotting {metric_name}")
    metric_entries = train_history[metric_name]
    val_metric_entries = train_history["val_" + metric_name]
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metric_entries)
    ax.plot(val_metric_entries)
    # config
    plot_title = get_default_if_none(plot_config.plot_title, metric_name)
    ax.set_title(plot_title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"])
    ax.grid()
    save_format = get_default_if_none(plot_config.save_format, "png")
    fig.savefig(f"{save_dir}/history_{metric_name}.{save_format}")
