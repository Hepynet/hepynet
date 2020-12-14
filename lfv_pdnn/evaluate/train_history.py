import logging

import matplotlib.pyplot as plt

logger = logging.getLogger("lfv_pdnn")


def plot_history(
    model_wrapper, plot_config, save_dir=None,
):
    """Evaluates training result.

        Args:
            figsize: tuple
                Defines plot size.

        """
    logger.info("Plotting training history")
    config = plot_config.clone()
    # accuracy curve
    plot_accuracy(
        model_wrapper, config.accuracy, save_dir=save_dir,
    )
    # loss curve
    plot_loss(
        model_wrapper, config.loss, save_dir=save_dir,
    )


def plot_accuracy(model_wrapper, plot_config, save_dir: str = ".",) -> None:
    """Plots accuracy vs training epoch."""
    logger.info("Plotting accuracy curve")
    accuracy_list = model_wrapper.train_history_accuracy
    val_accuracy_list = model_wrapper.train_history_val_accuracy
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(accuracy_list)
    ax.plot(val_accuracy_list)
    # config
    ax.set_title(plot_config.plot_title)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"], loc="lower left")
    ax.grid()
    fig.savefig(f"{save_dir}/history_accuracy.{plot_config.save_format}")


def plot_loss(model_wrapper, plot_config, save_dir: str = ".",) -> None:
    """Plots loss vs training epoch."""
    logger.info("Plotting loss curve")
    loss_list = model_wrapper.train_history_loss
    val_loss_list = model_wrapper.train_history_val_loss
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(loss_list)
    ax.plot(val_loss_list)
    # config
    ax.set_title(plot_config.plot_title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(["train", "val"], loc="lower left")
    ax.grid()
    fig.savefig(f"{save_dir}/history_loss.{plot_config.save_format}")
