import logging
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hepynet.common import config_utils
from hepynet.data_io import numpy_io
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def create_epoch_subdir(save_dir, epoch, n_digit) -> pathlib.Path:
    if save_dir is None:
        logger.error(f"Invalid save_dir: {save_dir}")
        return None
    if epoch is not None:
        sub_dir = pathlib.Path(f"{save_dir}/epoch_{str(epoch).zfill(n_digit)}")
    else:
        sub_dir = pathlib.Path(f"{save_dir}/epoch_final")
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir


def dump_fit_df(
    model_wrapper: hep_model.Model_Base,
    df_raw,
    df_train,
    job_config,
    save_dir="./",
):
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()

    sample_list = ic.sig_list + ic.bkg_list
    if ic.apply_data:
        sample_list += ic.data_list

    platform_meta = config_utils.load_current_platform_meta()
    data_path = platform_meta["data_path"]
    if not data_path:
        save_dir = pathlib.Path(save_dir)
    else:
        save_dir = pathlib.Path(data_path) / save_dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"> Saving dataframes to: {save_dir}")

    for sample in sample_list:
        logger.info(f"> > processing: {sample}")
        dump_branches = ac.fit_df.branches + ["weight"]
        # prepare contents
        dump_df: pd.DataFrame = df_raw.loc[
            df_raw["sample_name"] == sample, dump_branches
        ]
        input_df = df_train.loc[
            df_train["sample_name"] == sample, ic.selected_features
        ]
        predictions, _, _ = k_folds_predict(
            model_wrapper.get_model(), input_df.values, silence=True
        )
        # dump
        if len(tc.output_bkg_node_names) == 0:
            dump_df["dnn_out_sig"] = predictions
        else:
            for i, out_node in enumerate(["sig"] + tc.output_bkg_node_names):
                out_node = out_node.replace("+", "_")
                branch_name = f"dnn_out_{out_node}"
                dump_df[branch_name] = predictions[:, i]
        dump_df.reset_index(inplace=True)
        save_path = save_dir / f"{sample}.feather"
        dump_df.to_feather(save_path)


def dump_fit_npy(
    model_wrapper: hep_model.Model_Base,
    df_raw,
    df_train,
    job_config,
    npy_dir="./",
):
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()

    sample_list = ic.sig_list + ic.bkg_list
    if ic.apply_data:
        sample_list += ic.data_list

    platform_meta = config_utils.load_current_platform_meta()
    data_path = platform_meta["data_path"]
    if not data_path:
        save_dir = pathlib.Path(npy_dir)
    else:
        save_dir = pathlib.Path(data_path) / npy_dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"> Arrays to be saved to {save_dir}")

    for sample in sample_list:
        dump_branches = ac.cfg_fit_npy.fit_npy_branches + ["weight"]
        # prepare contents
        dump_df = df_raw.loc[df_raw["sample_name"] == sample, dump_branches]
        input_df = df_train.loc[
            df_train["sample_name"] == sample, ic.selected_features
        ]
        predictions, _, _ = k_folds_predict(
            model_wrapper.get_model(), input_df.values
        )
        # dump
        for branch in dump_branches:
            branch_content = dump_df[branch].values
            save_path = f"{save_dir}/{sample}_{branch}.npy"
            numpy_io.save_npy_array(branch_content, save_path)
        if len(tc.output_bkg_node_names) == 0:
            save_path = f"{save_dir}/{sample}_dnn_out.npy"
            numpy_io.save_npy_array(predictions, save_path)
        else:
            for i, out_node in enumerate(["sig"] + tc.output_bkg_node_names):
                out_node = out_node.replace("+", "_")
                save_path = f"{save_dir}/{sample}_dnn_out_{out_node}.npy"
                numpy_io.save_npy_array(predictions[:, i], save_path)


def k_folds_predict(
    k_fold_models, x, silence=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pred_k_folds = list()
    if not silence:
        logger.info("Predicting y scores")
    num_folds = len(k_fold_models)
    for fold_num, fold_model in enumerate(k_fold_models):
        if not silence:
            logger.info(
                f"> Predicting with model in fold {fold_num + 1}/{num_folds}"
            )
        y_fold_pred = fold_model.predict(x)
        y_pred_k_folds.append(y_fold_pred)
    y_pred_mean = np.mean(y_pred_k_folds, axis=0)
    y_pred_max = np.maximum.reduce(y_pred_k_folds)
    y_pred_min = np.minimum.reduce(y_pred_k_folds)
    return y_pred_mean, y_pred_min, y_pred_max


def paint_bars(
    ax,
    data,
    labels: list,
    weights,
    bins: int,
    range: tuple,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    x_unit: str = None,
    x_scale: float = None,
    density: bool = False,
    use_error: bool = False,
    color: str = None,
    fmt: str = ".k",
) -> None:
    """Plot with vertical bar, can be used for data display.

        Note:
        According to ROOT:
        "The error per bin will be computed as sqrt(sum of squares of weight) for each bin."

    """
    plt.ioff()
    # Check input
    data_1dim = np.array([])
    weight_1dim = np.array([])
    if isinstance(data, np.ndarray):
        data = [data]
        weights = [weights]
    for datum, weight in zip(data, weights):
        assert isinstance(
            datum, np.ndarray
        ), "data element should be numpy array."
        assert isinstance(
            weight, np.ndarray
        ), "weights element should be numpy array."
        assert (
            datum.shape == weight.shape
        ), "Input weights should be None or have same type as arrays."
        if len(data_1dim) == 0:
            data_1dim = datum
            weight_1dim = weight
        else:
            data_1dim = np.concatenate((data_1dim, datum))
            weight_1dim = np.concatenate((weight_1dim, weight))

    # Scale x axis
    if x_scale is not None:
        data_1dim = data_1dim * x_scale
    # Make bar plot
    # get bin error and edges
    plot_ys, _ = np.histogram(
        data_1dim, bins=bins, range=range, weights=weight_1dim, density=density
    )
    sum_weight_squares, bin_edges = np.histogram(
        data_1dim, bins=bins, range=range, weights=np.power(weight_1dim, 2)
    )
    if density:
        error_scale = 1 / (np.sum(weight_1dim) * (range[1] - range[0]) / bins)
        errors = np.sqrt(sum_weight_squares) * error_scale
    else:
        errors = np.sqrt(sum_weight_squares)
    # Only plot ratio when bin is not 0.
    bin_centers = np.array([])
    bin_ys = np.array([])
    bin_yerrs = np.array([])
    for i, y1 in enumerate(plot_ys):
        if y1 != 0:
            ele_center = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1])])
            bin_centers = np.concatenate((bin_centers, ele_center))
            ele_y = np.array([y1])
            bin_ys = np.concatenate((bin_ys, ele_y))
            ele_yerr = np.array([errors[i]])
            bin_yerrs = np.concatenate((bin_yerrs, ele_yerr))
    # plot bar
    bin_size = bin_edges[1] - bin_edges[0]
    if use_error:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=bin_yerrs,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    else:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=None,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    # Config
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        if x_unit is not None:
            ax.set_xlabel(x_label + "/" + x_unit)
        else:
            ax.set_xlabel(x_label)
    else:
        if x_unit is not None:
            ax.set_xlabel(x_unit)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if range is not None:
        ax.axis(xmin=range[0], xmax=range[1])
    ax.legend(loc="upper right")
