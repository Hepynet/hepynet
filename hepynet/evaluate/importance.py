import copy
import logging

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

import hepynet.common.hepy_type as ht
from hepynet.evaluate import evaluate_utils
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def calculate_auc(y_tag, y_pred, weights):
    """Returns auc of given sig/bkg array."""
    auc_value = []
    num_nodes = y_tag.shape[1]
    for node_id in range(num_nodes):
        fpr_dm, tpr_dm, _ = roc_curve(
            y_tag[:, node_id], y_pred[:, node_id], sample_weight=weights
        )
        sort_index = fpr_dm.argsort()
        fpr_dm = fpr_dm[sort_index]
        tpr_dm = tpr_dm[sort_index]
        auc_value.append(auc(fpr_dm, tpr_dm))
    return auc_value


def plot_feature_importance(
    model_wrapper: hep_model.Model_Base,
    df: pd.DataFrame,
    job_config,
    save_dir,
    max_feature=50,
):
    """Calculates importance of features and sort the feature.

    Definition of feature importance used here can be found in:
    https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data

    """
    logger.info("Plotting feature importance.")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    ac = job_config.apply.clone()
    # prepare
    model = model_wrapper.get_model()
    cols = ic.selected_features
    test_index = df["is_train"] == False
    y_test = df.loc[test_index, ["y"]].to_numpy("float32")
    y_pred = df.loc[test_index, ["y_pred"]].to_numpy("float32")
    wt_test = df.loc[test_index, "weight"].to_numpy("float32")
    all_nodes = ["sig"] + tc.output_bkg_node_names
    # Make plots
    fig_save_pattern = f"{save_dir}/importance_{{}}.png"
    num_feature = len(cols)
    if num_feature < max_feature:
        canvas_height = num_feature / 2 + 5
    else:
        canvas_height = max_feature / 2 + 5
    base_auc = calculate_auc(y_test, y_pred, wt_test)
    # Calculate importance
    feature_auc = []
    logger.info("Making predictions with shuffled one column")
    for num, feature_name in enumerate(cols):
        x_shuffle = df.loc[test_index, cols].values
        np.random.shuffle(x_shuffle[:, num])
        y_pred, _, _ = evaluate_utils.k_folds_predict(
            model, x_shuffle, silence=True
        )
        current_auc = calculate_auc(y_test, y_pred, wt_test)
        feature_auc.append(current_auc)
    for node_id, node in enumerate(all_nodes):
        logger.info(f"> making importance plot for node: {node}")
        fig_save_path = fig_save_pattern.format(node)
        fig, ax = plt.subplots(figsize=(11.111, canvas_height))
        logger.info(f"> base auc: {base_auc[node_id]}")
        feature_importance = np.zeros(num_feature)
        for num, feature_name in enumerate(cols):
            current_auc = feature_auc[num][node_id]
            feature_importance[num] = (1 - current_auc) / (
                1 - base_auc[node_id]
            )
            logger.info(f"> {feature_name} : {feature_importance[num]}")

        # Sort
        sort_list = np.argsort(feature_importance).astype(int)
        sorted_importance = feature_importance[sort_list]
        sorted_names = [cols[i] for i in sort_list]
        rank_str = " > ".join(sorted_names[::-1])
        logger.info(f"> feature importance rank: {rank_str}")
        # Plot
        if num_feature > max_feature:
            num_show = max_feature
        else:
            num_show = num_feature
        ax.barh(
            np.arange(num_show),
            sorted_importance[:num_show],
            align="center",
            alpha=0.5,
            log=ac.cfg_importance_study.log,
        )
        ax.axvline(x=1, ls="--", color="r")
        _, x_max = ax.get_xlim()
        ax.set_xlim(0.1, x_max)
        _, y_max = ax.get_ylim()
        ax.set_ylim(-0.5, y_max + 2)
        if ac.plot_atlas_label:
            ampl.plot.draw_atlas_label(
                0.05, 0.95, ax=ax, **(ac.atlas_label.get_config_dict())
            )
        ax.set_title("feature importance")
        ax.set_yticks(np.arange(num_show))
        ax.set_yticklabels(sorted_names[:num_show])
        fig.subplots_adjust(left=0.3)
        fig.savefig(fig_save_path)
