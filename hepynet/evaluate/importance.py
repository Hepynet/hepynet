import logging

import matplotlib.pyplot as plt
import numpy as np

from hepynet.evaluate import roc
from hepynet.train import hep_model

logger = logging.getLogger("hepynet")


def plot_feature_importance(
    model_wrapper: hep_model.Model_Base, job_config, save_dir, log=True, max_feature=16
):
    """Calculates importance of features and sort the feature.

    Definition of feature importance used here can be found in:
    https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data

    """
    logger.info("Plotting feature importance.")
    ic = job_config.input.clone()
    tc = job_config.train.clone()
    # prepare
    model = model_wrapper.get_model()
    feedbox = model_wrapper.get_feedbox()
    num_feature = len(feedbox.get_job_config().input.selected_features)
    selected_feature_names = np.array(feedbox.get_job_config().input.selected_features)
    input_df = feedbox.get_train_test_df(
        sig_key=ic.sig_key,
        bkg_key=ic.bkg_key,
        multi_class_bkgs=tc.output_bkg_node_names,
        reset_mass=False,
    )
    cols = ic.selected_features
    test_index = input_df["is_train"] == False
    x_test = input_df.loc[test_index, cols].values
    y_test = input_df.loc[test_index, ["y"]].values
    wt_test = input_df.loc[test_index, "weight"].values
    all_nodes = []
    if y_test.ndim == 2:
        all_nodes = ["sig"] + tc.output_bkg_node_names
    else:
        all_nodes = ["sig"]
    # Make plots
    fig_save_pattern = f"{save_dir}/importance_{{}}.png"
    if num_feature > 16:
        canvas_height = 16
    else:
        canvas_height = num_feature
    base_auc = roc.calculate_auc(x_test, y_test, wt_test, model, rm_last_two=True)
    # Calculate importance
    feature_auc = []
    for num, feature_name in enumerate(selected_feature_names):
        current_auc = roc.calculate_auc(
            x_test, y_test, wt_test, model, shuffle_col=num, rm_last_two=True
        )
        feature_auc.append(current_auc)
    for node_id, node in enumerate(all_nodes):
        logger.info(f"making importance plot for node: {node}")
        fig_save_path = fig_save_pattern.format(node)
        fig, ax = plt.subplots(figsize=(9, canvas_height))
        logger.info(f"base auc: {base_auc[node_id]}")
        feature_importance = np.zeros(num_feature)
        for num, feature_name in enumerate(selected_feature_names):
            current_auc = feature_auc[num][node_id]
            feature_importance[num] = (1 - current_auc) / (1 - base_auc[node_id])
            logger.info(f"{feature_name} : {feature_importance[num]}")

        # Sort
        sort_list = np.flip(np.argsort(feature_importance))
        sorted_importance = feature_importance[sort_list]
        sorted_names = selected_feature_names[sort_list]
        logger.info(f"feature importance rank: {sorted_names}")
        # Plot
        if num_feature > max_feature:
            num_show = max_feature
        else:
            num_show = num_feature
        ax.barh(
            np.flip(np.arange(num_show)),
            sorted_importance[:num_show],
            align="center",
            alpha=0.5,
            log=log,
        )
        ax.axvline(x=1, ls="--", color="r")
        ax.set_title("feature importance")
        ax.set_yticks(np.arange(num_show))
        ax.set_yticklabels(sorted_names[:num_show])
        fig.savefig(fig_save_path)
