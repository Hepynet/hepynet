import ast
import csv
import datetime
import logging
import math
import pathlib
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import auc, roc_curve

from hepynet.common import array_utils, common_utils, config_utils
from hepynet.common.hepy_const import SCANNED_PARAS
from hepynet.data_io import numpy_io
from hepynet.evaluate import (
    importance,
    kinematics,
    mva_scores,
    roc,
    significance,
    train_history,
)
from hepynet.main import job_utils
from hepynet.train import model, train_utils
logger = logging.getLogger("hepynet")


class job_executor(object):
    """Core class to execute a pdnn job based on given cfg file."""

    def __init__(self, yaml_config_path):
        """Initialize executor."""
        self.job_config = None
        self.get_config(yaml_config_path)
        # timing

    def execute_jobs(self):
        """Execute all planned jobs."""
        self.job_config.print()
        self.set_save_dir()
        # Execute single job if parameter scan is not needed
        if not self.job_config.para_scan.perform_para_scan:
            self.execute_single_job()
        # Otherwise perform scan as specified
        else:
            # TODO: add Basyesian scan back
            pass
            # self.get_scan_space()
            # self.execute_tuning_jobs()
            # # Perform final training with best hyperparmaters
            # logger.info("#" * 80)
            # logger.info("Performing final training with best hyper parameter set")
            # keys = list(self.best_hyper_set.keys())
            # for key in keys:
            #     value = self.best_hyper_set[key]
            #     value = float(value)
            #     if type(value) is float:
            #         if value.is_integer():
            #             value = int(value)
            #     setattr(self, key, value)
            # self.execute_single_job()
            # return 0

    def execute_single_job(self):
        """Execute single DNN training with given configuration."""
        # Prepare
        jc = self.job_config.job
        rc = self.job_config.run
        if jc.job_type == "apply":
            if rc.load_dir == None:
                rc.load_dir = jc.save_dir

        # set up model
        self.set_model()
        # set up inputs
        self.set_model_input()

        # train or apply
        if jc.job_type == "train":
            self.execute_train_job()
        elif jc.job_type == "apply":
            self.execute_apply_job()
        else:
            logger.critical(
                f"job.job_type must be train or apply, {jc.job_type} is not supported"
            )

        # post procedure
        plt.close("all")

        # return training meta data
        return self.model_wrapper.get_train_performance_meta()

    def execute_train_job(self):
        # train
        self.model_wrapper.compile()
        mod_save_path = f"{self.job_config.run.save_sub_dir}/models"
        self.model_wrapper.train(
            self.job_config, model_save_dir=mod_save_path,
        )
        # save model and meta data
        tc = self.job_config.train
        if tc.save_model:
            model_save_name = tc.model_name
            self.model_wrapper.save_model(
                save_dir=mod_save_path, file_name=model_save_name
            )

    def execute_apply_job(self):
        jc = self.job_config.job
        rc = self.job_config.run
        ic = self.job_config.input
        tc = self.job_config.train
        ac = self.job_config.apply
        # setup save parameters if reports need to be saved
        fig_save_path = None
        rc.save_dir = f"{rc.save_sub_dir}/apply/{jc.job_name}"
        pathlib.Path(rc.save_dir).mkdir(parents=True, exist_ok=True)

        # save metrics curve
        if ac.book_history:
            train_history.plot_history(
                self.model_wrapper, self.job_config, save_dir=rc.save_dir
            )

        # save kinematic plots
        if ac.book_kine_study:
            logger.info("Making input distribution plots")
            kinematics.plot_input_distributions(
                self.model_wrapper,
                self.job_config,
                save_dir=f"{rc.save_sub_dir}/kinematics/raw",
            )
            kinematics.plot_input_distributions(
                self.model_wrapper,
                self.job_config,
                save_dir=f"{rc.save_sub_dir}/kinematics/processed",
                show_reshaped=True,
            )
        # Make correlation plot
        if ac.book_cor_matrix:
            logger.info("Making correlation plot")
            kinematics.plot_correlation_matrix(
                self.model_wrapper, save_dir=rc.save_sub_dir
            )

        # Check models in different epochs, check only final if not specified
        model_path_list = ["_final"]
        if ac.check_model_epoch:
            if jc.job_type == "apply":
                model_dir = rc.load_dir
                job_name = jc.load_job_name
            else:
                model_dir = rc.save_dir
                job_name = jc.job_name
            model_path_list += train_utils.get_model_epoch_path_list(
                model_dir, tc.model_name, job_name=job_name,
            )
        max_epoch = 10
        total_models = len(model_path_list)
        epoch_interval = 1
        if total_models > max_epoch:
            epoch_interval = math.ceil(total_models / max_epoch)
            if epoch_interval > 5:
                epoch_interval = 5
        for model_num, model_path in enumerate(model_path_list):
            if model_num % epoch_interval == 0 or model_num == 1:
                logger.info(">" * 80)
                logger.info(f"Checking model:{model_path}")
                identifier = "final"
                if model_path != "_final":
                    identifier = "epoch{:02d}".format(model_num)
                    self.model_wrapper.load_model_with_path(model_path)
                # Overtrain check
                if ac.book_roc:
                    logger.info("Making roc curve plot")
                    roc.plot_multi_class_roc(
                        self.model_wrapper, self.job_config,
                    )
                if ac.book_train_test_compare:
                    logger.info("Making train/test compare plots")
                    mva_scores.plot_train_test_compare(
                        self.model_wrapper, self.job_config
                    )
                # Make feature importance check
                if ac.book_importance_study:
                    logger.info("Checking input feature importance")
                    importance.plot_feature_importance(
                        self.model_wrapper,
                        self.job_config,
                        identifier=identifier,
                        max_feature=12,
                    )
                # Extra plots (use model on non-mass-reset arrays)
                if ac.book_mva_scores_data_mc:
                    logger.info("Making data/mc scores distributions plots")
                    mva_scores.plot_mva_scores(
                        self.model_wrapper,
                        self.job_config,
                        file_name=f"mva_scores_{identifier}",
                    )
                # show kinemetics at different dnn cut
                if ac.book_cut_kine_study:
                    logger.info("Making kinematic plots with different DNN cut")
                    for dnn_cut in ac.cfg_kine_study.dnn_cut_list:
                        kinematics.plot_input_distributions(
                            self.model_wrapper,
                            self.job_config,
                            dnn_cut=dnn_cut,
                            save_dir=f"{rc.save_sub_dir}/kinematics/model_{identifier}_cut_p{dnn_cut * 100}",
                            compare_cut_sb_separated=True,
                        )
                # Make significance scan plot
                if ac.book_significance_scan:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.set_title("significance scan")
                    significance.plot_significance_scan(
                        ax,
                        self.model_wrapper,
                        ic.sig_key,
                        ic.bkg_key,
                        save_dir=rc.save_dir,
                        significance_algo=ac.cfg_significance_scan.significance_algo,
                        suffix="_" + identifier,
                    )
                    fig_save_path = (
                        rc.save_dir + "/significance_scan_" + identifier + ".png"
                    )
                    self.fig_significance_scan_path = fig_save_path
                    fig.savefig(fig_save_path)
                # TODO: 2d significance scan
                """
                if ac.book_2d_significance_scan:
                    # save original model wrapper
                    temp_model_wrapper = self.model_wrapper
                    # set up model wrapper for significance scan
                    model_class = model.get_model_class(tc.model_class)
                    scan_model_wrapper = model_class(
                        tc.model_name, ic.selected_features, rc.hypers
                    )
                    if model_path != "_final":
                        scan_model_wrapper.load_model_with_path(model_path)
                    else:
                        scan_model_wrapper.load_model(
                            rc.load_dir, tc.model_name, job_name=jc.load_job_name,
                        )
                    scan_model_wrapper.set_inputs(temp_model_wrapper.feedbox)
                    self.model_wrapper = scan_model_wrapper
                    save_dir = f"{rc.save_sub_dir}/apply/{jc.job_name}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cut_ranges_dn = ac.cfg_2d_significance_scan[
                        "significance_cut_ranges_dn"
                    ]
                    cut_ranges_up = ac.cfg_2d_significance_scan[
                        "significance_cut_ranges_up"
                    ]
                    # 2D density
                    evaluate.plot_2d_density(
                        self,
                        save_plot=True,
                        save_dir=save_dir,
                        save_file_name="2D_density_" + identifier,
                    )
                    # 2D significance scan
                    evaluate.plot_2d_significance_scan(
                        self,
                        save_plot=True,
                        save_dir=save_dir,
                        save_file_name="2D_scan_significance_" + identifier,
                        cut_ranges_dn=cut_ranges_dn,
                        cut_ranges_up=cut_ranges_up,
                        dnn_cut_min=ac.cfg_2d_significance_scan[
                            "significance_dnn_cut_min"
                        ],
                        dnn_cut_max=ac.cfg_2d_significance_scan[
                            "significance_dnn_cut_max"
                        ],
                        dnn_cut_step=ac.cfg_2d_significance_scan[
                            "significance_dnn_cut_step"
                        ],
                    )
                    # restore original model wrapper
                    self.model_wrapper = temp_model_wrapper
                """

                if ac.book_fit_npy:
                    save_region = ac.cfg_fit_npy.fit_npy_region
                    if save_region is None:
                        save_region = ic.region
                    npy_dir = (
                        f"{ac.cfg_fit_npy.npy_save_dir}/{ic.campaign}/{save_region}"
                    )
                    feedbox = self.model_wrapper.feedbox
                    keras_model = self.model_wrapper.get_model()
                    logger.info("Dumping numpy arrays for fitting.")
                    train_utils.dump_fit_npy(
                        feedbox,
                        keras_model,
                        ac.cfg_fit_npy.fit_npy_branches,
                        tc.output_bkg_node_names,
                        npy_dir=npy_dir,
                    )
                logger.info("<" * 80)

    '''
        def execute_tuning_jobs(self):
            print("#" * 80)
            print("Executing parameters scanning.")
            space = self.space
            # prepare history record file
            self.tuning_history_file = self.save_sub_dir + "/tuning_history.csv"
            if not os.path.exists(self.save_sub_dir):
                os.makedirs(self.save_sub_dir)
            history_file = open(self.tuning_history_file, "w")
            writer = csv.writer(history_file)
            writer.writerow(["loss", "auc", "iteration", "epochs", "train_time", "params"])
            history_file.close()
            # perform Bayesion tuning
            self.iteration = 0
            bayes_trials = Trials()
            best_set = fmin(
                fn=self.execute_tuning_job,
                space=space,
                algo=tpe.suggest,
                max_evals=self.max_scan_iterations,
                trials=bayes_trials,
            )
            self.best_hyper_set = best_set
            print("#" * 80)
            print("best hyperparameters set:")
            print(best_set)
            # make plots
            results = pd.read_csv(self.tuning_history_file)
            bayes_params = pd.DataFrame(
                columns=list(ast.literal_eval(results.loc[0, "params"]).keys()),
                index=list(range(len(results))),
            )
            # Add the results with each parameter a different column
            for i, params in enumerate(results["params"]):
                bayes_params.loc[i, :] = list(ast.literal_eval(params).values())

            bayes_params["iteration"] = results["iteration"]
            bayes_params["loss"] = results["loss"]
            bayes_params["auc"] = results["auc"]
            bayes_params["epochs"] = results["epochs"]
            bayes_params["train_time"] = results["train_time"]

            bayes_params.head()

            # Plot the random search distribution and the bayes search distribution
            print("#" * 80)
            print("Making scan plots")
            save_dir_distributions = self.save_sub_dir + "/hyper_distributions"
            if not os.path.exists(save_dir_distributions):
                os.makedirs(save_dir_distributions)
            save_dir_evo = self.save_sub_dir + "/hyper_evolvements"
            if not os.path.exists(save_dir_evo):
                os.makedirs(save_dir_evo)
            for hyper in list(self.space.keys()):
                # plot distributions
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.kdeplot(
                    [sample(space[hyper]) for _ in range(100000)],
                    label="Sampling Distribution",
                    ax=ax,
                )
                sns.kdeplot(bayes_params[hyper], label="Bayes Optimization")
                ax.axvline(x=best_set[hyper], color="orange", linestyle="-.")
                ax.legend(loc=1)
                ax.set_title("{} Distribution".format(hyper))
                ax.set_xlabel("{}".format(hyper))
                ax.set_ylabel("Density")
                fig.savefig(save_dir_distributions + "/" + hyper + "_distribution.png")
                # plot evolvements
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.regplot("iteration", hyper, data=bayes_params)
                ax.set(
                    xlabel="Iteration",
                    ylabel="{}".format(hyper),
                    title="{} over Search".format(hyper),
                )
                fig.savefig(save_dir_evo + "/" + hyper + "_evo.png")
                plt.close("all")
            # plot extra evolvements
            for hyper in ["loss", "auc", "epochs", "train_time"]:
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.regplot("iteration", hyper, data=bayes_params)
                ax.set(
                    xlabel="Iteration",
                    ylabel="{}".format(hyper),
                    title="{} over Search".format(hyper),
                )
                fig.savefig(save_dir_evo + "/" + hyper + "_evo.png")
                plt.close("all")

        def execute_tuning_job(self, params, loss_type="val_loss"):
        """Execute one quick DNN training for hyperparameter tuning."""
        print("+ {}".format(params))
        job_start_time = time.perf_counter()
        # Keep track of evals
        self.iteration += 1
        try:
            # set parameters
            keys = list(params.keys())
            for key in keys:
                value = params[key]
                if type(value) is float:
                    if value.is_integer():
                        value = int(value)
                setattr(self, key, value)
            # Prepare
            if self.use_early_stop:
                self.early_stop_paras = {}
                self.early_stop_paras["monitor"] = self.early_stop_monitor
                self.early_stop_paras["min_delta"] = self.early_stop_min_delta
                self.early_stop_paras["patience"] = self.early_stop_patience
                self.early_stop_paras["mode"] = self.early_stop_mode
                self.early_stop_paras[
                    "restore_best_weights"
                ] = self.early_stop_restore_best_weights
            else:
                self.early_stop_paras = {}
            hypers = {}
            hypers["layers"] = self.layers
            hypers["nodes"] = self.nodes
            hypers["output_bkg_node_names"] = self.output_bkg_node_names
            hypers["learn_rate"] = self.learn_rate
            hypers["decay"] = self.learn_rate_decay
            hypers["dropout_rate"] = self.dropout_rate
            hypers["metrics"] = self.train_metrics
            hypers["weighted_metrics"] = self.train_metrics_weighted
            hypers["use_early_stop"] = self.use_early_stop
            hypers["early_stop_paras"] = self.early_stop_paras
            hypers["momentum"] = self.momentum
            hypers["nesterov"] = self.nesterov
            self.model_wrapper = getattr(model, self.model_class)(
                self.model_name,
                self.selected_features,
                hypers,
                sig_key=self.sig_key,
                bkg_key=self.bkg_key,
                data_key=self.data_key,
            )
            # Set up training or loading model
            bkg_dict = numpy_io.load_npy_arrays(
                self.npy_path,
                self.campaign,
                self.region,
                self.channel,
                self.bkg_list,
                self.selected_features,
                cut_features=self.cut_features,
                cut_values=self.cut_values,
                cut_types=self.cut_types,
            )
            sig_dict = numpy_io.load_npy_arrays(
                self.npy_path,
                self.campaign,
                self.region,
                self.channel,
                self.sig_list,
                self.selected_features,
                cut_features=self.cut_features,
                cut_values=self.cut_values,
                cut_types=self.cut_types,
            )
            if self.apply_data:
                data_dict = numpy_io.load_npy_arrays(
                    self.npy_path,
                    self.campaign,
                    self.region,
                    self.channel,
                    self.data_list,
                    self.selected_features,
                    cut_features=self.cut_features,
                    cut_values=self.cut_values,
                    cut_types=self.cut_types,
                )
            else:
                data_dict = None
            feedbox = feed_box.Feedbox(
                sig_dict,
                bkg_dict,
                xd_dict=data_dict,
                selected_features=self.selected_features,
                apply_data=self.apply_data,
                reshape_array=self.norm_array,
                reset_mass=self.reset_feature,
                reset_mass_name=self.reset_feature_name,
                remove_negative_weight=self.rm_negative_weight_events,
                sig_weight=self.sig_sumofweight,
                bkg_weight=self.bkg_sumofweight,
                data_weight=self.data_sumofweight,
                test_rate=self.test_rate,
                rdm_seed=940926,
                model_meta=self.model_wrapper.model_meta,
                verbose=0,
            )
            self.model_wrapper.set_inputs(feedbox, apply_data=self.apply_data)
            self.model_wrapper.compile()
            final_loss = self.model_wrapper.tuning_train(
                batch_size=self.batch_size,
                epochs=self.epochs,
                val_split=self.val_split,
                sig_class_weight=self.sig_class_weight,
                bkg_class_weight=self.bkg_class_weight,
                verbose=0,
            )
            # Calculate auc
            try:
                fpr_dm, tpr_dm, _ = roc_curve(
                    self.model_wrapper.y_val,
                    self.model_wrapper.get_model().predict(self.model_wrapper.x_val),
                    sample_weight=self.model_wrapper.wt_val,
                )
                val_auc = auc(fpr_dm, tpr_dm)
            except:
                val_auc = 0
            # Get epochs
            epochs = len(self.model_wrapper.train_history.history["loss"])
        except:
            final_loss = 1000
            val_auc = 0
            epochs = 0
        # post procedure
        job_end_time = time.perf_counter()
        history_file = open(self.tuning_history_file, "a")
        writer = csv.writer(history_file)
        writer.writerow(
            [final_loss, val_auc, self.iteration, epochs, self.job_execute_time, params]
        )
        history_file.close()
        # return loss value
        loss_value = None
        loss_type = self.scan_loss_type
        if loss_type == "val_loss":
            loss_value = final_loss
        elif loss_type == "1_val_auc":
            loss_value = 1 - val_auc
        else:
            raise ValueError("Unsupported loss_type")
        print(">>> loss: {}".format(loss_value))
        return loss_value
    '''

    def get_config(self, yaml_path):
        """Retrieves configurations from yaml file."""
        cfg_path = job_utils.get_valid_cfg_path(yaml_path)
        if not pathlib.Path(cfg_path).is_file():
            logger.error("No vallid configuration file path provided.")
            raise FileNotFoundError
        yaml_dict = config_utils.load_yaml_dict(cfg_path)
        job_config_temp = config_utils.Hepy_Config(yaml_dict)
        # Check whether need to import other (default) ini file first
        if hasattr(job_config_temp, "config"):
            import_ini_path_list = job_config_temp.config.include
            if import_ini_path_list:
                for cfg_path in import_ini_path_list:
                    self.get_config(cfg_path)
                    logger.info(f"Included config: {cfg_path}")
        if self.job_config:
            self.job_config.update(yaml_dict)
        else:
            self.job_config = job_config_temp

        datestr = datetime.date.today().strftime("%Y-%m-%d")
        ic = self.job_config.input
        rc = self.job_config.run
        rc.datestr = datestr
        rc.npy_path = f"{ic.arr_path}/{ic.arr_version}/{ic.variation}"
        if ic.selected_features:
            rc.input_dim = len(ic.selected_features)
        rc.config_collected = True

    def get_scan_space(self):
        """Get hyperparameter scan space.
        TODO: need to be refactored
        """
        pass
        # space = {}
        # valid_cfg_path = job_utils.get_valid_cfg_path(self.para_scan_cfg)
        # config = config_utils.Hepy_Config(valid_cfg_path)
        ## get available scan variables:
        # for para in SCANNED_PARAS:
        #    para_pdf = self.try_parse_str(
        #        para + "_pdf", config, "scanned_para", para + "_pdf"
        #    )
        #    para_setting = self.try_parse_list(para, config, "scanned_para", para)
        #    if para_pdf is not None:
        #        dim_name = para.split("scan_")[1]
        #        if para_pdf == "choice":
        #            space[dim_name] = hp.choice(dim_name, para_setting)
        #        elif para_pdf == "uniform":
        #            space[dim_name] = hp.uniform(
        #                dim_name, para_setting[0], para_setting[1]
        #            )
        #        elif para_pdf == "quniform":
        #            space[dim_name] = hp.quniform(
        #                dim_name, para_setting[0], para_setting[1], para_setting[2]
        #            )
        #        elif para_pdf == "loguniform":
        #            space[dim_name] = hp.loguniform(
        #                dim_name, np.log(para_setting[0]), np.log(para_setting[1])
        #            )
        #        elif para_pdf == "qloguniform":
        #            space[dim_name] = hp.qloguniform(
        #                dim_name,
        #                np.log(para_setting[0]),
        #                np.log(para_setting[1]),
        #                para_setting[2],
        #            )
        #        else:
        #            raise ValueError("Unsupported scan parameter pdf type.")
        # self.space = space

    def set_model(self) -> None:
        logger.info("Setting up model")
        tc = self.job_config.train
        model_class = model.get_model_class(tc.model_class)
        self.model_wrapper = model_class(self.job_config)

    def set_model_input(self) -> None:
        logger.info("Processing inputs")
        jc = self.job_config.job
        rc = self.job_config.run
        tc = self.job_config.train
        # load model for "apply" job
        if jc.job_type == "apply":
            self.model_wrapper.load_model(
                rc.load_dir, tc.model_name, job_name=jc.load_job_name,
            )
        self.model_wrapper.set_inputs(self.job_config)

    def set_save_dir(self) -> None:
        """Sets the directory to save the outputs"""
        jc = self.job_config.job
        rc = self.job_config.run
        # Set save sub-directory for this task
        if jc.job_type == "train":
            dir_pattern = f"{jc.save_dir}/{rc.datestr}_{jc.job_name}_v{{}}"
            output_match = common_utils.get_newest_file_version(dir_pattern)
            rc.save_sub_dir = output_match["path"]
        elif jc.job_type == "apply":
            # use same directory as input "train" directory for "apply" type jobs
            dir_pattern = f"{jc.save_dir}/{rc.datestr}_{jc.load_job_name}_v{{}}"
            output_match = common_utils.get_newest_file_version(
                dir_pattern, use_existing=True
            )
            if output_match:
                rc.save_sub_dir = output_match["path"]
            else:
                logger.warning(
                    f"Can't find existing train folder matched with date {rc.datestr}, trying to search without specifying the date."
                )
                dir_pattern = f"{jc.save_dir}/*_{jc.load_job_name}_v{{}}"
                output_match = common_utils.get_newest_file_version(
                    dir_pattern, use_existing=True
                )
                if output_match:
                    rc.save_sub_dir = output_match["path"]
                else:
                    logger.error(
                        "Can't find existing train folder matched pattern, please check the settings."
                    )
