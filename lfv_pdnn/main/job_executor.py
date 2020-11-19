import ast
import csv
import datetime
import itertools
import json
import math
import os
import platform
import re
import time
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.stochastic import sample
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.common.hepy_const import *
from lfv_pdnn.common.logging_cfg import *
from lfv_pdnn.data_io import feed_box, numpy_io
from lfv_pdnn.main import job_utils
from lfv_pdnn.train import evaluate, model, train_utils
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from sklearn.metrics import auc, roc_curve


class job_executor(object):
    """Core class to execute a pdnn job based on given cfg file."""

    def __init__(self, input_path):
        """Initialize executor."""
        # set up default values for parameters
        # general
        self.job_create_time = str(datetime.datetime.now())
        self.datestr = datetime.date.today().strftime("%Y-%m-%d")
        self.cfg_path = input_path
        self.cfg_is_collected = False
        self.array_is_loaded = False
        self.scan_para_dict = {}
        # Initialize [job] section
        self.job_name = None
        self.job_type = None
        self.save_dir = None
        self.load_dir = None
        self.load_job_name = None
        self.Initialize_dir = None
        self.Initialize_job_name = None
        # Initialize [array] section
        self.arr_version = None
        self.campaign = None
        self.region = ""
        self.arr_path = None
        self.npy_path = None
        self.bkg_key = None
        self.bkg_sumofweight = None
        self.sig_key = None
        self.sig_sumofweight = None
        self.data_key = None
        self.data_sumofweight = None
        self.bkg_list = []
        self.sig_list = []
        self.data_list = []
        self.selected_features = []
        self.validation_features = []
        self.input_dim = None
        self.channel = None
        self.norm_array = True
        self.reset_feature = None
        self.reset_feature_name = None
        self.cut_features = []
        self.cut_values = []
        self.cut_types = []
        # Initialize [model] section
        self.model_name = None
        self.model_class = None
        self.layers = None
        self.nodes = None
        self.output_bkg_node_names = []
        self.dropout_rate = 0.5
        self.momentum = 0.5
        self.nesterov = True
        self.rm_negative_weight_events = True
        self.learn_rate = None
        self.learn_rate_decay = None
        self.test_rate = None
        self.batch_size = None
        self.epochs = None
        self.val_split = None
        self.sig_class_weight = None
        self.bkg_class_weight = None
        self.train_metrics = []
        self.train_metrics_weighted = []
        self.use_early_stop = None
        self.early_stop_monitor = None
        self.early_stop_min_delta = None
        self.early_stop_patience = None
        self.early_stop_mode = None
        self.early_stop_restore_best_weights = None
        self.save_model = None
        # Initialize [para_scan]
        self.perform_para_scan = None
        self.max_scan_iterations = None
        self.scan_loss_type = None
        self.para_scan_cfg = None
        # Initialize [report] section
        self.plot_bkg_list = []
        self.kine_cfg = None
        self.plot_density = True
        self.apply_data = False
        self.apply_data_range = []
        self.book_roc = False
        self.book_train_test_compare = False
        self.book_importance_study = False
        self.book_mc_data_compare = False
        self.book_kine_study = False
        self.book_cut_kine_study = False
        self.dnn_cut_list = []
        self.print_ratio_table = False
        self.book_cor_matrix = False
        self.book_significance_scan = False
        self.book_2d_significance_scan = False
        self.significance_dnn_cut_min = None
        self.significance_dnn_cut_max = None
        self.significance_dnn_cut_step = None
        self.book_fit_ntup = False
        self.fit_ntup_branches = []
        self.fit_ntup_region = None
        self.ntup_save_dir = None
        self.significance_algo = None
        self.significance_cut_ranges_dn = []
        self.significance_cut_ranges_up = []
        self.enable_model_study = None
        self.pop_plots = None
        self.print_report = None
        self.save_tb_logs = None
        self.verbose = None
        self.check_model_epoch = None
        self.cfg_is_collected = False
        # [scanned_para] section
        for para_name in SCANNED_PARAS:
            setattr(self, para_name, [])
        # timing
        self.job_execute_time = -1

    def execute_jobs(self):
        """Execute all planned jobs."""
        # Get config
        if not self.cfg_is_collected:
            self.get_config()
        self.show_configurations()
        # Set save sub-directory for this task
        if self.job_type == "train":
            dir_pattern = (
                self.save_dir + "/" + self.datestr + "_" + self.job_name + "_v{}"
            )
            output_match = common_utils.get_newest_file_version(dir_pattern)
            self.save_sub_dir = output_match["path"]
        elif self.job_type == "apply":
            # use same directory as input "train" directory for "apply" type jobs
            dir_pattern = (
                self.save_dir + "/" + self.datestr + "_" + self.load_job_name + "_v{}"
            )
            output_match = common_utils.get_newest_file_version(
                dir_pattern, use_existing=True
            )
            if output_match:
                self.save_sub_dir = output_match["path"]
            else:
                logging.warning(
                    "Can't find existing train folder matched with date"
                    + self.datestr
                    + ", trying to search without specifying the date."
                )
                dir_pattern = self.save_dir + "/*_" + self.load_job_name + "_v{}"
                output_match = common_utils.get_newest_file_version(
                    dir_pattern, use_existing=True
                )
                if output_match:
                    self.save_sub_dir = output_match["path"]
                else:
                    logging.error(
                        "Can't find existing train folder matched pattern, please check the settings."
                    )

        # Suppress inevitably ROOT warnings in python
        # ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")
        # Execute job(s)
        if (
            self.perform_para_scan is not True
        ):  # Execute single job if parameter scan is not needed
            self.execute_single_job()
        else:  # Otherwise perform scan as specified
            self.get_scan_space()
            self.execute_tuning_jobs()
            # Perform final training with best hyperparmaters
            print("#" * 80)
            print("Performing final training with best hyper parameter set.")
            keys = list(self.best_hyper_set.keys())
            for key in keys:
                value = self.best_hyper_set[key]
                value = float(value)
                if type(value) is float:
                    if value.is_integer():
                        value = int(value)
                setattr(self, key, value)
            self.execute_single_job()
            return 0

    def execute_single_job(self):
        """Execute single DNN training with given configuration."""
        # Prepare
        job_start_time = time.perf_counter()

        if self.job_type == "train":
            # model load directory should be same as save directory for "train" type jobs
            self.load_dir = self.save_dir
            self.load_job_name = self.job_name
        elif self.job_type == "apply":
            if self.load_dir == None:
                self.load_dir = self.save_dir

        if self.save_tb_logs:
            save_dir = self.save_sub_dir + "/tb_logs"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # set path to current os style, otherwise tf will report error
            self.save_tb_logs_path_subdir = os.path.normpath(save_dir)
        else:
            self.save_tb_logs_path_subdir = None
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
            validation_features=self.validation_features,
            sig_key=self.sig_key,
            bkg_key=self.bkg_key,
            data_key=self.data_key,
            save_tb_logs=self.save_tb_logs,
            tb_logs_path=self.save_tb_logs_path_subdir,
        )
        # Set up training or loading model
        bkg_dict = numpy_io.load_npy_arrays(
            self.npy_path,
            self.campaign,
            self.region,
            self.channel,
            self.bkg_list,
            self.selected_features,
            validation_features=self.validation_features,
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
            validation_features=self.validation_features,
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
                validation_features=self.validation_features,
                cut_features=self.cut_features,
                cut_values=self.cut_values,
                cut_types=self.cut_types,
            )
        else:
            data_dict = None
        if self.job_type == "apply":
            self.model_wrapper.load_model(
                self.load_dir, self.model_name, job_name=self.load_job_name,
            )
            # need to get feedbox after model loaded to get model meta
            feedbox = feed_box.Feedbox(
                sig_dict,
                bkg_dict,
                xd_dict=data_dict,
                selected_features=self.selected_features,
                validation_features=self.validation_features,
                apply_data=self.apply_data,
                reshape_array=self.norm_array,
                reset_mass=self.reset_feature,
                reset_mass_name=self.reset_feature_name,
                remove_negative_weight=self.rm_negative_weight_events,
                sig_weight=self.sig_sumofweight,
                bkg_weight=self.bkg_sumofweight,
                data_weight=self.data_sumofweight,
                test_rate=self.test_rate,
                rdm_seed=None,
                model_meta=self.model_wrapper.model_meta,
                verbose=self.verbose,
            )
            self.model_wrapper.set_inputs(feedbox, apply_data=self.apply_data)
        else:
            logging.debug(f"Selected_features quantity: {len(self.selected_features)}")
            logging.debug(f"Selected_features: {self.selected_features}")
            feedbox = feed_box.Feedbox(
                sig_dict,
                bkg_dict,
                xd_dict=data_dict,
                selected_features=self.selected_features,
                validation_features=self.validation_features,
                apply_data=self.apply_data,
                reshape_array=self.norm_array,
                reset_mass=self.reset_feature,
                reset_mass_name=self.reset_feature_name,
                remove_negative_weight=self.rm_negative_weight_events,
                sig_weight=self.sig_sumofweight,
                bkg_weight=self.bkg_sumofweight,
                data_weight=self.data_sumofweight,
                test_rate=self.test_rate,
                rdm_seed=None,
                model_meta=self.model_wrapper.model_meta,
                verbose=self.verbose,
            )
            self.model_wrapper.set_inputs(feedbox, apply_data=self.apply_data)
            self.model_wrapper.compile()
            self.model_wrapper.train(
                batch_size=self.batch_size,
                epochs=self.epochs,
                val_split=self.val_split,
                sig_class_weight=self.sig_class_weight,
                bkg_class_weight=self.bkg_class_weight,
                verbose=self.verbose,
                save_dir=self.save_sub_dir + "/models",
            )

        # Logs
        if self.save_model and (self.job_type == "train"):
            mod_save_path = self.save_sub_dir + "/models"
            model_save_name = self.model_name
            self.model_wrapper.save_model(
                save_dir=mod_save_path, file_name=model_save_name
            )
        if self.enable_model_study:
            # Performance plots
            self.fig_performance_path = None
            self.report_path = None
            # setup save parameters if reports need to be saved
            fig_save_path = None
            save_dir = self.save_sub_dir + "/reports/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # show and save according to setting
            if self.book_kine_study:
                print(">> Making input distribution plots")
                evaluate.plot_input_distributions(
                    self.model_wrapper,
                    apply_data=False,
                    style_cfg_path=self.kine_cfg,
                    save_dir=self.save_sub_dir + "/kinematics/raw",
                )
                evaluate.plot_input_distributions(
                    self.model_wrapper,
                    apply_data=False,
                    style_cfg_path=self.kine_cfg,
                    save_dir=self.save_sub_dir + "/kinematics/processed",
                    show_reshaped=True,
                )
            # Make correlation plot
            if self.book_cor_matrix:
                print(">> Making correlation plot")
                fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
                ax[0].set_title("bkg correlation")
                evaluate.plot_correlation_matrix(
                    ax[0], self.model_wrapper.get_corrcoef(), matrix_key="bkg"
                )
                ax[1].set_title("sig correlation")
                evaluate.plot_correlation_matrix(
                    ax[1], self.model_wrapper.get_corrcoef(), matrix_key="sig"
                )
                fig_save_path = save_dir + "/correlation_matrix.png"
                self.fig_correlation_matrix_path = fig_save_path
                fig.savefig(fig_save_path)

            model_path_list = ["_final"]
            if self.check_model_epoch:
                if self.job_type == "apply":
                    model_dir = self.load_dir
                    job_name = self.load_job_name
                else:
                    model_dir = self.save_dir
                    job_name = self.job_name
                model_path_list += train_utils.get_model_epoch_path_list(
                    model_dir, self.model_name, job_name=job_name,
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
                    print(">>>> Checking model:", model_path)
                    identifier = "final"
                    if model_path != "_final":
                        identifier = "epoch{:02d}".format(model_num)
                        self.model_wrapper.load_model_with_path(model_path)
                    # Make performance plots
                    print(">> Making performance plots")
                    self.fig_performance_path = fig_save_path
                    self.model_wrapper.show_performance(
                        apply_data=False,  # don't apply data in training performance
                        show_fig=self.pop_plots,
                        save_fig=True,
                        save_dir=save_dir,
                        job_type=self.job_type,
                    )
                    # Overtrain check
                    if self.book_roc:
                        print(">> Making roc curve plot")
                        evaluate.plot_multi_class_roc(
                            self.model_wrapper, save_dir=save_dir,
                        )
                    if self.book_train_test_compare:
                        print(">> Making train/test compare plots")
                        if (
                            self.job_type == "train"
                            and self.model_wrapper.feedbox.reset_mass == True
                        ):
                            evaluate.plot_overtrain_check(
                                self.model_wrapper,
                                save_dir=save_dir,
                                bins=50,
                                log=True,
                                reset_mass=True,
                            )
                        evaluate.plot_overtrain_check(
                            self.model_wrapper,
                            save_dir=save_dir,
                            bins=50,
                            log=True,
                            reset_mass=False,
                        )
                    # Make feature importance check
                    if self.book_importance_study:
                        print(">> Checking input feature importance")
                        evaluate.plot_feature_importance(
                            self.model_wrapper,
                            save_dir,
                            identifier=identifier,
                            max_feature=12,
                        )
                    # Extra plots (use model on non-mass-reset arrays)
                    if self.book_mc_data_compare:
                        print(">> Making data/mc scores distributions plots")
                        evaluate.plot_scores_separate_root(
                            self.model_wrapper,
                            self.plot_bkg_list,
                            apply_data=self.apply_data,
                            apply_data_range=self.apply_data_range,
                            plot_title="DNN scores",
                            bins=50,
                            x_range=(0, 1),
                            scale_sig=True,
                            density=self.plot_density,
                            save_plot=True,
                            save_dir=save_dir,
                            save_file_name="DNN_scores_" + identifier,
                        )
                    # show kinemetics at different dnn cut
                    if self.book_cut_kine_study:
                        print(">> Making kinematic plots with different DNN cut")
                        for dnn_cut in self.dnn_cut_list:
                            evaluate.plot_input_distributions(
                                self.model_wrapper,
                                apply_data=False,
                                figsize=(8, 6),
                                style_cfg_path=self.kine_cfg,
                                save_dir=self.save_sub_dir
                                + "/kinematics/model_{}_cut_p{}".format(
                                    identifier, dnn_cut * 100
                                ),
                                dnn_cut=dnn_cut,
                                compare_cut_sb_separated=True,
                                plot_density=False,
                                print_ratio_table=self.print_ratio_table,
                            )
                    # Make significance scan plot
                    if self.book_significance_scan:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.set_title("significance scan")
                        evaluate.plot_significance_scan(
                            ax,
                            self.model_wrapper,
                            save_dir=save_dir,
                            significance_algo=self.significance_algo,
                            suffix="_" + identifier,
                        )
                        fig_save_path = (
                            save_dir + "/significance_scan_" + identifier + ".png"
                        )
                        self.fig_significance_scan_path = fig_save_path
                        fig.savefig(fig_save_path)
                    # 2d significance scan
                    if self.book_2d_significance_scan:
                        # save original model wrapper
                        temp_model_wrapper = self.model_wrapper
                        # set up model wrapper for significance scan
                        scan_model_wrapper = getattr(model, self.model_class)(
                            self.model_name, self.selected_features, hypers,
                        )
                        if model_path != "_final":
                            scan_model_wrapper.load_model_with_path(model_path)
                        else:
                            scan_model_wrapper.load_model(
                                self.load_dir,
                                self.model_name,
                                job_name=self.load_job_name,
                            )
                        scan_model_wrapper.set_inputs(temp_model_wrapper.feedbox)
                        self.model_wrapper = scan_model_wrapper
                        save_dir = self.save_sub_dir + "/reports/"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        cut_ranges_dn = self.significance_cut_ranges_dn
                        cut_ranges_up = self.significance_cut_ranges_up
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
                            dnn_cut_min=self.significance_dnn_cut_min,
                            dnn_cut_max=self.significance_dnn_cut_max,
                            dnn_cut_step=self.significance_dnn_cut_step,
                        )
                        # restore original model wrapper
                        self.model_wrapper = temp_model_wrapper
                    if self.book_fit_ntup:
                        save_region = self.fit_ntup_region
                        if save_region is None:
                            save_region = self.region
                        ntup_dir = (
                            self.ntup_save_dir + "/" + self.campaign + "/" + save_region
                        )
                        feedbox = self.model_wrapper.feedbox
                        keras_model = self.model_wrapper.get_model()
                        # dump signal ntuple
                        train_utils.dump_fit_ntup(
                            feedbox,
                            keras_model,
                            self.fit_ntup_branches,
                            self.output_bkg_node_names,
                            ntup_dir=ntup_dir,
                        )

        if self.print_report:
            self.fig_dnn_scores_lin_path = save_dir + "/DNN_scores_lin_final.png"
            self.fig_dnn_scores_log_path = save_dir + "/DNN_scores_log_final.png"
            pdf_save_path = save_dir + "/summary_report.pdf"
            self.generate_report(pdf_save_path=pdf_save_path)
            self.report_path = pdf_save_path

        # post procedure
        plt.close("all")
        job_end_time = time.perf_counter()
        self.job_execute_time = job_end_time - job_start_time
        # return training meta data
        return self.model_wrapper.get_train_performance_meta()

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

    '''
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
        self.job_execute_time = job_end_time - job_start_time
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

    def get_config(self, path=None):
        """Retrieves configurations from ini file."""
        # Set parser
        if path is None:
            ini_path = self.cfg_path
        else:
            ini_path = path
        ini_path = job_utils.get_valid_cfg_path(ini_path)
        if not os.path.isfile(ini_path):
            raise ValueError("No vallid configuration file path provided.")
        config = ConfigParser()
        config.read(ini_path)
        # Check whether need to import other (default) ini file first
        import_ini_path = None
        import_ini_path_list = []
        try:
            import_ini_path = config.get("config", "include")
            import_ini_path_list = import_ini_path.replace(" ", "").split(",")
        except:
            pass
        if import_ini_path is not None:
            for ini_path in import_ini_path_list:
                print("Including:", ini_path)
                self.get_config(ini_path)
                print("Included:", ini_path)
        # Load [job] section
        self.try_parse_str("job_name", config, "job", "job_name")
        self.try_parse_str("job_type", config, "job", "job_type")
        self.try_parse_str("save_dir", config, "job", "save_dir")
        self.try_parse_str("load_dir", config, "job", "load_dir")
        self.try_parse_str("load_job_name", config, "job", "load_job_name")
        # Load [array] section
        self.try_parse_str("arr_version", config, "array", "arr_version")
        self.try_parse_str("campaign", config, "array", "campaign")
        self.try_parse_str("region", config, "array", "region")
        self.try_parse_str("arr_path", config, "array", "arr_path")
        self.npy_path = str(self.arr_path) + "/" + str(self.arr_version)
        self.try_parse_str("bkg_key", config, "array", "bkg_key")
        self.try_parse_float("bkg_sumofweight", config, "array", "bkg_sumofweight")
        self.try_parse_str("sig_key", config, "array", "sig_key")
        self.try_parse_float("sig_sumofweight", config, "array", "sig_sumofweight")
        self.try_parse_str("data_key", config, "array", "data_key")
        self.try_parse_float("data_sumofweight", config, "array", "data_sumofweight")
        self.try_parse_list("bkg_list", config, "array", "bkg_list")
        self.try_parse_list("sig_list", config, "array", "sig_list")
        self.try_parse_list("data_list", config, "array", "data_list")
        self.try_parse_list("selected_features", config, "array", "selected_features")
        if self.selected_features is not None:
            self.input_dim = len(self.selected_features)
        else:
            self.input_dim = None
        self.try_parse_list(
            "validation_features", config, "array", "validation_features"
        )
        self.try_parse_str("channel", config, "array", "channel")
        self.try_parse_bool("norm_array", config, "array", "norm_array")
        self.try_parse_bool("reset_feature", config, "array", "reset_feature")
        self.try_parse_str("reset_feature_name", config, "array", "reset_feature_name")
        self.try_parse_bool(
            "rm_negative_weight_events", config, "array", "rm_negative_weight_events"
        )
        self.try_parse_list("cut_features", config, "array", "cut_features")
        self.try_parse_list("cut_values", config, "array", "cut_values")
        self.try_parse_list("cut_types", config, "array", "cut_types")
        # Load [model] section
        self.try_parse_str("model_name", config, "model", "model_name")
        self.try_parse_str("model_class", config, "model", "model_class")
        self.try_parse_int("layers", config, "model", "layers")
        self.try_parse_int("nodes", config, "model", "nodes")
        self.try_parse_list(
            "output_bkg_node_names", config, "model", "output_bkg_node_names"
        )
        self.try_parse_float("dropout_rate", config, "model", "dropout_rate")
        self.try_parse_float("momentum", config, "model", "momentum")
        self.try_parse_bool("nesterov", config, "model", "nesterov")
        self.try_parse_float("learn_rate", config, "model", "learn_rate")
        self.try_parse_float("learn_rate_decay", config, "model", "learn_rate_decay")
        self.try_parse_float("test_rate", config, "model", "test_rate")
        self.try_parse_int("batch_size", config, "model", "batch_size")
        self.try_parse_int("epochs", config, "model", "epochs")
        self.try_parse_float("val_split", config, "model", "val_split")
        self.try_parse_float("sig_class_weight", config, "model", "sig_class_weight")
        self.try_parse_float("bkg_class_weight", config, "model", "bkg_class_weight")
        self.try_parse_list("train_metrics", config, "model", "train_metrics")
        self.try_parse_list(
            "train_metrics_weighted", config, "model", "train_metrics_weighted"
        )
        self.try_parse_bool("use_early_stop", config, "model", "use_early_stop")
        self.try_parse_str("early_stop_monitor", config, "model", "early_stop_monitor")
        self.try_parse_float(
            "early_stop_min_delta", config, "model", "early_stop_min_delta"
        )
        self.try_parse_int(
            "early_stop_patience", config, "model", "early_stop_patience"
        )
        self.try_parse_str("early_stop_mode", config, "model", "early_stop_mode")
        self.try_parse_bool(
            "early_stop_restore_best_weights",
            config,
            "model",
            "early_stop_restore_best_weights",
        )
        self.try_parse_bool("save_model", config, "model", "save_model")
        # Load [para_scan]
        self.try_parse_bool(
            "perform_para_scan", config, "para_scan", "perform_para_scan"
        )
        self.try_parse_int(
            "max_scan_iterations", config, "para_scan", "max_scan_iterations"
        )
        self.try_parse_str("scan_loss_type", config, "para_scan", "scan_loss_type")
        self.try_parse_str("para_scan_cfg", config, "para_scan", "para_scan_cfg")
        # Load [report] section
        self.try_parse_list("plot_bkg_list", config, "report", "plot_bkg_list")
        self.try_parse_bool("plot_density", config, "report", "plot_density")
        self.try_parse_bool("apply_data", config, "report", "apply_data")
        self.try_parse_list("apply_data_range", config, "report", "apply_data_range")
        self.try_parse_str("kine_cfg", config, "report", "kine_cfg")
        self.try_parse_bool("book_roc", config, "report", "book_roc")
        self.try_parse_bool(
            "book_train_test_compare", config, "report", "book_train_test_compare"
        )
        self.try_parse_bool(
            "book_importance_study", config, "report", "book_importance_study"
        )
        self.try_parse_bool(
            "book_mc_data_compare", config, "report", "book_mc_data_compare"
        )
        self.try_parse_bool("book_kine_study", config, "report", "book_kine_study")
        self.try_parse_bool(
            "book_cut_kine_study", config, "report", "book_cut_kine_study"
        )
        self.try_parse_list("dnn_cut_list", config, "report", "dnn_cut_list")
        self.try_parse_bool("print_ratio_table", config, "report", "print_ratio_table")
        self.try_parse_bool("book_cor_matrix", config, "report", "book_cor_matrix")
        self.try_parse_bool(
            "book_significance_scan", config, "report", "book_significance_scan"
        )
        self.try_parse_bool(
            "book_2d_significance_scan", config, "report", "book_2d_significance_scan"
        )
        self.try_parse_float(
            "significance_dnn_cut_min", config, "report", "significance_dnn_cut_min"
        )
        self.try_parse_float(
            "significance_dnn_cut_max", config, "report", "significance_dnn_cut_max"
        )
        self.try_parse_float(
            "significance_dnn_cut_step", config, "report", "significance_dnn_cut_step"
        )
        self.try_parse_bool("book_fit_ntup", config, "report", "book_fit_ntup")
        self.try_parse_list("fit_ntup_branches", config, "report", "fit_ntup_branches")
        self.try_parse_str("fit_ntup_region", config, "report", "fit_ntup_region")
        self.try_parse_str("ntup_save_dir", config, "report", "ntup_save_dir")
        self.try_parse_str("significance_algo", config, "report", "significance_algo")
        self.try_parse_list(
            "significance_cut_ranges_dn", config, "report", "significance_cut_ranges_dn"
        )
        self.try_parse_list(
            "significance_cut_ranges_up", config, "report", "significance_cut_ranges_up"
        )
        self.try_parse_bool(
            "enable_model_study", config, "report", "enable_model_study"
        )
        self.try_parse_bool("pop_plots", config, "report", "pop_plots")
        self.try_parse_bool("print_report", config, "report", "print_report")
        self.try_parse_bool("save_tb_logs", config, "report", "save_tb_logs")
        self.try_parse_int("verbose", config, "report", "verbose")
        self.try_parse_bool("check_model_epoch", config, "report", "check_model_epoch")

        if self.perform_para_scan:
            pass
            # self.get_config_scan() # TODO this method will be replaced in the future

        self.cfg_is_collected = True

    def get_config_scan(self):
        """Load parameters scan configuration file."""
        config = ConfigParser()
        valid_cfg_path = job_utils.get_valid_cfg_path(self.para_scan_cfg)
        config.read(valid_cfg_path)
        # get available scan variables:
        for para in SCANNED_PARAS:
            self.try_parse_list(para, config, "scanned_para", para)

    def get_scan_para_list(self):
        """Gets a list of dictionary to reset parameters for new scan job."""
        used_para_lists = []
        used_para_names = []
        for para_name in SCANNED_PARAS:
            scanned_values = getattr(self, para_name)
            if len(scanned_values) > 0:
                used_para_names.append(para_name)
                used_para_lists.append(scanned_values)
        combs = list(itertools.product(*used_para_lists))
        # Get scan_list
        scan_list = []
        for comb in combs:
            scan_dict_single = {}
            for (key, value) in zip(used_para_names, comb):
                scan_dict_single[key] = value
            scan_list.append(scan_dict_single)
        if len(scan_list) < 1:
            raise ValueError("Empty scan parameter list, please check .ini file.")
        # Get corresponding scan_list identifiers
        used_para_ids = [list(range(len(para))) for para in used_para_lists]
        scan_list_id = list(itertools.product(*used_para_ids))
        # Summary
        print("Scan parameters list loaded.")
        print("Scaned parameters are:")
        for (para_name, para_list) in zip(used_para_names, used_para_lists):
            print("*", para_name, ":", para_list)
        print("Total combinations/scans:", len(scan_list))
        return scan_list, scan_list_id

    def get_scan_space(self):
        """Get hyperparameter scan space."""
        space = {}
        config = ConfigParser()
        valid_cfg_path = job_utils.get_valid_cfg_path(self.para_scan_cfg)
        config.read(valid_cfg_path)
        # get available scan variables:
        for para in SCANNED_PARAS:
            para_pdf = self.try_parse_str(
                para + "_pdf", config, "scanned_para", para + "_pdf"
            )
            para_setting = self.try_parse_list(para, config, "scanned_para", para)
            if para_pdf is not None:
                dim_name = para.split("scan_")[1]
                if para_pdf == "choice":
                    space[dim_name] = hp.choice(dim_name, para_setting)
                elif para_pdf == "uniform":
                    space[dim_name] = hp.uniform(
                        dim_name, para_setting[0], para_setting[1]
                    )
                elif para_pdf == "quniform":
                    space[dim_name] = hp.quniform(
                        dim_name, para_setting[0], para_setting[1], para_setting[2]
                    )
                elif para_pdf == "loguniform":
                    space[dim_name] = hp.loguniform(
                        dim_name, np.log(para_setting[0]), np.log(para_setting[1])
                    )
                elif para_pdf == "qloguniform":
                    space[dim_name] = hp.qloguniform(
                        dim_name,
                        np.log(para_setting[0]),
                        np.log(para_setting[1]),
                        para_setting[2],
                    )
                else:
                    raise ValueError("Unsupported scan parameter pdf type.")
        self.space = space

    def generate_report(self, pdf_save_path=None):
        """Generate a brief report to show how is the model."""
        # Initalize
        if pdf_save_path is None:
            pdf_save_path = (
                self.save_sub_dir
                + "/"
                + self.job_name
                + "_report_"
                + self.datestr
                + ".pdf"
            )
        doc = SimpleDocTemplate(
            pdf_save_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Justify", alignment=TA_JUSTIFY))
        reports = []
        # Reports
        # head
        ptext = "JOB NAME: " + self.job_name
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "JOB TYPE: " + self.job_type
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "DATE TIME: " + self.job_create_time
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "JOB EXECUTE TIME (min): " + str(self.job_execute_time / 60.0)
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        # machine info
        ptext = "MACHINE INFO:"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "-" * 80
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "machine:" + platform.machine()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "version:" + platform.version()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "platform:" + platform.platform()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "system:" + platform.system()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "processor:" + platform.processor()
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        # parameters
        ptext = "KEY PARAMETERS:"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "-" * 80
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        ptext = "config file location: " + re.sub(r"[\s+]", "", self.cfg_path)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "[array]"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "array version                    : " + self.arr_version
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "array campaign                   : " + self.campaign
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "channel                          : " + self.channel
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "background list                  : " + str(self.bkg_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "signal list                      : " + str(self.sig_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data list                        : " + str(self.data_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features                : " + str(self.selected_features)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "validation features              : " + str(self.validation_features)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "reset feature                    : " + str(self.reset_feature)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "reset feature name             : " + str(self.reset_feature_name)
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        ptext = "bkg arrays path                  : " + self.arr_path
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg arrays used                  : " + self.bkg_key
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg total weight set             : " + str(self.bkg_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig arrays used                  : " + self.sig_key
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig total weight                 : " + str(self.sig_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data arrays used                 : " + str(self.data_key)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data total weight                : " + str(self.data_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data normalize input variables   : " + str(self.norm_array)
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        ptext = "[model]"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "name                             : " + self.model_name
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "class                            : " + self.model_class
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "layers                           : " + str(self.layers)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "nodes                            : " + str(self.nodes)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "dropout_rate                     : " + str(self.dropout_rate)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "momentum                         : " + str(self.momentum)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "nesterov                         : " + str(self.nesterov)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "rm_negative_weight_events        : " + str(
            self.rm_negative_weight_events
        )
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "learn rate                       : " + str(self.learn_rate)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "learn decay                      : " + str(self.learn_rate_decay)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "test ratio                       : " + str(self.test_rate)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "validation split                 : " + str(self.val_split)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "batch size                       : " + str(self.batch_size)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "epochs                           : " + str(self.epochs)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "signal class weight              : " + str(self.sig_class_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "background class weight          : " + str(self.bkg_class_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "use early stop                   : " + str(self.use_early_stop)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop monitor               : " + str(self.early_stop_monitor)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop min_delta             : " + str(self.early_stop_min_delta)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop patience              : " + str(self.early_stop_patience)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop mode                  : " + str(self.early_stop_mode)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop restore_best_weights  : " + str(
            self.early_stop_restore_best_weights
        )
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "model saved                      : " + str(self.save_model)
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        ptext = "[TensorBoard logs]"
        reports.append(Paragraph(ptext, styles["Justify"]))
        # Evaluation results
        reports.append(PageBreak())
        ptext = "PERFORMANCE PLOTS:"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "-" * 80
        reports.append(Paragraph(ptext, styles["Justify"]))
        # fig = self.fig_performance_path
        # im = Image(fig, 6.4 * inch, 7.2 * inch)
        # reports.append(im)
        # show total weights of sig/bkg/data
        # ptext = "sig total weight  : " + str(self.model_wrapper.total_weight_sig)
        # reports.append(Paragraph(ptext, styles["Justify"]))
        # ptext = "bkg total weight  : " + str(self.model_wrapper.total_weight_bkg)
        # reports.append(Paragraph(ptext, styles["Justify"]))
        # ptext = "data total weight : " + str(self.model_wrapper.total_weight_data)
        # reports.append(Paragraph(ptext, styles["Justify"]))
        # dnn scores
        # fig = self.fig_dnn_scores_lin_path
        # im1 = Image(fig, 3.2 * inch, 2.4 * inch)
        # fig = self.fig_dnn_scores_log_path
        # im2 = Image(fig, 3.2 * inch, 2.4 * inch)
        # reports.append(Table([[im1, im2]]))
        # significance scan and feature importance
        # if self.book_importance_study:
        #    fig = self.fig_feature_importance_path
        #    im = Image(fig, 3.2 * inch, 2.4 * inch)
        #    reports.append(im)
        if self.book_significance_scan:
            fig = self.fig_significance_scan_path
            im = Image(fig, 3.2 * inch, 2.4 * inch)
            reports.append(im)
        # correlation matrix
        if self.book_cor_matrix:
            fig = self.fig_correlation_matrix_path
            im = Image(fig, 6.4 * inch, 3.2 * inch)
            reports.append(im)
        # build/save
        doc.build(reports)

    def show_configurations(self):
        print("#" * 80)
        print("Configurations of this job are listed below:")
        print("#" * 80)
        # [job] section
        print("[job]")
        print("> job_name:", self.job_name)
        print("> job_type:", self.job_type)
        print("> save_dir:", self.save_dir)
        print("> load_dir:", self.load_dir)
        print("> load_job_name:", self.load_job_name)
        # [array] section
        print("[array]")
        print("> arr_version:", self.arr_version)
        print("> campaign:", self.campaign)
        print("> region:", self.region)
        print("> arr_path:", self.arr_path)
        print("> bkg_key:", self.bkg_key)
        print("> bkg_sumofweight:", self.bkg_sumofweight)
        print("> sig_key:", self.sig_key)
        print("> sig_sumofweight:", self.sig_sumofweight)
        print("> data_key:", self.data_key)
        print("> data_sumofweight:", self.data_sumofweight)
        print("> bkg_list:", self.bkg_list)
        print("> sig_list:", self.sig_list)
        print("> data_list:", self.data_list)
        print("> selected_features:", self.selected_features)
        print("> validation_features:", self.validation_features)
        print("> channel:", self.channel)
        print("> norm_array:", self.norm_array)
        print("> reset_feature:", self.reset_feature)
        print("> reset_feature_name:", self.reset_feature_name)
        print("> rm_negative_weight_events:", self.rm_negative_weight_events)
        print("> cut_features:", self.cut_features)
        print("> cut_values:", self.cut_values)
        print("> cut_types:", self.cut_types)
        # [model] section
        print("[model]")
        print("> model_name:", self.model_name)
        print("> model_class:", self.model_class)
        print("> layers:", self.layers)
        print("> nodes:", self.nodes)
        print("> output_bkg_node_names:", self.output_bkg_node_names)
        print("> dropout_rate:", self.dropout_rate)
        print("> momentum:", self.momentum)
        print("> nesterov:", self.nesterov)
        print("> learn_rate:", self.learn_rate)
        print("> learn_rate_decay:", self.learn_rate_decay)
        print("> test_rate:", self.test_rate)
        print("> batch_size:", self.batch_size)
        print("> epochs:", self.epochs)
        print("> val_split:", self.val_split)
        print("> sig_class_weight:", self.sig_class_weight)
        print("> bkg_class_weight:", self.bkg_class_weight)
        print("> train_metrics:", self.train_metrics)
        print("> train_metrics_weighted:", self.train_metrics_weighted)
        print("> use_early_stop:", self.use_early_stop)
        print("> early_stop_monitor:", self.early_stop_monitor)
        print("> early_stop_min_delta:", self.early_stop_min_delta)
        print("> early_stop_patience:", self.early_stop_patience)
        print("> early_stop_mode:", self.early_stop_mode)
        print(
            "> early_stop_restore_best_weights:", self.early_stop_restore_best_weights
        )
        print("> save_model:", self.save_model)
        # [para_scan]
        print("[para_scan]")
        print("> perform_para_scan:", self.perform_para_scan)
        print("> max_scan_iterations:", self.max_scan_iterations)
        print("> scan_loss_type:", self.scan_loss_type)
        print("> para_scan_cfg:", self.para_scan_cfg)
        # [report] section
        print("[report]")
        print("> plot_bkg_list:", self.plot_bkg_list)
        print("> plot_density:", self.plot_density)
        print("> apply_data:", self.apply_data)
        print("> apply_data_range:", self.apply_data_range)
        print("> kine_cfg:", self.kine_cfg)
        print("> book_roc:", self.book_roc)
        print("> book_train_test_compare:", self.book_train_test_compare)
        print("> book_importance_study:", self.book_importance_study)
        print("> book_mc_data_compare:", self.book_mc_data_compare)
        print("> book_kine_study:", self.book_kine_study)
        print("> book_cut_kine_study:", self.book_cut_kine_study)
        print("> dnn_cut_list:", self.dnn_cut_list)
        print("> print_ratio_table:", self.print_ratio_table)
        print("> book_cor_matrix:", self.book_cor_matrix)
        print("> book_significance_scan:", self.book_significance_scan)
        print("> book_2d_significance_scan:", self.book_2d_significance_scan)
        print("> significance_dnn_cut_min:", self.significance_dnn_cut_min)
        print("> significance_dnn_cut_max:", self.significance_dnn_cut_max)
        print("> significance_dnn_cut_step:", self.significance_dnn_cut_step)
        print("> book_fit_ntup:", self.book_fit_ntup)
        print("> fit_ntup_branches:", self.fit_ntup_branches)
        print("> fit_ntup_region:", self.fit_ntup_region)
        print("> ntup_save_dir:", self.ntup_save_dir)
        print("> significance_algo:", self.significance_algo)
        print("> significance_cut_ranges_dn:", self.significance_cut_ranges_dn)
        print("> significance_cut_ranges_up:", self.significance_cut_ranges_up)
        print("> enable_model_study:", self.enable_model_study)
        print("> print_report:", self.print_report)
        print("> save_tb_logs:", self.save_tb_logs)
        print("> verbose:", self.verbose)
        print("> check_model_epoch:", self.check_model_epoch)
        print("#" * 80)

    def set_para(self, parsed_val, data_type, config_parser, section, val_name):
        """Sets parameters for training manually."""
        if data_type == "bool":
            pass
        elif data_type == "float":
            float_temp = config_parser.getfloat(section, val_name)
            setattr(self, parsed_val, float_temp)

    def try_parse_bool(self, parsed_val, config_parse, section, val_name):
        try:
            value = config_parse.getboolean(section, val_name)
            setattr(self, parsed_val, value)
            return value
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)
            return None

    def try_parse_float(self, parsed_val, config_parser, section, val_name):
        try:
            value = config_parser.getfloat(section, val_name)
            setattr(self, parsed_val, value)
            return value
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)
            return None

    def try_parse_int(self, parsed_val, config_parser, section, val_name):
        try:
            value = config_parser.getint(section, val_name)
            setattr(self, parsed_val, value)
            return value
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)
            return None

    def try_parse_str(self, parsed_val, config_parser, section, val_name):
        try:
            value = config_parser.get(section, val_name)
            setattr(self, parsed_val, value)
            return value
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)
            return None

    def try_parse_list(self, parsed_val, config_parser, section, val_name):
        try:
            value = json.loads(config_parser.get(section, val_name))
            if not isinstance(value, list):
                value = [value]
            setattr(self, parsed_val, value)
            return value
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)
            return None
