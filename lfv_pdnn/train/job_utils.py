import datetime
import itertools
import json
import os
import platform
import re
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A3, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import ROOT
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.data_io import get_arrays
from lfv_pdnn.train import model, train_utils

SCANNED_PARAS = [
    'scan_learn_rate', 'scan_learn_rate_decay', 'scan_batch_size',
    'scan_sig_sumofweight', 'scan_bkg_sumofweight', 'scan_sig_class_weight',
    'scan_bkg_class_weight', 'scan_sig_key', 'scan_bkg_key', 'scan_channel',
    'scan_early_stop_patience'
]
# possible main directory names, in docker it's "work", otherwise it's "pdnn-lfv"
MAIN_DIR_NAMES = ["pdnn-lfv", "work"]


class job_executor(object):
    """Core class to execute a pdnn job based on given cfg file."""
    def __init__(self, input_path):
        """Initialize executor."""
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
        self.bkg_dict_path = None
        self.bkg_key = None
        self.bkg_sumofweight = None
        self.bkg_rm_neg_weight = False
        self.sig_dict_path = None
        self.sig_key = None
        self.sig_sumofweight = None
        self.sig_rm_neg_weight = False
        self.data_dict_path = None
        self.data_key = None
        self.data_sumofweight = None
        self.data_rm_neg_weight = False
        self.bkg_list = []
        self.sig_list = []
        self.data_list = []
        self.selected_features = []
        self.input_dim = None
        self.channel = None
        self.norm_array = True
        self.reset_feature = None
        self.reset_feature_name = None
        # Initialize [model] section
        self.model_name = None
        self.model_class = None
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
        self.para_scan_cfg = None
        self.scan_id = None
        # Initialize [report] section
        self.plot_bkg_list = []
        self.kine_cfg = None
        self.plot_density = True
        self.apply_data = False
        self.show_report = None
        self.save_pdf_report = None
        self.save_tb_logs = None
        self.verbose = None
        self.cfg_is_collected = False
        # [scanned_para] section
        for para_name in SCANNED_PARAS:
            setattr(self, para_name, [])

    def execute_jobs(self):
        """Execute all planned jobs."""
        # Get config
        if not self.cfg_is_collected:
            self.get_config()
        # Set save sub-directory for this task
        dir_pattern = self.save_dir + '/' + self.datestr + '_' + self.job_name \
                      + "_v{}"
        self.save_sub_dir = common_utils.get_newest_file_version(
            dir_pattern)['path']
        # Suppress inevitably ROOT warnings in python
        ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")
        # Execute job(s)
        if self.perform_para_scan is not True:  # Execute single job if parameter scan is not needed
            self.execute_single_job()
        else:  # Otherwise perform scan as specified
            print('#' * 80)
            print("Executing parameters scanning.")
            scan_list, scan_list_id = self.get_scan_para_list()
            scanned_var_names = list(scan_list[0].keys())
            scan_meta_report = [
                scanned_var_names + [
                    "asimov_ori", "asimov_best", "asimov_cut", "auc_tr",
                    "auc_te", "auc_tr_ori", "auc_te_ori"
                ]
            ]
            # asimov_ori: original significance
            # asimov_best: best significance with DNN applied
            # asimov_cut: DNN cut for best significance
            # auc_tr: auc (area under roc curve) value with train dataset
            # auc_te: auc value with test dataset
            # auc_tr_ori: auc value with train dataset
            # auc_te_ori: auc value with test dataset
            for job_num, (scan_set, scan_set_id) in enumerate(
                    zip(scan_list, scan_list_id)):
                # Train with current scan parameter set
                print('*' * 80)
                print("Scanning parameter set {}/{}:".format(
                    job_num + 1, len(scan_list)))
                common_utils.display_dict(scan_set)
                keys = list(scan_set.keys())
                scan_id = "scan"
                scanned_var_value_list = []
                for num, key in enumerate(keys):
                    scanned_var_name = key.split("scan_")[1]
                    scanned_var_value_list.append(scan_set[key])
                    setattr(self, scanned_var_name, scan_set[key])
                    scan_id = scan_id + "--" + scanned_var_name + "_" \
                      + str(scan_set_id[num])
                print("scan id:", scan_id)
                setattr(self, "scan_id", scan_id)
                meta_data_single = self.execute_single_job(
                )  # performs sigle train
                report_keys = [
                    "original_significance", "max_significance",
                    "max_significance_threshould", "auc_train", "auc_test",
                    "auc_train_original", "auc_test_original"
                ]
                # Add scan meta data contents
                single_scan_meta_data = scanned_var_value_list
                for report_key in report_keys:
                    report_value = meta_data_single[report_key]
                    if isinstance(report_value, float):
                        report_value = format(report_value, ".6f")
                    single_scan_meta_data.append(report_value)
                scan_meta_report.append(single_scan_meta_data)
            make_table(scan_meta_report,
                       self.save_sub_dir + "/scan_meta_report.pdf",
                       num_para=len(scanned_var_names))

    def execute_single_job(self):
        """Execute single DNN training with given configuration."""

        # Prepare
        if not self.array_is_loaded:
            self.load_arrays()
        xs = array_utils.modify_array(
            self.sig_dict[self.sig_key],
            select_channel=True,
            remove_negative_weight=self.sig_rm_neg_weight)
        xb = array_utils.modify_array(
            self.bkg_dict[self.bkg_key],
            select_channel=True,
            remove_negative_weight=self.bkg_rm_neg_weight)
        xd = array_utils.modify_array(
            self.data_dict[self.data_key],
            select_channel=True,
            remove_negative_weight=self.data_rm_neg_weight)
        for key in self.plot_bkg_list:
            self.plot_bkg_dict[key] = array_utils.modify_array(
                self.plot_bkg_dict[key],
                select_channel=True,
                remove_negative_weight=self.bkg_rm_neg_weight)
        if self.save_tb_logs:
            save_dir = self.save_sub_dir + '/tb_logs'
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
                "restore_best_weights"] = self.early_stop_restore_best_weights
        else:
            self.early_stop_paras = {}
        self.model = getattr(model, self.model_class)(
            self.model_name,
            self.input_dim,
            learn_rate=self.learn_rate,
            decay=self.learn_rate_decay,
            dropout_rate=0.5,
            metrics=self.train_metrics,
            weighted_metrics=self.train_metrics_weighted,
            selected_features=self.selected_features,
            save_tb_logs=self.save_tb_logs,
            tb_logs_path=self.save_tb_logs_path_subdir,
            use_early_stop=self.use_early_stop,
            early_stop_paras=self.early_stop_paras)
        # Set up training or loading model
        self.model.prepare_array(xs,
                                 xb,
                                 xd=xd,
                                 apply_data=self.apply_data,
                                 norm_array=self.norm_array,
                                 reset_mass=self.reset_feature,
                                 reset_mass_name=self.reset_feature_name,
                                 sig_weight=self.sig_sumofweight,
                                 bkg_weight=self.bkg_sumofweight,
                                 data_weight=self.data_sumofweight,
                                 test_rate=self.test_rate,
                                 verbose=self.verbose)
        if self.job_type == "apply":
            self.model.load_model(self.load_dir,
                                  self.model_name,
                                  job_name=self.load_job_name,
                                  model_class=self.model_class)
        else:
            self.model.compile()
            self.model.train(batch_size=self.batch_size,
                             epochs=self.epochs,
                             val_split=self.val_split,
                             sig_class_weight=self.sig_class_weight,
                             bkg_class_weight=self.bkg_class_weight,
                             verbose=self.verbose)
        # Logs
        if self.show_report or self.save_pdf_report:
            # Performance plots
            self.fig_performance_path = None
            self.fig_non_mass_reset_path = None
            self.report_path = None
            # setup save parameters if reports need to be saved
            fig_save_path = None
            save_dir = None
            if self.perform_para_scan:
                save_dir = self.save_sub_dir + "/reports/scan_" + self.scan_id
            else:
                save_dir = self.save_sub_dir + "/reports/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # show and save according to setting
            self.model.show_input_distributions(apply_data=False,
                                                figsize=(8, 6),
                                                style_cfg_path=self.kine_cfg,
                                                save_fig=True,
                                                save_dir=self.save_sub_dir +
                                                "/kinematics")
            # Make correlation plot
            fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
            ax[0].set_title("bkg correlation")
            self.model.plot_correlation_matrix(ax[0], matrix_key="bkg")
            ax[1].set_title("sig correlation")
            self.model.plot_correlation_matrix(ax[1], matrix_key="sig")
            if self.save_pdf_report:
                fig_save_path = save_dir + '/correlation_matrix.png'
                self.fig_correlation_matrix_path = fig_save_path
                fig.savefig(fig_save_path)
            # Make performance plots
            fig_save_path = save_dir + '/performance.png'
            self.fig_performance_path = fig_save_path
            self.model.show_performance(apply_data=self.apply_data,
                                        show_fig=self.show_report,
                                        save_fig=self.save_pdf_report,
                                        save_path=fig_save_path,
                                        job_type=self.job_type)
            # Make significance scan plot
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.set_title("significance scan")
            self.model.plot_significance_scan(ax)
            if self.save_pdf_report:
                fig_save_path = save_dir + '/significance_scan.png'
                self.fig_significance_scan_path = fig_save_path
                fig.savefig(fig_save_path)
            # Extra plots (use model on non-mass-reset arrays)
            fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
            self.model.plot_scores_separate(ax[0],
                                            self.plot_bkg_dict,
                                            self.plot_bkg_list,
                                            self.selected_features,
                                            apply_data=self.apply_data,
                                            plot_title="DNN scores (lin)",
                                            bins=50,
                                            range=(-0.25, 1.25),
                                            density=self.plot_density,
                                            log=False)
            self.model.plot_scores_separate(ax[1],
                                            self.plot_bkg_dict,
                                            self.plot_bkg_list,
                                            self.selected_features,
                                            apply_data=self.apply_data,
                                            plot_title="DNN scores (log)",
                                            bins=50,
                                            range=(-0.25, 1.25),
                                            density=self.plot_density,
                                            log=True)
            fig.tight_layout()
            self.model.plot_scores_separate_root(
                self.plot_bkg_dict,
                self.plot_bkg_list,
                self.selected_features,
                apply_data=self.apply_data,
                plot_title="DNN scores (lin)",
                bins=25,
                range=(0, 1),
                scale_sig=True,
                density=self.plot_density,
                log_scale=False,
                save_plot=True,
                save_dir=save_dir,
                save_file_name="DNN_scores_lin")
            self.model.plot_scores_separate_root(
                self.plot_bkg_dict,
                self.plot_bkg_list,
                self.selected_features,
                apply_data=self.apply_data,
                plot_title="DNN scores (log)",
                bins=25,
                range=(0, 1),
                scale_sig=True,
                density=self.plot_density,
                log_scale=True,
                save_plot=True,
                save_dir=save_dir,
                save_file_name="DNN_scores_log")
            if self.save_pdf_report:
                fig_save_path = save_dir + '/non-mass-reset_plots.png'
                fig.savefig(fig_save_path)
                self.fig_non_mass_reset_path = fig_save_path
                self.fig_dnn_scores_lin_path = save_dir + "/DNN_scores_lin.png"
                self.fig_dnn_scores_log_path = save_dir + "/DNN_scores_log.png"
                pdf_save_path = save_dir + '/summary_report.pdf'
                self.generate_report(pdf_save_path=pdf_save_path)
                self.report_path = pdf_save_path
        if self.save_model and (self.job_type == "train"):
            mod_save_path = self.save_sub_dir + "/models"
            model_save_name = self.model_name
            if self.perform_para_scan:
                model_save_name = self.model_name + "_" + self.scan_id
            self.model.save_model(save_dir=mod_save_path,
                                  file_name=model_save_name)
        # return training meta data
        return self.model.get_train_performance_meta()

    def get_config(self, path=None):
        """Retrieves configurations from ini file."""
        # Set parser
        if path is None:
            ini_path = self.cfg_path
        else:
            ini_path = path
        ini_path = get_valid_cfg_path(ini_path)
        if not os.path.isfile(ini_path):
            raise ValueError("No vallid configuration file path provided.")
        config = ConfigParser()
        config.read(ini_path)
        # Check whether need to import other (default) ini file first
        default_ini_path = None
        try:
            default_ini_path = config.get('config', 'include')
        except:
            pass
        if default_ini_path is not None:
            print("Including:", default_ini_path)
            self.get_config(default_ini_path)
        # Load [job] section
        self.try_parse_str('job_name', config, 'job', 'job_name')
        self.try_parse_str('job_type', config, 'job', 'job_type')
        self.try_parse_str('save_dir', config, 'job', 'save_dir')
        self.try_parse_str('load_dir', config, 'job', 'load_dir')
        self.try_parse_str('load_job_name', config, 'job', 'load_job_name')
        # Load [array] section
        self.try_parse_str('arr_version', config, 'array', 'arr_version')
        self.try_parse_str('campaign', config, 'array', 'campaign')
        self.try_parse_str('bkg_dict_path', config, 'array', 'bkg_dict_path')
        self.try_parse_str('bkg_key', config, 'array', 'bkg_key')
        self.try_parse_float('bkg_sumofweight', config, 'array',
                             'bkg_sumofweight')
        self.try_parse_bool('bkg_rm_neg_weight', config, 'array',
                            'bkg_rm_neg_weight')
        self.try_parse_str('sig_dict_path', config, 'array', 'sig_dict_path')
        self.try_parse_str('sig_key', config, 'array', 'sig_key')
        self.try_parse_float('sig_sumofweight', config, 'array',
                             'sig_sumofweight')
        self.try_parse_bool('sig_rm_neg_weight', config, 'array',
                            'sig_rm_neg_weight')
        self.try_parse_str('data_dict_path', config, 'array', 'data_dict_path')
        self.try_parse_str('data_key', config, 'array', 'data_key')
        self.try_parse_float('data_sumofweight', config, 'array',
                             'data_sumofweight')
        self.try_parse_bool('data_rm_neg_weight', config, 'array',
                            'data_rm_neg_weight')
        self.try_parse_list('bkg_list', config, 'array', 'bkg_list')
        self.try_parse_list('sig_list', config, 'array', 'sig_list')
        self.try_parse_list('data_list', config, 'array', 'data_list')
        self.try_parse_list('selected_features', config, 'array',
                            'selected_features')
        if self.selected_features is not None:
            self.input_dim = len(self.selected_features)
        else:
            self.input_dim = None
        self.try_parse_str('channel', config, 'array', 'channel')
        self.try_parse_bool('norm_array', config, 'array', 'norm_array')
        self.try_parse_bool('reset_feature', config, 'array', 'reset_feature')
        self.try_parse_str('reset_feature_name', config, 'array',
                           'reset_feature_name')
        # Load [model] section
        self.try_parse_str('model_name', config, 'model', 'model_name')
        self.try_parse_str('model_class', config, 'model', 'model_class')
        self.try_parse_float('learn_rate', config, 'model', 'learn_rate')
        self.try_parse_float('learn_rate_decay', config, 'model',
                             'learn_rate_decay')
        self.try_parse_float('test_rate', config, 'model', 'test_rate')
        self.try_parse_int('batch_size', config, 'model', 'batch_size')
        self.try_parse_int('epochs', config, 'model', 'epochs')
        self.try_parse_float('val_split', config, 'model', 'val_split')
        self.try_parse_float('sig_class_weight', config, 'model',
                             'sig_class_weight')
        self.try_parse_float('bkg_class_weight', config, 'model',
                             'bkg_class_weight')
        self.try_parse_list('train_metrics', config, 'model', 'train_metrics')
        self.try_parse_list('train_metrics_weighted', config, 'model',
                            'train_metrics_weighted')
        self.try_parse_bool('use_early_stop', config, 'model',
                            'use_early_stop')
        self.try_parse_str('early_stop_monitor', config, 'model',
                           'early_stop_monitor')
        self.try_parse_float('early_stop_min_delta', config, 'model',
                             'early_stop_min_delta')
        self.try_parse_int('early_stop_patience', config, 'model',
                           'early_stop_patience')
        self.try_parse_str('early_stop_mode', config, 'model',
                           'early_stop_mode')
        self.try_parse_bool('early_stop_restore_best_weights', config, 'model',
                            'early_stop_restore_best_weights')
        self.try_parse_bool('save_model', config, 'model', 'save_model')
        # Load [para_scan]
        self.try_parse_bool('perform_para_scan', config, 'para_scan',
                            'perform_para_scan')
        self.try_parse_str('para_scan_cfg', config, 'para_scan',
                           'para_scan_cfg')
        # Load [report] section
        self.try_parse_list('plot_bkg_list', config, 'report', 'plot_bkg_list')
        self.try_parse_bool('plot_density', config, 'report', 'plot_density')
        self.try_parse_bool('apply_data', config, 'report', 'apply_data')
        self.try_parse_str('kine_cfg', config, 'report', 'kine_cfg')
        self.try_parse_bool('show_report', config, 'report', 'show_report')
        self.try_parse_bool('save_pdf_report', config, 'report',
                            'save_pdf_report')
        self.try_parse_bool('save_tb_logs', config, 'report', 'save_tb_logs')
        self.try_parse_int('verbose', config, 'report', 'verbose')

        if self.perform_para_scan:
            self.get_config_scan()

        self.cfg_is_collected = True

    def get_config_scan(self):
        """Load parameters scan configuration file."""
        config = ConfigParser()
        valid_cfg_path = get_valid_cfg_path(self.para_scan_cfg)
        config.read(valid_cfg_path)
        # get available scan variables:
        for para in SCANNED_PARAS:
            self.try_parse_list(para, config, 'scanned_para', para)

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
            raise ValueError(
                "Empty scan parameter list, please check .ini file.")
        # Get corresponding scan_list identifiers
        used_para_ids = [list(range(len(para))) for para in used_para_lists]
        scan_list_id = list(itertools.product(*used_para_ids))
        # Summary
        print("Scan parameters list loaded.")
        print("Scaned parameters are:")
        for (para_name, para_list) in zip(used_para_names, used_para_lists):
            print('*', para_name, ':', para_list)
        print("Total combinations/scans:", len(scan_list))
        return scan_list, scan_list_id

    def generate_report(self, pdf_save_path=None):
        """Generate a brief report to show how is the model."""
        # Initalize
        if pdf_save_path is None:
            pdf_save_path = self.save_sub_dir + '/' + self.job_name \
              + '_report_' + self.datestr + '.pdf'
        doc = SimpleDocTemplate(pdf_save_path,
                                pagesize=letter,
                                rightMargin=72,
                                leftMargin=72,
                                topMargin=72,
                                bottomMargin=18)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        reports = []
        # Reports
        # head
        ptext = "JOB NAME: " + self.job_name
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "JOB TYPE: " + self.job_type
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "DATE TIME: " + self.job_create_time
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        # machine info
        ptext = "MACHINE INFO:"
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "-" * 80
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = 'machine:' + platform.machine()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = 'version:' + platform.version()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = 'platform:' + platform.platform()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = 'system:' + platform.system()
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = 'processor:' + platform.processor()
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
        ptext = "selected features id             : " + str(self.bkg_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features id             : " + str(self.sig_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features id             : " + str(self.data_list)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features id             : " + str(
            self.selected_features)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features id             : " + str(self.reset_feature)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "selected features id             : " + str(
            self.reset_feature_name)
        reports.append(Paragraph(ptext, styles["Justify"]))
        reports.append(Spacer(1, 12))
        ptext = "bkg arrays path                  : " + self.bkg_dict_path
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg arrays used                  : " + self.bkg_key
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg total weight set             : " + str(
            self.bkg_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg remove negtive weight:   " + str(self.bkg_rm_neg_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig arrays path                  : " + self.sig_dict_path
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig arrays used                  : " + self.sig_key
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig total weight                 : " + str(
            self.sig_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "sig remove negtive weight        : " + str(
            self.sig_rm_neg_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data arrays path                 : " + self.data_dict_path
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data arrays used                 : " + self.data_key
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data total weight                : " + str(
            self.data_sumofweight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data remove negtive weight       : " + str(
            self.data_rm_neg_weight)
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
        ptext = "learn rate                       : " + str(self.learn_rate)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "learn decay                      : " + str(
            self.learn_rate_decay)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "test ratio                       : " + str(self.test_rate)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "validation split                 : " + str(self.val_split)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "batch size                       : " + str(self.batch_size)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "epochs                           : " + str(self.epochs)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "signal class weight              : " + str(
            self.sig_class_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "background class weight          : " + str(
            self.bkg_class_weight)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "use early stop                   : " + str(
            self.use_early_stop)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop monitor               : " + str(
            self.early_stop_monitor)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop min_delta             : " + str(
            self.early_stop_min_delta)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop patience              : " + str(
            self.early_stop_patience)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop mode                  : " + str(
            self.early_stop_mode)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "early stop restore_best_weights  : " + str(
            self.early_stop_restore_best_weights)
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
        fig = self.fig_performance_path
        im = Image(fig, 6.4 * inch, 7.2 * inch)
        reports.append(im)
        ## show total weights of sig/bkg/data
        ptext = "sig total weight  : " + str(self.model.total_weight_sig)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "bkg total weight  : " + str(self.model.total_weight_bkg)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ptext = "data total weight : " + str(self.model.total_weight_data)
        reports.append(Paragraph(ptext, styles["Justify"]))
        ### dnn scores
        fig = self.fig_dnn_scores_lin_path
        im1 = Image(fig, 3.2 * inch, 2.4 * inch)
        fig = self.fig_dnn_scores_log_path
        im2 = Image(fig, 3.2 * inch, 2.4 * inch)
        reports.append(Table([[im1, im2]]))
        ### significance scan
        fig = self.fig_significance_scan_path
        im = Image(fig, 6.4 * inch, 2 * inch)
        reports.append(im)
        ### correlation matrix
        fig = self.fig_correlation_matrix_path
        im = Image(fig, 6.4 * inch, 3.2 * inch)
        reports.append(im)
        # build/save
        doc.build(reports)

    def load_arrays(self):
        """Get training arrays."""
        self.bkg_dict = get_arrays.get_bkg(self.bkg_dict_path, self.campaign,
                                           self.channel, self.bkg_list,
                                           self.selected_features)
        self.sig_dict = get_arrays.get_sig(self.sig_dict_path, self.campaign,
                                           self.channel, self.sig_list,
                                           self.selected_features)
        self.data_dict = get_arrays.get_data(self.data_dict_path,
                                             self.campaign, self.channel,
                                             self.data_list,
                                             self.selected_features)
        self.array_is_loaded = True
        if self.show_report or self.save_pdf_report:
            self.plot_bkg_dict = {
                key: self.bkg_dict[key]
                for key in self.plot_bkg_list
            }

    def set_para(self, parsed_val, data_type, config_parser, section,
                 val_name):
        """Sets parameters for training manually."""
        if data_type == 'bool':
            pass
        elif data_type == 'float':
            float_temp = config_parser.getfloat(section, val_name)
            setattr(self, parsed_val, float_temp)

    def try_parse_bool(self, parsed_val, config_parse, section, val_name):
        try:
            setattr(self, parsed_val,
                    config_parse.getboolean(section, val_name))
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)

    def try_parse_float(self, parsed_val, config_parser, section, val_name):
        try:
            setattr(self, parsed_val,
                    config_parser.getfloat(section, val_name))
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)

    def try_parse_int(self, parsed_val, config_parser, section, val_name):
        try:
            setattr(self, parsed_val, config_parser.getint(section, val_name))
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)

    def try_parse_str(self, parsed_val, config_parser, section, val_name):
        try:
            setattr(self, parsed_val, config_parser.get(section, val_name))
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)

    def try_parse_list(self, parsed_val, config_parser, section, val_name):
        try:
            value = json.loads(config_parser.get(section, val_name))
            if not isinstance(value, list):
                value = [value]
            setattr(self, parsed_val, value)
        except:
            if not hasattr(self, parsed_val):
                setattr(self, parsed_val, None)


def get_valid_cfg_path(path):
    """Finds valid path for cfg file in /share folder.
  
  If path is already valid:
    Nothing will be done and original path will be returned.
  If path is not valid:
    Try to add share folder path before to see whether we can get a valid path.
    Otherwise, raise error to ask configuration correction.

  """
    # Check path:
    if os.path.isfile(path):
        return path
    # Check try add share folder prefix
    current_dir = os.getcwd()
    main_dirs = []
    for main_dir_name in MAIN_DIR_NAMES:
        try:
            found_dirs = re.findall(".*" + main_dir_name, current_dir)
            main_dirs += found_dirs
        except:
            pass
    share_dir = None
    for temp in main_dirs:
        share_dir_temp = temp + '/share'
        if os.path.isdir(share_dir_temp):
            share_dir = share_dir_temp
            break
    if share_dir is None:
        raise ValueError('No valid path found, please check .ini file.')
    if os.path.isfile(share_dir + '/train/' + path):
        return share_dir + '/train/' + path
    elif os.path.isfile(share_dir + '/' + path):
        return share_dir + '/' + path
    else:
        raise ValueError('No valid path found, please check .ini file.')


def make_table(data, save_path, num_para=1):
    """Makes table for scan meta data and so on.
    
    Input example:
        data = [
            ["col-1", "col-2", "col-3", "col-4" ],
            [1,2,3,4],
            ["a", "b", "c", "d"]
        ]
    """
    pdf = SimpleDocTemplate(save_path, pagesize=A3)
    table = Table(data)
    # add style
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ])
    table.setStyle(style)
    # 2) Alternate backgroud color
    rowNumb = len(data)
    for i in range(1, rowNumb):
        if i % 2 == 0:
            bc = colors.burlywood
        else:
            bc = colors.beige

        ts = TableStyle([('BACKGROUND', (0, i), (-1, i), bc)])
        table.setStyle(ts)
    # 3) Change background color for scanned parameters
    ts = TableStyle([('BACKGROUND', (0, 1), (num_para - 1, -1),
                      colors.powderblue)])
    table.setStyle(ts)
    # 4) Add borders
    ts = TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('LINEBEFORE', (2, 1), (2, -1), 2, colors.red),
        ('LINEABOVE', (0, 2), (-1, 2), 2, colors.green),
        ('GRID', (0, 0), (-1, -1), 2, colors.black),
    ])
    table.setStyle(ts)
    elems = []
    elems.append(table)
    pdf.build(elems)
