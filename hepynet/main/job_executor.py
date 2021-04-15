import datetime
import logging
import pathlib
import random as python_random

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from hepynet.common import common_utils, config_utils
from hepynet.evaluate import (
    evaluate_utils,
    importance,
    kinematics,
    mva_scores,
    roc,
    significance,
    train_history,
)
from hepynet.main import job_utils
from hepynet.train import train_utils

# from hepynet.common.hepy_const import SCANNED_PARAS


logger = logging.getLogger("hepynet")


class job_executor(object):
    """Core class to execute a pdnn job based on given cfg file."""

    def __init__(self, yaml_config_path):
        """Initialize executor."""
        self.job_config = None
        self.get_config(yaml_config_path)
        # set up style
        ampl.use_atlas_style(usetex=False)
        ampl.set_color_cycle(pal="ATLAS", n=10)

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

        if jc.fix_rdm_seed:
            self.fix_random_seed()

        self.set_model()
        self.set_model_input()

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
        self.model_wrapper.compile()
        self.model_wrapper.train()

    def execute_apply_job(self):
        jc = self.job_config.job
        rc = self.job_config.run
        ic = self.job_config.input
        tc = self.job_config.train
        ac = self.job_config.apply
        # setup save parameters if reports need to be saved
        rc.save_dir = f"{rc.save_sub_dir}/apply/{jc.job_name}"
        pathlib.Path(rc.save_dir).mkdir(parents=True, exist_ok=True)

        # load inputs
        df = self.model_wrapper.get_feedbox().get_processed_df()
        df_raw = self.model_wrapper.get_feedbox().get_raw_df()

        # Studies not depending on models
        ## metrics curves
        if ac.book_history:
            train_history.plot_history(
                self.model_wrapper, self.job_config, save_dir=rc.save_dir
            )
        ## input kinematic plots
        if ac.book_kine:
            logger.info("Plotting input (raw) distributions.")
            kinematics.plot_input(
                df_raw, self.job_config, save_dir=f"{rc.save_dir}/kinematics/raw",
            )
            logger.info("Plotting input (processed) distributions.")
            kinematics.plot_input(
                df, self.job_config, save_dir=f"{rc.save_dir}/kinematics/processed",
            )
        ## correlation matrix
        if ac.book_cor_matrix:
            logger.info("Making correlation matrix")
            kinematics.plot_correlation_matrix(
                df_raw, self.job_config, save_dir=f"{rc.save_dir}/kinematics"
            )

        # Studies depending on models

        ## generate fit arrays
        if ac.book_fit_npy:
            save_region = ac.cfg_fit_npy.fit_npy_region
            if save_region is None:
                save_region = ic.region
            npy_dir = f"{ac.cfg_fit_npy.npy_save_dir}/{ic.campaign}/{save_region}"
            logger.info("Dumping numpy arrays for fitting.")
            evaluate_utils.dump_fit_npy(
                self.model_wrapper, df_raw, df, self.job_config, npy_dir=npy_dir,
            )

        ## loop over models at different epochs
        epoch_checklist = [None]
        if ac.check_model_epoch:
            for epoch_id in range(tc.epochs):
                epoch = epoch_id + 1
                if epoch % ac.epoch_check_interval == 1:
                    epoch_checklist.append(epoch)
        for epoch in epoch_checklist:
            logger.info(">" * 80)
            if epoch is None:
                logger.info(f"Checking model(s) at final epoch")
                epoch_str = "final"
            else:
                logger.info(f"Checking model(s) at epoch {epoch}")
                epoch_str = str(epoch)
            self.model_wrapper.load_model(epoch=epoch)

            # create epoch sub-directory
            epoch_subdir = evaluate_utils.create_epoch_subdir(
                rc.save_dir, epoch, len(str(tc.epochs))
            )
            if epoch_subdir is None:
                logger.error(
                    f"Can't create epoch subdir, skip evaluation at epoch {epoch_str}!"
                )
                return

            # roc
            if ac.book_roc:
                logger.info("Making roc curve plot")
                roc.plot_multi_class_roc(
                    self.model_wrapper, df, self.job_config, epoch_subdir
                )
            # overtrain check
            if ac.book_train_test_compare:
                logger.info("Making train/test compare plots")
                mva_scores.plot_train_test_compare(
                    self.model_wrapper, df, self.job_config, epoch_subdir
                )
            # data/mc scores comparison
            if ac.book_mva_scores_data_mc:
                logger.info("Making data/mc scores distributions plots")
                mva_scores.plot_mva_scores(
                    self.model_wrapper, df, self.job_config, epoch_subdir,
                )
            # Make significance scan plot
            if ac.book_significance_scan:
                significance.plot_significance_scan(
                    self.model_wrapper, df, self.job_config, epoch_subdir
                )
            # kinematics with DNN cuts
            if ac.book_cut_kine_study:
                logger.info("Making kinematic plots with different DNN cut")
                for dnn_cut in ac.cfg_cut_kine_study.dnn_cut_list:
                    dnn_kine_path = epoch_subdir / f"kine_cut_dnn_p{dnn_cut * 100}"
                    dnn_kine_path.mkdir(parents=True, exist_ok=True)
                    kinematics.plot_input_dnn(
                        self.model_wrapper,
                        df,
                        self.job_config,
                        dnn_cut=dnn_cut,
                        save_dir=dnn_kine_path,
                    )
            # feature permuted importance
            if ac.book_importance_study:
                logger.info("Checking input feature importance")
                importance.plot_feature_importance(
                    self.model_wrapper, df, self.job_config, epoch_subdir
                )

            logger.info("<" * 80)

        return

    def fix_random_seed(self) -> None:
        """Fixes random seed

        Ref:
            https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

        Note:
            The seed setting funcions called in this function shouldn't be set 
            again in later code, otherwise extra randomness will be introduced 
            (even if set same seed). The reason is unknown yet.

        """
        seed = self.job_config.job.rdm_seed
        # The below is necessary for starting Numpy generated random numbers
        # in a well-defined initial state.
        np.random.seed(seed)
        # The below is necessary for starting core Python generated random numbers
        # in a well-defined state.
        python_random.seed(seed)
        # The below set_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see:
        # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
        tf.random.set_seed(seed)

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
        jc = self.job_config.job
        model_class = train_utils.get_model_class(tc.model_class)
        self.model_wrapper = model_class(self.job_config)
        # load model for "apply" job
        if jc.job_type == "apply":
            self.model_wrapper.load_model()

    def set_model_input(self) -> None:
        logger.info("Processing inputs")
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
            pathlib.Path(rc.save_sub_dir).mkdir(parents=True, exist_ok=True)
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
            pathlib.Path(rc.save_sub_dir).mkdir(parents=True, exist_ok=True)
