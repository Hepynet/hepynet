import datetime
import logging
import pathlib
import random as python_random
import shutil
import time

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from cycler import cycler

from hepynet.common import common_utils, config_utils
from hepynet.data_io import feed_box
from hepynet.evaluate import (
    evaluate_utils,
    importance,
    kinematics,
    metrics,
    mva_scores,
    significance,
    train_history,
)
from hepynet.main import job_utils
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")


class job_executor(object):
    """Core class to execute a pdnn job based on given cfg file."""

    def __init__(self, yaml_config_path):
        """Initialize executor."""
        self.job_config = None
        self.get_config(yaml_config_path)
        # set up style
        ampl.use_atlas_style(usetex=False)
        # set up color cycle
        color_cycle = self.job_config.apply.color_cycle
        default_cycler = cycler(color=color_cycle)
        plt.rc("axes", prop_cycle=default_cycler)

    def execute_jobs(self, resume=False):
        """Execute all planned jobs."""
        self.job_config.print()
        self.set_save_dir(resume=resume)
        # Execute single job if parameter scan is not needed
        if not self.job_config.para_scan.perform_para_scan:
            self.execute_single_job(resume=resume)
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

    def execute_single_job(self, resume=False):
        """Execute single DNN training with given configuration."""
        # Prepare
        jc = self.job_config.job
        rc = self.job_config.run
        if jc.job_type == "apply":
            if rc.load_dir == None:
                rc.load_dir = jc.save_dir

        if jc.fix_rdm_seed:
            self.fix_random_seed()

        # Check best tuned config overwrite
        if self.job_config.config.best_tune_overwrite:
            logger.info("Overwriting training config with best tuned results")
            best_hypers_path = (
                pathlib.Path(rc.save_sub_dir) / "best_hypers.yaml"
            )
            if best_hypers_path.is_file():
                with open(best_hypers_path, "r") as best_hypers_file:
                    best_hypers = yaml.load(
                        best_hypers_file, Loader=yaml.FullLoader
                    )
                    self.job_config.train.update(best_hypers)

        self.set_model()
        self.set_model_input()

        if jc.job_type == "train":
            self.execute_train_job()
        elif jc.job_type == "tune":
            self.execute_tune_job(resume=resume)
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
        # Save the final job_config
        rc = self.job_config.run
        yaml_path = pathlib.Path(rc.save_sub_dir) / "train_config.yaml"
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(self.job_config.get_config_dict(), yaml_file, indent=4)
        # Train
        self.model_wrapper.build()
        self.model_wrapper.train()

    def execute_tune_job(self, resume=False):
        # Save the final job_config
        ic = self.job_config.input
        uc = self.job_config.tune
        rc = self.job_config.run
        yaml_path = pathlib.Path(rc.save_sub_dir) / "tune_config.yaml"
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(self.job_config.get_config_dict(), yaml_file, indent=4)

        # prepare tmp inputs
        feedbox = feed_box.Feedbox(self.job_config.clone())
        ## get input
        input_df = feedbox.get_processed_df()
        cols = ic.selected_features
        # load and save train/test
        train_index = (
            input_df["is_train"] == True
        )  # index for train and validation
        x = input_df.loc[train_index, cols].values
        y = input_df.loc[train_index, ["y"]].values
        wt = input_df.loc[train_index, "weight"].values

        val_ids = np.random.choice(
            range(len(wt)),
            int(len(wt) * 1.0 * uc.model.val_split),
            replace=False,
        )

        train_ids = np.setdiff1d(np.array(range(len(wt))), val_ids)
        x_train = x[train_ids]
        y_train = y[train_ids]
        wt_train = wt[train_ids]
        x_val = x[val_ids]
        y_val = y[val_ids]
        wt_val = wt[val_ids]
        # remove negative weight events
        if ic.rm_negative_weight_events == True:
            wt_train = wt_train.clip(min=0)
        tune_input_dir = pathlib.Path(rc.tune_input_cache)
        np.save(tune_input_dir / "x_train.npy", x_train)
        np.save(tune_input_dir / "y_train.npy", y_train)
        np.save(tune_input_dir / "wt_train.npy", wt_train)
        np.save(tune_input_dir / "x_val.npy", x_val)
        np.save(tune_input_dir / "y_val.npy", y_val)
        np.save(tune_input_dir / "wt_val.npy", wt_val)
        logger.info(f"Temporary tuning input files saved to: {tune_input_dir}")

        # tuning hypers
        analysis = train_utils.ray_tune(
            self.model_wrapper, self.job_config, resume=resume
        )

        # save results
        best_config = analysis.best_config
        save_path = pathlib.Path(rc.save_sub_dir) / "best_hypers.yaml"
        with open(save_path, "w") as best_hypers_file:
            yaml.dump(best_config, best_hypers_file, indent=4)
        results = analysis.results
        save_path = pathlib.Path(rc.save_sub_dir) / "tune_results.yaml"
        with open(save_path, "w") as tune_results_file:
            yaml.dump(results, tune_results_file, indent=4)

        # remove temporary tuning inputs
        shutil.rmtree(tune_input_dir, ignore_errors=True)
        # remove temporary tuning logs
        if uc.rm_tmp_log:
            #tmp_log_dir = pathlib.Path(rc.save_sub_dir) / "tmp_log"
            tmp_log_dir = uc.tmp_dir
            shutil.rmtree(tmp_log_dir, ignore_errors=True)

    def execute_apply_job(self):
        jc = self.job_config.job
        rc = self.job_config.run
        ic = self.job_config.input
        tc = self.job_config.train
        ac = self.job_config.apply
        # setup save parameters if reports need to be saved
        rc.save_dir = f"{rc.save_sub_dir}/apply/{jc.job_name}"
        pathlib.Path(rc.save_dir).mkdir(parents=True, exist_ok=True)

        # Save the final job_config
        yaml_path = pathlib.Path(rc.save_dir) / "apply_config.yaml"
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(self.job_config.get_config_dict(), yaml_file, indent=4)

        # load inputs
        df_raw = self.model_wrapper.get_feedbox().get_raw_df()
        if logger.level == logging.DEBUG:
            logger.warn(
                f"Randomly sampling 10000 events as input for debugging purpose."
            )
            time.sleep(3)
            df_raw = df_raw.sample(n=10000)
            df_raw.reset_index(drop=True, inplace=True)

        df = self.model_wrapper.get_feedbox().get_processed_df(raw_df=df_raw)

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
                df_raw,
                self.job_config,
                save_dir=f"{rc.save_dir}/kinematics/raw",
                is_raw=True,
            )
            logger.info("Plotting input (processed) distributions.")
            kinematics.plot_input(
                df,
                self.job_config,
                save_dir=f"{rc.save_dir}/kinematics/processed",
                is_raw=False,
            )
        ## correlation matrix
        if ac.book_cor_matrix:
            logger.info("Making correlation matrix")
            kinematics.plot_correlation_matrix(
                df_raw, self.job_config, save_dir=f"{rc.save_dir}/kinematics"
            )

        # Studies depending on models
        if ac.jump_model_studies:
            logger.info("Ignoring model dependent studies as specified")
            return

        ## generate fit arrays
        if ac.book_fit_inputs:
            save_region = ac.fit_df.region
            if save_region is None:
                save_region = ic.region
            logger.info("Dumping numpy arrays for fitting.")
            evaluate_utils.dump_fit_df(
                self.model_wrapper,
                df_raw,
                df,
                self.job_config,
                save_dir=ac.fit_df.save_dir,
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

            # update y values predictions
            y_pred, _, _ = evaluate_utils.k_folds_predict(
                self.model_wrapper.get_model(), df[ic.selected_features].values
            )
            df.loc[:, "y_pred"] = y_pred

            # metrics (PR, ROC, confusion matrix)
            if ac.book_confusion_matrix or ac.book_roc or ac.book_pr:
                metrics.make_metrics_plot(
                    df_raw, df, self.job_config, epoch_subdir
                )
            # overtrain check
            if ac.book_train_test_compare:
                mva_scores.plot_train_test_compare(
                    df, self.job_config, epoch_subdir
                )
            # data/mc scores comparison
            if ac.book_mva_scores_data_mc:
                mva_scores.plot_mva_scores(
                    df_raw, df, self.job_config, epoch_subdir,
                )
            # Make significance scan plot
            if ac.book_significance_scan:
                significance.plot_significance_scan(
                    df, self.job_config, epoch_subdir
                )
            # kinematics with DNN cuts
            if ac.book_cut_kine_study:
                for dnn_cut in ac.cfg_cut_kine_study.dnn_cut_list:
                    dnn_kine_path = (
                        epoch_subdir / f"kine_cut_dnn_p{dnn_cut * 100}"
                    )
                    dnn_kine_path.mkdir(parents=True, exist_ok=True)
                    kinematics.plot_input_dnn(
                        df_raw,
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

        jc = self.job_config.job
        ic = self.job_config.input
        rc = self.job_config.run
        if jc.date_str is None:
            rc.datestr = datetime.date.today().strftime("%Y-%m-%d")
        else:
            rc.datestr = jc.date_str
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

    def set_save_dir(self, resume=False) -> None:
        """Sets the directory to save the outputs"""
        jc = self.job_config.job
        rc = self.job_config.run

        # Determine sub-directory for result saving
        create_new = True
        job_dir_name = jc.job_name
        if jc.job_type == "tune":
            if resume:
                create_new = False
        elif jc.job_type == "train":
            if jc.tune_job_name != "TUNE_JOB_NAME_DEF":
                create_new = False
                job_dir_name = jc.tune_job_name
        elif jc.job_type == "apply":
            create_new = False
            if jc.tune_job_name != "TUNE_JOB_NAME_DEF":
                job_dir_name = jc.tune_job_name
            elif jc.train_job_name != "TRAIN_JOB_NAME_DEF":
                job_dir_name = jc.train_job_name
            else:
                logger.critical(
                    f"Job type is apply but neither tune_job_name nor train_job_name specified! Please update the config."
                )
                exit()
        else:
            logger.critical(f"Unknown job_type {jc.job_type}!")
            exit()

        if create_new:
            dir_pattern = f"{jc.save_dir}/{rc.datestr}_{job_dir_name}_v{{}}"
            output_match = common_utils.get_newest_file_version(dir_pattern)
            rc.save_sub_dir = output_match["path"]
        else:
            if jc.fix_date_str:
                dir_pattern = (
                    f"{jc.save_dir}/{rc.datestr}_{job_dir_name}_v{{}}"
                )
            else:
                dir_pattern = f"{jc.save_dir}/*_{job_dir_name}_v{{}}"
            output_match = common_utils.get_newest_file_version(
                dir_pattern, use_existing=True
            )
            if output_match:
                rc.save_sub_dir = output_match["path"]
            else:
                logger.error(
                    f"Can't find existing work folder matched pattern {dir_pattern}, please check the settings."
                )
                exit()
        logger.info(f"Setup work directory: {rc.save_sub_dir}")
        pathlib.Path(rc.save_sub_dir).mkdir(parents=True, exist_ok=True)

        # set input cache for tune job
        if jc.job_type == "tune":
            tune_input_cache = pathlib.Path(rc.save_sub_dir) / "tmp"
            tune_input_cache.mkdir(parents=True, exist_ok=True)
            rc.tune_input_cache = str(tune_input_cache)

