# -*- coding: utf-8 -*-
"""Model wrapper class for DNN training"""
import copy
import datetime
import logging
import pathlib
from typing import Dict, Iterable, Optional, Tuple

import keras
import numpy as np
import ray
import tensorflow as tf
import yaml
from keras import backend as K
from keras.callbacks import ModelCheckpoint, callbacks
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from sklearn.metrics import roc_auc_score

import hepynet.common.hepy_type as ht
from hepynet.data_io import feed_box
from hepynet.train import train_utils

logger = logging.getLogger("hepynet")

# fix tensorflow 2.2 issue
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        logger.info(
            f"GPU availability: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
        )
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(e)
# Set up mixed float16 training for TF >= 2.4 TODO: need to upgrade TF
# mixed_precision.set_global_policy('mixed_float16')

# self-defined metrics functions
def plain_acc(y_true, y_pred):
    return K.mean(K.less(K.abs(y_pred * 1.0 - y_true * 1.0), 0.5))
    # return 1-K.mean(K.abs(y_pred-y_true))


class Model_Base(object):
    """Base model of deep neural network
    """

    def __init__(self, job_config: ht.config):
        self._job_config = job_config.clone()
        self._model_create_time = str(datetime.datetime.now())
        self._model_is_compiled = False
        self._model_is_loaded = False
        self._model_is_saved = False
        self._model_is_trained = False
        self._model_name = self._job_config.train.model_name
        self._model_save_path = None
        self._train_history = list()

        num_folds = self._job_config.train.k_folds
        if isinstance(num_folds, int) and num_folds >= 2:
            self._num_folds = num_folds
        else:
            self._num_folds = 1

    def build(self):
        logger.warn(
            "The virtual function Model_Base.build() is called, please implement override funcion!"
        )

    def get_feedbox(self) -> feed_box.Feedbox:
        return self._feedbox

    def get_job_config(self) -> ht.config:
        return self._job_config

    def get_model(self, fold_num=None):
        """Returns model."""
        if not self._model_is_compiled:
            logger.warning("Model is not compiled")
        if fold_num is None:
            return self._model
        else:
            return self._model[fold_num]

    def get_model_meta(self) -> dict:
        return self._model_meta

    def get_model_save_dir(
        self, fold_num: Optional[int] = None
    ) -> ht.pathlike:
        save_dir = f"{self._job_config.run.save_sub_dir}/models"
        if fold_num is not None:
            save_dir += f"/fold_{fold_num}"
        return save_dir

    def set_inputs(self, job_config: ht.config):
        """Prepare feedbox to generate inputs"""
        rc = self._job_config.run.clone()
        try:
            input_dir = pathlib.Path(rc.save_sub_dir) / "models"
            with open(input_dir / "norm_dict.yaml", "r") as norm_file:
                norm_dict = yaml.load(norm_file, Loader=yaml.UnsafeLoader)
            logger.info(f"Successfully loaded norm_dict in {input_dir}")
        except:
            norm_dict = None
        feedbox = feed_box.Feedbox(job_config, norm_dict=norm_dict)
        self._model_meta["norm_dict"] = copy.deepcopy(feedbox.get_norm_dict())
        self._feedbox = feedbox
        self._array_prepared = feedbox._array_prepared

    def load_model(self, epoch: Optional[int] = None):
        """Loads saved model."""
        # load model(s)
        self._model = list()
        num_exist_models = 0
        for fold_num in range(self._num_folds):
            model_dir = self.get_model_save_dir(fold_num=fold_num)
            if epoch is None:
                model_path = pathlib.Path(f"{model_dir}/{self._model_name}.h5")
            else:
                model_path = pathlib.Path(
                    f"{model_dir}/{self._model_name}_epoch{epoch}.h5"
                )
            if model_path.exists():
                fold_model = keras.models.load_model(
                    model_path,
                    custom_objects={"plain_acc": plain_acc,},
                    compile=False,
                )  # it's important to specify custom_objects
                self._model.append(fold_model)
                num_exist_models += 1
        self._model_is_loaded = True
        # Load parameters
        model_dir = self.get_model_save_dir()
        self.load_model_parameters(model_dir)
        self.model_paras_is_loaded = True
        if epoch is None:
            logger.info(f"{num_exist_models}/{self._num_folds} models loaded")
        else:
            logger.info(
                f"{num_exist_models}/{self._num_folds} models loaded for epoch {epoch}"
            )

    def load_model_parameters(self, model_dir: ht.pathlike):
        """Retrieves model parameters from yaml file."""
        paras_path = f"{model_dir}/fold_{0}/{self._model_name}_paras.yaml"
        with open(paras_path, "r") as paras_file:
            paras_dict = yaml.load(paras_file, Loader=yaml.UnsafeLoader)
        # update meta data
        model_meta_save = paras_dict["model_meta"]
        self._model_meta = model_meta_save
        self._model_name = model_meta_save["model_name"]
        self._model_label = model_meta_save["model_label"]
        self._model_note = model_meta_save["model_note"]
        self._model_create_time = model_meta_save["model_create_time"]
        self._model_is_compiled = model_meta_save["model_is_compiled"]
        self._model_is_saved = model_meta_save["model_is_saved"]
        self._model_is_trained = model_meta_save["model_is_trained"]
        # load train history
        for fold_num in range(self._num_folds):
            paras_path = (
                f"{model_dir}/fold_{fold_num}/{self._model_name}_paras.yaml"
            )
            with open(paras_path, "r") as paras_file:
                fold_paras_dict = yaml.load(
                    paras_file, Loader=yaml.UnsafeLoader
                )
            self._train_history[fold_num] = fold_paras_dict["train_history"]

    def save_model(
        self, file_name: Optional[str] = None, fold_num: Optional[int] = None
    ):
        """Saves trained model."""
        # Define save path
        save_dir = self.get_model_save_dir(fold_num=fold_num)
        if file_name is None:
            file_name = self._model_name
        # Check path
        save_path = f"{save_dir}/{file_name}.h5"
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Save
        if fold_num is not None:
            self._model[fold_num].save(save_path)
        else:
            self._model[0].save(save_path)
        self._model_save_path = save_path
        logger.debug(
            f"model: {self._model_name} has been saved to: {save_path}"
        )
        self._model_is_saved = True

    def save_model_paras(
        self, file_name: Optional[str] = None, fold_num: Optional[int] = None
    ):
        """Save model parameters to yaml file."""
        rc = self._job_config.run.clone()
        # prepare paras
        paras_dict = dict()
        model_meta_save = copy.deepcopy(self._model_meta)
        model_meta_save["model_name"] = self._model_name
        model_meta_save["model_label"] = self._model_label
        model_meta_save["model_note"] = self._model_note
        model_meta_save["model_create_time"] = self._model_create_time
        model_meta_save["model_is_compiled"] = self._model_is_compiled
        model_meta_save["model_is_saved"] = self._model_is_saved
        model_meta_save["model_is_trained"] = self._model_is_trained
        paras_dict["model_meta"] = model_meta_save
        paras_dict["train_history"] = self._train_history[fold_num]
        # save to file
        save_dir = self.get_model_save_dir(fold_num=fold_num)
        if file_name is None:
            file_name = self._model_name
        save_path = f"{save_dir}/{file_name}_paras.yaml"
        with open(save_path, "w") as write_file:
            yaml.dump(paras_dict, write_file, indent=2)
        logger.debug(f"model parameters has been saved to: {save_path}")

        norm_dict_path = (
            pathlib.Path(rc.save_sub_dir) / "models" / "norm_dict.yaml"
        )
        with open(norm_dict_path, "w") as norm_file:
            yaml.dump(self._feedbox.get_norm_dict(), norm_file, indent=2)


class Model_Sequential_Base(Model_Base):
    """Sequential model base.

    Note:
        This class should not be used directly
    """

    def __init__(self, job_config: ht.config):
        super().__init__(job_config)
        tc = self._job_config.train
        ic = self._job_config.input
        # Model parameters
        self._model_label = "mod_seq_base"
        self._model_note = "Basic sequential model."
        self._model_input_dim = len(ic.selected_features)
        if isinstance(tc.k_folds, int) and tc.k_folds >= 2:
            self._num_folds = tc.k_folds
            models = list()
            for _ in range(tc.k_folds):
                models.append(Sequential())
            self._model = models
        else:
            self._num_folds = 1
            self._model = [Sequential()]
        self._train_history = [None] * self._num_folds
        # Arrays
        self._array_prepared = False
        self._model_meta = {
            "norm_dict": None,
        }
        # Report
        self._save_tb_logs = tc.save_tb_logs
        self._tb_logs_path = tc.tb_logs_path

    def get_hypers(self):
        """Gets a dictionary of hyper-parameters that defines a training"""
        tc = self._job_config.train.clone()
        hypers = {
            # hypers for building model
            "layers": tc.layers,
            "nodes": tc.nodes,
            "dropout_rate": tc.dropout_rate,
            "output_bkg_node_names": tc.output_bkg_node_names,
            "train_metrics": tc.train_metrics,
            "train_metrics_weighted": tc.train_metrics_weighted,
            "learn_rate": tc.learn_rate,
            "learn_rate_decay": tc.learn_rate_decay,
            "momentum": tc.momentum,
            "nesterov": tc.nesterov,
            # hypers for training
            "val_split": tc.val_split,
            "batch_size": tc.batch_size,
            "epochs": tc.epochs,
            "sig_class_weight": tc.sig_class_weight,
            "bkg_class_weight": tc.bkg_class_weight,
        }
        return hypers

    def get_inputs(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Gets train/test datasets from feedbox"""
        ic = self._job_config.input.clone()
        ## get input
        input_df = self._feedbox.get_processed_df()
        cols = ic.selected_features
        # load train/test
        train_index = input_df["is_train"] == True
        test_index = input_df["is_train"] == False
        x_train = input_df.loc[train_index, cols].values
        x_test = input_df.loc[test_index, cols].values
        y_train = input_df.loc[train_index, ["y"]].values
        y_test = input_df.loc[test_index, ["y"]].values
        wt_train = input_df.loc[train_index, "weight"].values
        wt_test = input_df.loc[test_index, "weight"].values
        del input_df
        # remove negative weight events
        if ic.rm_negative_weight_events == True:
            wt_train = wt_train.clip(min=0)
            wt_test = wt_test.clip(min=0)

        return {
            "train": (x_train, y_train, wt_train),
            "test": (x_test, y_test, wt_test),
        }

    def get_train_callbacks(self, fold_num: Optional[int] = None) -> list:
        """Prepares callbacks of training"""
        train_callbacks = []
        tc = self._job_config.train.clone()
        if self._save_tb_logs:  # TODO: add back this function
            pass
            # if self._tb_logs_path is None:
            #    self._tb_logs_path = "temp_logs/{}".format(self._model_label)
            #    logger.warning(
            #        "TensorBoard logs path not specified, set path to: {}".format(
            #            self._tb_logs_path
            #        )
            #    )
            # tb_callback = TensorBoard(log_dir=self._tb_logs_path, histogram_freq=1)
            # train_callbacks.append(tb_callback)
        if tc.use_early_stop:
            early_stop_callback = callbacks.EarlyStopping(
                monitor=tc.early_stop_paras.monitor,
                min_delta=tc.early_stop_paras.min_delta,
                patience=tc.early_stop_paras.patience,
                mode=tc.early_stop_paras.mode,
                restore_best_weights=tc.early_stop_paras.restore_best_weights,
            )
            train_callbacks.append(early_stop_callback)
        ## set up check point to save model in each epoch
        if tc.save_model:
            model_save_dir = self.get_model_save_dir(fold_num=fold_num)
            pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            path_pattern = (
                f"{model_save_dir}/{self._model_name}_epoch{{epoch}}.h5"
            )
            checkpoint = ModelCheckpoint(path_pattern, monitor="val_loss")
            train_callbacks.append(checkpoint)
        return train_callbacks

    def train(self):
        # Check
        if not self._model_is_compiled:
            logger.critical("DNN model is not yet built, rebuilding")
            self.build()
        if not self._array_prepared:
            logger.critical("Training data is not ready, pleas set up inputs")
            exit(1)
        # Prepare
        tc = self._job_config.train.clone()
        input_dict = self.get_inputs()
        model = self.get_model()
        hypers = self.get_hypers()
        n_folds = self._num_folds
        verbose = tc.verbose
        # Input
        x_train, y_train, wt_train = input_dict["train"]
        x_test, y_test, wt_test = input_dict["test"]
        # Train
        logger.info("-" * 40)
        logger.info("Loading inputs")
        (
            train_index_list,
            validation_index_list,
        ) = train_utils.get_train_val_indices(
            y_train, y_train, wt_train, hypers["val_split"], k_folds=tc.k_folds
        )
        for fold_num in range(n_folds):
            logger.info(f"Training start. Using model: {self._model_name}")
            logger.info(f"Model info: {self._model_note}")
            if n_folds >= 2:
                logger.info(
                    f"Performing k-fold training {fold_num + 1}/{n_folds}"
                )
            fold_model = model[fold_num]
            fold_model.summary()
            train_index = train_index_list[fold_num]
            val_index = validation_index_list[fold_num]
            x_fold = x_train[train_index]
            y_fold = y_train[train_index]
            wt_fold = wt_train[train_index]
            val_x_fold = x_train[val_index]
            val_y_fold = y_train[val_index]
            val_wt_fold = wt_train[val_index]
            val_fold = (val_x_fold, val_y_fold, val_wt_fold)
            logger.info(
                f"> Training on {len(y_fold)}, validating on {len(val_y_fold)} events."
            )
            history_obj = fold_model.fit(
                x_fold,
                y_fold,
                batch_size=hypers["batch_size"],
                epochs=hypers["epochs"],
                validation_data=val_fold,
                shuffle=True,
                class_weight={
                    1: hypers["sig_class_weight"],
                    0: hypers["bkg_class_weight"],
                },
                sample_weight=wt_fold,
                callbacks=self.get_train_callbacks(fold_num=fold_num),
                verbose=verbose,
            )
            logger.info("Training finished.")
            # evaluation
            logger.info("Evaluate with test dataset:")
            score = fold_model.evaluate(
                x_test, y_test, verbose=verbose, sample_weight=wt_test,
            )
            if not isinstance(score, Iterable):
                logger.info(f"> test loss: {score}")
            else:
                for i, metric in enumerate(fold_model.metrics_names):
                    logger.info(f"> test - {metric}: {score[i]}")
            # save training details
            self._train_history[fold_num] = history_obj.history
            self.save_model(fold_num=fold_num)
            self.save_model_paras(fold_num=fold_num)
        # Update status
        self._model_is_trained = True


class Model_Sequential_Flat(Model_Sequential_Base):
    """Flat sequential model"""

    def __init__(self, job_config):
        super().__init__(job_config)

        self._model_label = "mod_seq"
        self._model_note = "Sequential model with flexible layers and nodes."
        self._tune_fun_name = "tune_Model_Sequential_Flat"

    def build(self):
        hypers = self.get_hypers()
        for fold_num in range(self._num_folds):
            fold_model = self._model[fold_num]
            self.build_single(fold_model, hypers)
        self._model_is_compiled = True

    def build_single(self, fold_model, hypers):
        # Add layers
        for layer in range(int(hypers["layers"])):
            # input layer
            if layer == 0:
                fold_model.add(
                    Dense(
                        hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                        input_dim=self._model_input_dim,
                    )
                )
            # hidden layers
            else:
                fold_model.add(
                    Dense(
                        hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                    )
                )
            if hypers["dropout_rate"] != 0:
                fold_model.add(Dropout(hypers["dropout_rate"]))
        # output layer
        if hypers["output_bkg_node_names"]:
            num_nodes_out = len(hypers["output_bkg_node_names"]) + 1
        else:
            num_nodes_out = 1
        fold_model.add(
            Dense(
                num_nodes_out,
                kernel_initializer="glorot_uniform",
                activation="sigmoid",
            )
        )
        # Compile
        # transfer self-defined metrics into real function
        metrics = copy.deepcopy(hypers["train_metrics"])
        weighted_metrics = copy.deepcopy(hypers["train_metrics_weighted"])
        if "plain_acc" in metrics:
            index = metrics.index("plain_acc")
            metrics[index] = plain_acc
        if "plain_acc" in weighted_metrics:
            index = weighted_metrics.index("plain_acc")
            weighted_metrics[index] = plain_acc
        if "auc" in weighted_metrics:
            index = weighted_metrics.index("auc")
            weighted_metrics[index] = tf.keras.metrics.AUC(name="auc")
        # compile model
        fold_model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(
                lr=hypers["learn_rate"],
                decay=hypers["learn_rate_decay"],
                momentum=hypers["momentum"],
                nesterov=hypers["nesterov"],
            ),
            metrics=metrics,
            weighted_metrics=weighted_metrics,
        )

    def get_hypers_tune(self):
        """Gets a dict of hyperparameter (space) for auto-tuning"""
        ic = self._job_config.input.clone()
        rc = self._job_config.run.clone()
        model_cfg = self._job_config.tune.clone().model
        gs = train_utils.get_single_hyper
        hypers = {
            # hypers for building model
            "layers": gs(model_cfg.layers),
            "nodes": gs(model_cfg.nodes),
            "dropout_rate": gs(model_cfg.dropout_rate),
            "output_bkg_node_names": model_cfg.output_bkg_node_names,
            "tune_metrics": model_cfg.tune_metrics,
            "tune_metrics_weighted": model_cfg.tune_metrics_weighted,
            "learn_rate": gs(model_cfg.learn_rate),
            "learn_rate_decay": gs(model_cfg.learn_rate_decay),
            "momentum": gs(model_cfg.momentum),
            "nesterov": gs(model_cfg.nesterov),
            # hypers for training
            "batch_size": gs(model_cfg.batch_size),
            "epochs": gs(model_cfg.epochs),
            "sig_class_weight": gs(model_cfg.sig_class_weight),
            "bkg_class_weight": gs(model_cfg.bkg_class_weight),
            "use_early_stop": model_cfg.use_early_stop,
            "early_stop_paras": model_cfg.early_stop_paras.get_config_dict(),
            # job config
            "input_dim": len(ic.selected_features),
            # input dir
            "input_dir": str(pathlib.Path(rc.tune_input_cache).resolve()),
        }
        return hypers


# tuning functions for ray.tune

## Note: such functions take one config argument and report the score


def tune_Model_Sequential_Flat(config, checkpoint_dir=None):
    """Trainable function for ray-tune"""
    input_dir = pathlib.Path(config["input_dir"])
    x_train = np.load(input_dir / "x_train.npy")
    x_train_unreset = np.load(input_dir / "x_train_unreset.npy")
    y_train = np.load(input_dir / "y_train.npy")
    wt_train = np.load(input_dir / "wt_train.npy")
    x_val = np.load(input_dir / "x_val.npy")
    x_val_unreset = np.load(input_dir / "x_val_unreset.npy")
    y_val = np.load(input_dir / "y_val.npy")
    wt_val = np.load(input_dir / "wt_val.npy")

    # build model
    import tensorflow as tf  # must import inside function, other wise will get errors of ray tune

    model = tf.keras.models.Sequential()
    for layer in range(int(config["layers"])):
        ## input layer
        if layer == 0:
            model.add(
                tf.keras.layers.Dense(
                    config["nodes"],
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                    input_dim=config["input_dim"],
                )
            )
        ## hidden layers
        else:
            model.add(
                tf.keras.layers.Dense(
                    config["nodes"],
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
        if config["dropout_rate"] != 0:
            model.add(tf.keras.layers.Dropout(config["dropout_rate"]))
    ## output layer
    if config["output_bkg_node_names"]:
        num_nodes_out = len(config["output_bkg_node_names"]) + 1
    else:
        num_nodes_out = 1
    model.add(
        tf.keras.layers.Dense(
            num_nodes_out,
            kernel_initializer="glorot_uniform",
            activation="sigmoid",
        )
    )
    metric_auc = tf.keras.metrics.AUC()
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(
            lr=config["learn_rate"],
            decay=config["learn_rate_decay"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        ),
        metrics=config["tune_metrics"],
        weighted_metrics=list(config["tune_metrics_weighted"]) + [metric_auc],
    )

    # set callbacks
    callbacks = list()
    report_dict = {
        "auc": "auc",
        "val_auc": "val_auc",
    }  # default includes AUC of ROC
    for metric in config["tune_metrics"] + config["tune_metrics_weighted"]:
        report_dict[metric] = metric
        report_dict["val_" + metric] = "val_" + metric
    # tune_report_callback = TuneReportCallback(report_dict)
    # callbacks.append(tune_report_callback)
    if config["use_early_stop"]:
        es_config = config["early_stop_paras"]
        early_stop_callback = tf.keras.callbacks.EarlyStopping(**es_config)
        callbacks.append(early_stop_callback)

    # train
    # history_obj = model.fit(
    #    x_train,
    #    y_train,
    #    batch_size=config["batch_size"],
    #    epochs=config["epochs"],
    #    validation_data=(x_val, y_val, wt_val),
    #    shuffle=True,
    #    class_weight={
    #        1: config["sig_class_weight"],
    #        0: config["bkg_class_weight"],
    #    },
    #    sample_weight=wt_train,
    #    callbacks=callbacks,
    # )

    last_auc_unreset = 0
    for epoch_id in range(int(config["epochs"])):
        history_obj = model.fit(
            x_train,
            y_train,
            batch_size=config["batch_size"],
            epochs=1,
            validation_data=(x_val, y_val, wt_val),
            shuffle=True,
            class_weight={
                1: config["sig_class_weight"],
                0: config["bkg_class_weight"],
            },
            sample_weight=wt_train,
            callbacks=callbacks,
        )
        epoch_report = dict()
        for key in report_dict.keys():
            metric = str(key)
            epoch_report[metric] = history_obj.history[metric][-1]

        # y_pred = model.predict(x_val)
        # auc = roc_auc_score(y_val, y_pred, sample_weight=wt_val)
        # epoch_report["val_auc2"] = auc

        y_pred_unreset = model.predict(x_val_unreset)
        auc_unreset = roc_auc_score(
            y_val, y_pred_unreset, sample_weight=wt_val
        )
        epoch_report["auc_unreset"] = auc_unreset

        auc_unreset_improvement = auc_unreset - last_auc_unreset
        epoch_report["auc_unreset_improvement"] = auc_unreset_improvement
        last_auc_unreset = auc_unreset

        epoch_report["epoch_num"] = epoch_id + 1

        tune.report(**epoch_report)

    # evaluate metric
    # final_report = dict()
    # for key in report_dict.keys():
    #    metric = str(key)
    #    final_report[metric] = history_obj.history[metric][-1]
    # tune.report(**final_report)
