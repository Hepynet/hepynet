# -*- coding: utf-8 -*-
"""Model wrapper class for DNN training"""
import copy
import datetime
import glob
import logging
import math
import pathlib
import typing

logger = logging.getLogger("hepynet")
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

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
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, callbacks
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop

from hepynet.common import array_utils, common_utils
from hepynet.data_io import feed_box
from hepynet.train import evaluate, train_utils


# self-defined metrics functions
def plain_acc(y_true, y_pred):
    return K.mean(K.less(K.abs(y_pred * 1.0 - y_true * 1.0), 0.5))
    # return 1-K.mean(K.abs(y_pred-y_true))


def get_model_class(model_class: str):
    if model_class == "Model_Sequential_Flat":
        return Model_Sequential_Flat
    else:
        logger.critical(f"Unsupported model class: {model_class}")
        exit(1)


class Model_Base(object):
    """Base model of deep neural network
    """

    def __init__(self, name):
        """Initialize model.

        Args:
        name: str
            Name of the model.

        """
        self._model_create_time = str(datetime.datetime.now())
        self._model_is_compiled = False
        self._model_is_loaded = False
        self._model_is_saved = False
        self._model_is_trained = False
        self._model_name = name
        self._model_save_path = None
        self._train_history = None


class Model_Sequential_Base(Model_Base):
    """Sequential model base.

    Note:
        This class should not be used directly
    """

    def __init__(
        self, job_config,
    ):
        """Initialize model."""
        self._job_config = job_config.clone()
        tc = self._job_config.train
        ic = self._job_config.input
        super().__init__(tc.model_name)
        # Model parameters
        self._model_label = "mod_seq_base"
        self._model_note = "Basic sequential model."
        self._model_input_dim = len(ic.selected_features)
        self._model = Sequential()
        # Arrays
        self._array_prepared = False
        self._selected_features = ic.selected_features
        self._model_meta = {
            "norm_dict": None,
        }
        # Report
        self._save_tb_logs = tc.save_tb_logs
        self._tb_logs_path = tc.tb_logs_path

    def get_model(self):
        """Returns model."""
        if not self._model_is_compiled:
            logger.warning("Model is not compiled")
        return self._model

    def get_model_meta(self):
        return self._model_meta

    def get_train_history(self):
        """Returns train history."""
        if not self._model_is_compiled:
            logger.warning("Model is not compiled")
        if self._train_history is None:
            logger.warning("Empty training history found")
        return self._train_history

    def get_train_performance_meta(self):
        """Returns meta data of training performance

        Note:
            This function should be called after show_performance and
        plot_significance_scan being called, otherwise "-" will be used as
        content.

        """
        performance_meta_dict = {}
        # try collect significance scan result
        try:
            performance_meta_dict["original_significance"] = self.original_significance
            performance_meta_dict["max_significance"] = self.max_significance
            performance_meta_dict[
                "max_significance_threshold"
            ] = self.max_significance_threshold
        except:
            performance_meta_dict["original_significance"] = "-"
            performance_meta_dict["max_significance"] = "-"
            performance_meta_dict["max_significance_threshold"] = "-"
        # try collect auc value
        try:
            # performance_meta_dict["auc_train"] = self.auc_train
            # performance_meta_dict["auc_test"] = self.auc_test
            # performance_meta_dict["auc_train_original"] = self.auc_train_original
            # performance_meta_dict["auc_test_original"] = self.auc_test_original
            pass
        except:
            # performance_meta_dict["auc_train"] = "-"
            # performance_meta_dict["auc_test"] = "-"
            # performance_meta_dict["auc_train_original"] = "-"
            # performance_meta_dict["auc_test_original"] = "-"
            pass
        return performance_meta_dict

    def get_corrcoef(self) -> dict:
        bkg_array, _ = self.feedbox.get_reweight(
            "xb", array_key="all", reset_mass=False
        )
        d_bkg = pd.DataFrame(data=bkg_array, columns=list(self._selected_features),)
        bkg_matrix = d_bkg.corr()
        sig_array, _ = self.feedbox.get_reweight(
            "xs", array_key="all", reset_mass=False
        )
        d_sig = pd.DataFrame(data=sig_array, columns=list(self._selected_features),)
        sig_matrix = d_sig.corr()
        corrcoef_matrix_dict = {}
        corrcoef_matrix_dict["bkg"] = bkg_matrix
        corrcoef_matrix_dict["sig"] = sig_matrix
        return corrcoef_matrix_dict

    def load_model(self, load_dir, _model_name, job_name="*", date="*", version="*"):
        """Loads saved model."""
        # Search possible files
        search_pattern = (
            load_dir + "/" + date + "_" + job_name + "_" + version + "/models"
        )
        model_dir_list = glob.glob(search_pattern)
        if not model_dir_list:
            search_pattern = "/work/" + search_pattern
            logger.debug(f"search pattern:{search_pattern}")
            model_dir_list = glob.glob(search_pattern)
        model_dir_list = sorted(model_dir_list)
        # Choose the newest one
        if len(model_dir_list) < 1:
            raise FileNotFoundError("Model file that matched the pattern not found.")
        model_dir = model_dir_list[-1]
        if len(model_dir_list) > 1:
            logger.info(
                "More than one valid model file found, maybe you should try to specify more infomation."
            )
            logger.info(f"Loading the last matched model path: {model_dir}")
        else:
            logger.info(f"Loading model at: {model_dir}")
        self._model = keras.models.load_model(
            model_dir + "/" + _model_name + ".h5",
            custom_objects={"plain_acc": plain_acc},
        )  # it's important to specify
        # custom objects
        self._model_is_loaded = True
        # Load parameters
        # try:
        paras_path = model_dir + "/" + _model_name + "_paras.yaml"
        self.load_model_parameters(paras_path)
        self.model_paras_is_loaded = True
        # except:
        #    logger.warning("Model parameters not successfully loaded.")
        logger.info("Model loaded.")

    def load_model_with_path(self, model_path, paras_path=None):
        """Loads model with given path

        Note:
            Should load model parameters manually.
        """
        self._model = keras.models.load_model(
            model_path, custom_objects={"plain_acc": plain_acc},
        )  # it's important to specify
        if paras_path is not None:
            try:
                self.load_model_parameters(paras_path)
                self.model_paras_is_loaded = True
            except:
                logger.warning("Model parameters not successfully loaded.")
        logger.info("Model loaded.")

    def load_model_parameters(self, paras_path):
        """Retrieves model parameters from yaml file."""
        with open(paras_path, "r") as paras_file:
            paras_dict = yaml.load(paras_file, Loader=yaml.UnsafeLoader)
        # sorted by alphabet
        self._job_config.update(paras_dict["job_config_dict"])

        model_meta_save = paras_dict["model_meta"]
        self._model_meta = model_meta_save
        self._model_name = model_meta_save["model_name"]
        self._model_label = model_meta_save["model_label"]
        self._model_note = model_meta_save["model_note"]
        self._model_create_time = model_meta_save["model_create_time"]
        self._model_is_compiled = model_meta_save["model_is_compiled"]
        self._model_is_saved = model_meta_save["model_is_saved"]
        self._model_is_trained = model_meta_save["model_is_trained"]

        self._train_history = paras_dict["train_history"]

    def save_model(self, save_dir=None, file_name=None):
        """Saves trained model.

        Args:
            save_dir: str
            Path to save model.

        """
        # Define save path
        if save_dir is None:
            save_dir = "./models"
        if file_name is None:
            datestr = datetime.date.today().strftime("%Y-%m-%d")
            file_name = self._model_name + "_" + self._model_label + "_" + datestr
        # Check path
        save_path = save_dir + "/" + file_name + ".h5"
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Save
        self._model.save(save_path)
        self._model_save_path = save_path
        logger.debug(f"model: {self._model_name} has been saved to: {save_path}")
        # update path for yaml
        save_path = save_dir + "/" + file_name + "_paras.yaml"
        self.save_model_paras(save_path)
        logger.debug(f"model parameters has been saved to: {save_path}")
        self._model_is_saved = True

    def save_model_paras(self, save_path):
        """Save model parameters to yaml file."""
        paras_dict = dict()
        paras_dict["job_config_dict"] = self._job_config.get_config_dict()

        model_meta_save = copy.deepcopy(self._model_meta)
        model_meta_save["model_name"] = self._model_name
        model_meta_save["model_label"] = self._model_label
        model_meta_save["model_note"] = self._model_note
        model_meta_save["model_create_time"] = self._model_create_time
        model_meta_save["model_is_compiled"] = self._model_is_compiled
        model_meta_save["model_is_saved"] = self._model_is_saved
        model_meta_save["model_is_trained"] = self._model_is_trained
        paras_dict["model_meta"] = model_meta_save

        paras_dict["train_history"] = self._train_history

        with open(save_path, "w") as write_file:
            yaml.dump(paras_dict, write_file, indent=2)

    def set_inputs(self, job_config) -> None:
        """Prepares array for training."""
        feedbox = feed_box.Feedbox(
            job_config,
            model_meta=self.get_model_meta(),
        )
        if job_config.job.job_type == "apply":
            feedbox.load_sig_arrays()
            feedbox.load_bkg_arrays()
        self._model_meta["norm_dict"] = copy.deepcopy(feedbox.get_norm_dict())
        self.feedbox = feedbox
        self._array_prepared = feedbox._array_prepared

    def train(
        self, job_config, model_save_dir=None, file_name=None,
    ):
        """Performs training."""

        # prepare config alias
        ic = self._job_config.input
        tc = self._job_config.train

        # Check
        if self._model_is_compiled == False:
            logging.critical("DNN model is not yet compiled")
            exit(1)
        if self._array_prepared == False:
            logging.critical("Training data is not ready.")
            exit(1)

        # Train
        logger.info("-" * 40)
        logger.info(f"Training start. Using model: {self._model_name}")
        logger.info(f"Model info: {self._model_note}")
        self.class_weight = {1: tc.sig_class_weight, 0: tc.bkg_class_weight}
        ## setup callbacks
        train_callbacks = []
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
        if model_save_dir is None:
            model_save_dir = "./models"
        pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = self._model_name
        path_pattern = model_save_dir + "/" + file_name + "_epoch{epoch:03d}.h5"
        checkpoint = ModelCheckpoint(path_pattern, monitor="val_loss")
        train_callbacks.append(checkpoint)
        ## check input
        train_test_dict = self.feedbox.get_train_test_arrays(
            sig_key=ic.sig_key,
            bkg_key=ic.bkg_key,
            multi_class_bkgs=tc.output_bkg_node_names,
            output_keys=train_utils.COMB_KEYS,
        )
        self.feedbox = None
        x_train = train_test_dict["x_train"]
        x_test = train_test_dict["x_test"]
        y_train = train_test_dict["y_train"]
        y_test = train_test_dict["y_test"]
        wt_train = train_test_dict["wt_train"]
        wt_test = train_test_dict["wt_test"]
        train_test_dict = None
        self.get_model().summary()
        ## train
        history_obj = self.get_model().fit(
            x_train,
            y_train,
            batch_size=tc.batch_size,
            epochs=tc.epochs,
            validation_split=tc.val_split,
            sample_weight=wt_train,
            callbacks=train_callbacks,
            verbose=tc.verbose,
        )
        self._train_history = history_obj.history
        logger.info("Training finished.")

        # Evaluation
        logger.info("Evaluate with test dataset:")
        score = self.get_model().evaluate(
            x_test, y_test, verbose=tc.verbose, sample_weight=wt_test,
        )

        if not isinstance(score, typing.Iterable):
            logger.info(f"> test loss: {score}")
        else:
            for i, metric in enumerate(self.get_model().metrics_names):
                logger.info(f"> test - {metric}: {score[i]}")

        # update status
        self._model_is_trained = True

    '''
    def tuning_train(
        self,
        sig_key="all",
        bkg_key="all",
        batch_size=128,
        epochs=20,
        val_split=0.25,
        sig_class_weight=1.0,
        bkg_class_weight=1.0,
        verbose=1,
    ):
        """Performs quick training for hyperparameters tuning."""
        # Check
        if self._model_is_compiled == False:
            raise ValueError("DNN model is not yet compiled")
        if self._array_prepared == False:
            raise ValueError("Training data is not ready.")
        # separate validation samples
        train_test_dict = self.feedbox.get_train_test_arrays(
            sig_key=sig_key,
            bkg_key=bkg_key,
            multi_class_bkgs=self.model_hypers["output_bkg_node_names"],
            use_selected=False,
        )
        x_train = train_test_dict["x_train"]
        y_train = train_test_dict["y_train"]
        x_train_selected = train_test_dict["x_train_selected"]
        num_val = math.ceil(len(y_train) * val_split)
        x_tr = x_train_selected[:-num_val, :]
        x_val = x_train_selected[-num_val:, :]
        y_tr = y_train[:-num_val]
        y_val = y_train[-num_val:]
        wt_tr = x_train[:-num_val, -1]
        wt_val = x_train[-num_val:, -1]
        val_tuple = (x_val, y_val, wt_val)
        self.x_tr = x_tr
        self.x_val = x_val
        self.y_tr = y_tr
        self.y_val = y_val
        self.wt_tr = wt_tr
        self.wt_val = wt_val
        # Train
        self.class_weight = {1: sig_class_weight, 0: bkg_class_weight}
        train_callbacks = []
        if self.model_hypers["use_early_stop"]:
            early_stop_callback = callbacks.EarlyStopping(
                monitor=self.model_hypers["early_stop_paras"]["monitor"],
                min_delta=self.model_hypers["early_stop_paras"]["min_delta"],
                patience=self.model_hypers["early_stop_paras"]["patience"],
                mode=self.model_hypers["early_stop_paras"]["mode"],
                restore_best_weights=self.model_hypers["early_stop_paras"][
                    "restore_best_weights"
                ],
            )
            train_callbacks.append(early_stop_callback)
        self._train_history = self.get_model().fit(
            x_tr,
            y_tr,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_tuple,
            class_weight=self.class_weight,
            sample_weight=wt_tr,
            callbacks=train_callbacks,
            verbose=verbose,
        )
        # Final evaluation
        score = self.get_model().evaluate(
            x_val, y_val, verbose=verbose, sample_weight=wt_val
        )
        # update status
        self._model_is_trained = True
        return score[0]
    '''


class Model_Sequential_Flat(Model_Sequential_Base):
    """Sequential model optimized with old ntuple at Sep. 9th 2019.

    Major modification based on 1002 model:
        1. Change structure to make quantity of nodes decrease with layer num.

    """

    def __init__(
        self, job_config,
    ):
        super().__init__(job_config)

        self._model_label = "mod_seq"
        self._model_note = "Sequential model with flexible layers and nodes."

    def compile(self):
        """ Compile model, function to be changed in the future."""
        tc = self._job_config.train
        # Add layers
        for layer in range(tc.layers):
            # input layer
            if layer == 0:
                self._model.add(
                    Dense(
                        tc.nodes,
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                        input_dim=self._model_input_dim,
                    )
                )
            # hidden layers
            else:
                self._model.add(
                    Dense(
                        tc.nodes,
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                    )
                )
            if tc.dropout_rate != 0:
                self._model.add(Dropout(tc.dropout_rate))
        # output layer
        if tc.output_bkg_node_names:
            num_nodes_out = len(tc.output_bkg_node_names) + 1
        else:
            num_nodes_out = 1
        self._model.add(
            Dense(
                num_nodes_out,
                kernel_initializer="glorot_uniform",
                activation="sigmoid",
            )
        )
        # Compile
        # transfer self-defined metrics into real function
        metrics = copy.deepcopy(tc.train_metrics)
        weighted_metrics = copy.deepcopy(tc.train_metrics_weighted)
        if metrics is not None and "plain_acc" in metrics:
            index = metrics.index("plain_acc")
            metrics[index] = plain_acc
        if weighted_metrics is not None and "plain_acc" in weighted_metrics:
            index = weighted_metrics.index("plain_acc")
            weighted_metrics[index] = plain_acc
        # compile model
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(
                lr=tc.learn_rate,
                decay=tc.learn_rate_decay,
                momentum=tc.momentum,
                nesterov=tc.nesterov,
            ),
            metrics=metrics,
            weighted_metrics=weighted_metrics,
        )
        self._model_is_compiled = True

