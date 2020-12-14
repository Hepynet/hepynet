# -*- coding: utf-8 -*-
"""Model wrapper class for DNN training"""
import copy
import datetime
import glob
import json
import math
import os
import time
import warnings

# import eli5
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# fix tensorflow 2.2 issue
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# from eli5.sklearn import PermutationImportance
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, callbacks
from keras.layers import Concatenate, Dense, Dropout, Input, Layer
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.train import evaluate, train_utils
from matplotlib.ticker import FixedLocator, NullFormatter
from sklearn.metrics import auc, roc_curve


# self-defined metrics functions
def plain_acc(y_true, y_pred):
    return K.mean(K.less(K.abs(y_pred * 1.0 - y_true * 1.0), 0.5))
    # return 1-K.mean(K.abs(y_pred-y_true))


def get_model_class(model_name: str) -> None:
    if model_name == "Model_Base":
        return Model_Base
    elif model_name == "Model_Sequential_Flat":
        return Model_Sequential_Flat


class Model_Base(object):
    """Base model of deep neural network for pdnn training.

    In feature_list:
        model_create_time: datetime.datetime
        Time stamp of model object created time.
        model_is_compiled: bool
        Whether the model has been compiled.
        model_name: str
        Name of the model.
        train_history:
        Training of keras model, include 'acc', 'loss', 'val_acc' and 'val_loss'.

    """

    def __init__(self, name):
        """Initialize model.

        Args:
        name: str
            Name of the model.

        """
        self.model_create_time = str(datetime.datetime.now())
        self.model_is_compiled = False
        self.model_is_loaded = False
        self.model_is_saved = False
        self.model_is_trained = False
        self.model_name = name
        self.model_save_path = None
        self.train_history = None


class Model_Sequential_Base(Model_Base):
    """Sequential model base.

    Attributes:
        model_input_dim: int
        Number of input variables.
        model_learn_rate: float
        model_decay: float
        model: keras model
        Keras training model object.
        array_prepared = bool
        Whether the array for training has been prepared.
        x_train: numpy array
        x array for training.
        x_test: numpy array
        x array for testing.
        y_train: numpy array
        y array for training.
        y_test: numpy array
        y array for testing.
        xs_test: numpy array
        Signal component of x array for testing.
        xb_test: numpy array
        Background component of x array for testing.
        selected_features: list
        Names of input array of features that will be used for training.
        x_train_selected: numpy array
        x array for training with feature selection.
        x_test_selected: numpy array
        x array for testing with feature selection.
        xs_test_selected: numpy array
        Signal component of x array for testing with feature selection.
        xb_test_selected: numpy array
        Background component of x array for testing with feature selection.
        xs_selected: numpy array
        Signal component of x array (train + test) with feature selection.
        xb_selected: numpy array
        Background component of x array (train + test) with feature selection.

        Example:
        To use to model class, first to create the class:
        >>> model_name = "test model"
        >>> selected_features = ["pt", "eta", "phi"]
        >>> model_deep = model.model_0913(model_name, len(selected_features))
        Then compile model:
        >>> model_deep.compile()
        Prepare array for training:
        >>> xs_emu = np.load('path/to/numpy/signal/array.npy')
        >>> xb_emu = np.load('path/to/numpy/background/array.npy')
        >>> model_deep.prepare_array(xs_emu, xb_emu)
        Perform training:
        >>> model_deep.train(epochs = epochs, val_split = 0.1, verbose = 0)
        Make plots to shoe training performance:
        >>> model_deep.show_performance()

    """

    def __init__(
        self,
        name,
        input_features,
        hypers,
        validation_features=[],
        sig_key=None,
        bkg_key=None,
        data_key=None,
        save_tb_logs=False,
        tb_logs_path=None,
    ):
        """Initialize model."""
        super().__init__(name)
        # Model parameters
        self.model_label = "mod_seq_base"
        self.model_note = "Basic sequential model."
        self.model_input_dim = len(input_features)
        self.model_hypers = copy.deepcopy(hypers)
        self.model = Sequential()
        # Arrays
        self.array_prepared = False
        self.selected_features = input_features
        self.validation_features = validation_features
        self.model_meta = {
            "sig_key": sig_key,
            "bkg_key": bkg_key,
            "data_key": data_key,
            "norm_dict": None,
        }
        # Report
        self.save_tb_logs = save_tb_logs
        self.tb_logs_path = tb_logs_path

    def compile(self):
        """ Compile model, function to be changed in the future.

        Note:
            Needs to be override in child class

        """
        pass

    def get_model(self):
        """Returns model."""
        if not self.model_is_compiled:
            warnings.warn("Model is not compiled")
        return self.model

    def get_train_history(self):
        """Returns train history."""
        if not self.model_is_compiled:
            warnings.warn("Model is not compiled")
        if self.train_history is None:
            warnings.warn("Empty training history found")
        return self.train_history

    def get_train_performance_meta(self):
        """Returns meta data of trainning performance

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
                "max_significance_threshould"
            ] = self.max_significance_threshould
        except:
            performance_meta_dict["original_significance"] = "-"
            performance_meta_dict["max_significance"] = "-"
            performance_meta_dict["max_significance_threshould"] = "-"
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
        d_bkg = pd.DataFrame(data=bkg_array, columns=list(self.selected_features),)
        bkg_matrix = d_bkg.corr()
        sig_array, _ = self.feedbox.get_reweight(
            "xs", array_key="all", reset_mass=False
        )
        d_sig = pd.DataFrame(data=sig_array, columns=list(self.selected_features),)
        sig_matrix = d_sig.corr()
        corrcoef_matrix_dict = {}
        corrcoef_matrix_dict["bkg"] = bkg_matrix
        corrcoef_matrix_dict["sig"] = sig_matrix
        return corrcoef_matrix_dict

    def load_model(self, load_dir, model_name, job_name="*", date="*", version="*"):
        """Loads saved model."""
        # Search possible files
        search_pattern = (
            load_dir + "/" + date + "_" + job_name + "_" + version + "/models"
        )
        model_dir_list = glob.glob(search_pattern)
        if not model_dir_list:
            search_pattern = "/work/" + search_pattern
            print("search pattern:", search_pattern)
            model_dir_list = glob.glob(search_pattern)
        model_dir_list = sorted(model_dir_list)
        # Choose the newest one
        if len(model_dir_list) < 1:
            raise FileNotFoundError("Model file that matched the pattern not found.")
        model_dir = model_dir_list[-1]
        if len(model_dir_list) > 1:
            print(
                "More than one valid model file found, try to specify more infomation."
            )
            print("Loading the last matched model path:", model_dir)
        else:
            print("Loading model at:", model_dir)
        self.model = keras.models.load_model(
            model_dir + "/" + model_name + ".h5",
            custom_objects={"plain_acc": plain_acc},
        )  # it's important to specify
        # custom objects
        self.model_is_loaded = True
        # Load parameters
        try:
            paras_path = model_dir + "/" + model_name + "_paras.json"
            self.load_model_parameters(paras_path)
            self.model_paras_is_loaded = True
        except:
            warnings.warn("Model parameters not successfully loaded.")
        print("Model loaded.")

    def load_model_with_path(self, model_path, paras_path=None):
        """Loads model with given path

        Note:
            Should load model parameters manually.
        """
        self.model = keras.models.load_model(
            model_path, custom_objects={"plain_acc": plain_acc},
        )  # it's important to specify
        if paras_path is not None:
            try:
                self.load_model_parameters(paras_path)
                self.model_paras_is_loaded = True
            except:
                warnings.warn("Model parameters not successfully loaded.")
        print("Model loaded.")

    def load_model_parameters(self, paras_path):
        """Retrieves model parameters from json file."""
        with open(paras_path, "r") as paras_file:
            paras_dict = json.load(paras_file)
        # sorted by alphabet
        self.class_weight = common_utils.dict_key_strtoint(paras_dict["class_weight"])
        self.model_create_time = paras_dict["model_create_time"]
        self.model_input_dim = paras_dict["model_input_dim"]
        self.model_is_compiled = paras_dict["model_is_compiled"]
        self.model_is_saved = paras_dict["model_is_saved"]
        self.model_is_trained = paras_dict["model_is_trained"]
        self.model_label = paras_dict["model_label"]
        self.model_hypers = paras_dict["model_hypers"]
        self.model_meta = paras_dict["model_meta"]
        self.model_name = paras_dict["model_name"]
        self.model_note = paras_dict["model_note"]
        self.train_history_accuracy = paras_dict["train_history_accuracy"]
        self.train_history_val_accuracy = paras_dict["train_history_val_accuracy"]
        self.train_history_loss = paras_dict["train_history_loss"]
        self.train_history_val_loss = paras_dict["train_history_val_loss"]

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
            file_name = self.model_name + "_" + self.model_label + "_" + datestr
        # Check path
        save_path = save_dir + "/" + file_name + ".h5"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # Save
        self.model.save(save_path)
        self.model_save_path = save_path
        print(f"\033[F model:", self.model_name, "has been saved to:", save_path)
        # update path for json
        save_path = save_dir + "/" + file_name + "_paras.json"
        self.save_model_paras(save_path)
        print("model parameters has been saved to:", save_path)
        self.model_is_saved = True

    def save_model_paras(self, save_path):
        """Save model parameters to json file."""
        # sorted by alphabet
        paras_dict = {}
        paras_dict["class_weight"] = self.class_weight
        paras_dict["model_create_time"] = self.model_create_time
        paras_dict["model_input_dim"] = self.model_input_dim
        paras_dict["model_is_compiled"] = self.model_is_compiled
        paras_dict["model_is_saved"] = self.model_is_saved
        paras_dict["model_is_trained"] = self.model_is_trained
        paras_dict["model_label"] = self.model_label
        paras_dict["model_hypers"] = self.model_hypers
        paras_dict["model_meta"] = self.model_meta
        paras_dict["model_name"] = self.model_name
        paras_dict["model_note"] = self.model_note
        paras_dict["train_history_accuracy"] = self.train_history_accuracy
        paras_dict["train_history_val_accuracy"] = self.train_history_val_accuracy
        paras_dict["train_history_loss"] = self.train_history_loss
        paras_dict["train_history_val_loss"] = self.train_history_val_loss
        with open(save_path, "w") as write_file:
            json.dump(paras_dict, write_file, indent=2)

    def set_inputs(self, feedbox: dict, apply_data=False) -> None:
        """Prepares array for training."""
        self.feedbox = feedbox
        self.array_prepared = feedbox.array_prepared
        self.model_meta["norm_dict"] = feedbox.norm_dict

    def train(
        self,
        sig_key="all",
        bkg_key="all",
        batch_size=128,
        epochs=20,
        val_split=0.25,
        sig_class_weight=1.0,
        bkg_class_weight=1.0,
        verbose=1,
        save_dir=None,
        file_name=None,
    ):
        """Performs training."""
        # Check
        if self.model_is_compiled == False:
            raise ValueError("DNN model is not yet compiled")
        if self.array_prepared == False:
            raise ValueError("Training data is not ready.")
        # Train
        print("-" * 40)
        print("Training start. Using model:", self.model_name)
        print("Model info:", self.model_note)
        self.class_weight = {1: sig_class_weight, 0: bkg_class_weight}
        train_callbacks = []
        if self.save_tb_logs:
            if self.tb_logs_path is None:
                self.tb_logs_path = "temp_logs/{}".format(self.model_label)
                warnings.warn(
                    "TensorBoard logs path not specified, set path to: {}".format(
                        self.tb_logs_path
                    )
                )
            tb_callback = TensorBoard(log_dir=self.tb_logs_path, histogram_freq=1)
            train_callbacks.append(tb_callback)
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
        # set up check point to save model in each epoch
        if save_dir is None:
            save_dir = "./models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if file_name is None:
            datestr = datetime.date.today().strftime("%Y-%m-%d")
            file_name = self.model_name
        path_pattern = save_dir + "/" + file_name + "_epoch{epoch:03d}.h5"
        checkpoint = ModelCheckpoint(
            path_pattern, monitor="val_loss", verbose=1, period=1
        )
        train_callbacks.append(checkpoint)
        # check input
        train_test_dict = self.feedbox.get_train_test_arrays(
            sig_key=sig_key,
            bkg_key=bkg_key,
            multi_class_bkgs=self.model_hypers["output_bkg_node_names"],
        )
        x_train = train_test_dict["x_train"]
        x_test = train_test_dict["x_test"]
        y_train = train_test_dict["y_train"]
        y_test = train_test_dict["y_test"]
        wt_train = train_test_dict["wt_train"]
        wt_test = train_test_dict["wt_test"]
        if np.isnan(np.sum(x_train)):
            exit(1)
        if np.isnan(np.sum(y_train)):
            exit(1)
        self.get_model().summary()
        self.train_history = self.get_model().fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            sample_weight=wt_train,
            callbacks=train_callbacks,
            verbose=verbose,
        )
        print("Training finished.")
        # Quick evaluation
        print("Quick evaluation:")
        score = self.get_model().evaluate(
            x_test, y_test, verbose=verbose, sample_weight=wt_test,
        )
        print("> test loss:", score[0])
        print("> test accuracy:", score[1])
        print(self.get_model().metrics_names)
        print(score)
        # Save train history
        # save accuracy history
        self.train_history_accuracy = [
            float(ele) for ele in self.train_history.history["accuracy"]
        ]
        try:
            self.train_history_accuracy = [
                float(ele) for ele in self.train_history.history["acc"]
            ]
            self.train_history_val_accuracy = [
                float(ele) for ele in self.train_history.history["val_acc"]
            ]
        except:  # updated for tensorflow2.0
            self.train_history_accuracy = [
                float(ele) for ele in self.train_history.history["accuracy"]
            ]
            self.train_history_val_accuracy = [
                float(ele) for ele in self.train_history.history["val_accuracy"]
            ]
        # save loss history/
        self.train_history_loss = [
            float(ele) for ele in self.train_history.history["loss"]
        ]
        self.train_history_val_loss = [
            float(ele) for ele in self.train_history.history["val_loss"]
        ]
        # update status
        self.model_is_trained = True

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
        if self.model_is_compiled == False:
            raise ValueError("DNN model is not yet compiled")
        if self.array_prepared == False:
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
        self.train_history = self.get_model().fit(
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
        self.model_is_trained = True
        return score[0]


class Model_Sequential_Flat(Model_Sequential_Base):
    """Sequential model optimized with old ntuple at Sep. 9th 2019.

    Major modification based on 1002 model:
        1. Change structure to make quantity of nodes decrease with layer num.

    """

    def __init__(
        self,
        name,
        input_features,
        hypers,
        validation_features=[],
        sig_key="all",
        bkg_key="all",
        data_key="all",
        save_tb_logs=False,
        tb_logs_path=None,
    ):
        super().__init__(
            name,
            input_features,
            hypers,
            validation_features=validation_features,
            sig_key=sig_key,
            bkg_key=bkg_key,
            data_key=data_key,
            save_tb_logs=save_tb_logs,
            tb_logs_path=tb_logs_path,
        )

        self.model_label = "mod_seq"
        self.model_note = "Sequential model with flexible layers and nodes."
        assert (
            self.model_hypers["layers"] > 0
        ), "Model layer quantity should be positive"

    def compile(self):
        """ Compile model, function to be changed in the future."""
        # Add layers
        # input
        for layer in range(self.model_hypers["layers"]):
            if layer == 0:
                self.model.add(
                    Dense(
                        self.model_hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                        input_dim=self.model_input_dim,
                    )
                )
            else:
                self.model.add(
                    Dense(
                        self.model_hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                    )
                )
            if self.model_hypers["dropout_rate"] != 0:
                self.model.add(Dropout(self.model_hypers["dropout_rate"]))
        # output
        self.model.add(
            Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")
        )
        # Compile
        # transfer self-defined metrics into real function
        metrics = copy.deepcopy(self.model_hypers["metrics"])
        weighted_metrics = copy.deepcopy(self.model_hypers["weighted_metrics"])
        if "plain_acc" in metrics:
            index = metrics.index("plain_acc")
            metrics[index] = plain_acc
        if "plain_acc" in weighted_metrics:
            index = weighted_metrics.index("plain_acc")
            weighted_metrics[index] = plain_acc
        # compile model
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(
                lr=self.model_hypers["learn_rate"],
                decay=self.model_hypers["decay"],
                momentum=self.model_hypers["momentum"],
                nesterov=self.model_hypers["nesterov"],
            ),
            metrics=metrics,
            weighted_metrics=weighted_metrics,
        )
        self.model_is_compiled = True


class Model_Sequential_Multi_Class(Model_Sequential_Base):
    """Sequential model with multiple class"""

    def __init__(
        self,
        name,
        input_features,
        hypers,
        validation_features=[],
        sig_key="all",
        bkg_key="all",
        data_key="all",
        save_tb_logs=False,
        tb_logs_path=None,
    ):
        super().__init__(
            name,
            input_features,
            hypers,
            validation_features=validation_features,
            sig_key=sig_key,
            bkg_key=bkg_key,
            data_key=data_key,
            save_tb_logs=False,
            tb_logs_path=None,
        )

        self.model_label = "mod_seq"
        self.model_note = "Sequential model with multiple class."
        assert (
            self.model_hypers["layers"] > 0
        ), "Model layer quantity should be positive"

    def compile(self):
        """ Compile model, function to be changed in the future."""
        # Add layers
        # input
        for layer in range(self.model_hypers["layers"]):
            if layer == 0:
                self.model.add(
                    Dense(
                        self.model_hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                        input_dim=self.model_input_dim,
                    )
                )
            else:
                self.model.add(
                    Dense(
                        self.model_hypers["nodes"],
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                    )
                )
            if self.model_hypers["dropout_rate"] != 0:
                self.model.add(Dropout(self.model_hypers["dropout_rate"]))
        # output
        self.model.add(
            Dense(
                len(self.model_hypers["output_bkg_node_names"]) + 1,
                kernel_initializer="glorot_uniform",
                activation="softmax",
            )
        )
        # Compile
        # transfer self-defined metrics into real function
        metrics = copy.deepcopy(self.model_hypers["metrics"])
        weighted_metrics = copy.deepcopy(self.model_hypers["weighted_metrics"])
        if "plain_acc" in metrics:
            index = metrics.index("plain_acc")
            metrics[index] = plain_acc
        if "plain_acc" in weighted_metrics:
            index = weighted_metrics.index("plain_acc")
            weighted_metrics[index] = plain_acc
        # compile model
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(
                lr=self.model_hypers["learn_rate"],
                decay=self.model_hypers["decay"],
                momentum=self.model_hypers["momentum"],
                nesterov=self.model_hypers["nesterov"],
            ),
            metrics=metrics,
            weighted_metrics=weighted_metrics,
        )
        self.model_is_compiled = True

