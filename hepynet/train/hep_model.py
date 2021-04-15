# -*- coding: utf-8 -*-
"""Model wrapper class for DNN training"""
import copy
import datetime
import logging
import pathlib
from typing import Iterable, Optional

import keras
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import yaml
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, callbacks
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop

from hepynet.data_io import array_utils, feed_box
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

    def __init__(self, job_config):
        """Initialize model.

        Args:
        name: str
            Name of the model.

        """
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

    def compile(self):
        logger.warn(
            "The virtual function Model_Base.compile() is called, please implement override funcion!"
        )

    def get_feedbox(self) -> feed_box.Feedbox:
        return self._feedbox

    def get_job_config(self):
        return self._job_config

    def get_model(self, fold_num=None):
        """Returns model."""
        if not self._model_is_compiled:
            logger.warning("Model is not compiled")
        if fold_num is None:
            return self._model
        else:
            return self._model[fold_num]

    def get_model_meta(self):
        return self._model_meta

    def get_model_save_dir(self, fold_num=None):
        save_dir = f"{self._job_config.run.save_sub_dir}/models"
        if fold_num is not None:
            save_dir += f"/fold_{fold_num}"
        return save_dir

    def get_train_history(self, fold_num=0):
        """Returns train history."""
        fold_train_history = self._train_history[fold_num]
        if not self._model_is_compiled:
            logger.warning("Model is not compiled")
        if len(fold_train_history) == 0:
            logger.warning("Empty training history found")
        return fold_train_history

    def set_inputs(self, job_config) -> None:
        """Prepares array for training."""
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

    def load_model(self, epoch=None):
        """Loads saved model."""
        # # Search possible files
        # search_pattern = (
        #     load_dir + "/" + date + "_" + job_name + "_" + version + "/models"
        # )
        # model_dir_list = glob.glob(search_pattern)
        # if not model_dir_list:
        #     search_pattern = "/work/" + search_pattern
        #     logger.debug(f"search pattern:{search_pattern}")
        #     model_dir_list = glob.glob(search_pattern)
        # model_dir_list = sorted(model_dir_list)
        # # Choose the newest one
        # if len(model_dir_list) < 1:
        #     raise FileNotFoundError("Model file that matched the pattern not found.")
        # model_dir = model_dir_list[-1]
        # if len(model_dir_list) > 1:
        #     logger.info(
        #         "More than one valid model file found, maybe you should try to specify # more infomation."
        #     )
        #     logger.info(f"Loading the last matched model path: {model_dir}")
        # else:
        #     logger.info(f"Loading model at: {model_dir}")

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
                    model_path, custom_objects={"plain_acc": plain_acc},
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

    # def load_model_with_path(self, model_path, paras_path=None, fold_num=0):
    #    """Loads model with given path
    #
    #    Note:
    #        Should load model parameters manually.
    #    """
    #    self._model[fold_num] = keras.models.load_model(
    #        model_path, custom_objects={"plain_acc": plain_acc},
    #    )  # it's important to specify
    #    if paras_path is not None:
    #        try:
    #            self.load_model_parameters(paras_path)
    #            self.model_paras_is_loaded = True
    #        except:
    #            logger.warning("Model parameters not successfully loaded.")
    #    logger.info("Model loaded.")

    def load_model_parameters(self, model_dir):
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
            paras_path = f"{model_dir}/fold_{fold_num}/{self._model_name}_paras.yaml"
            with open(paras_path, "r") as paras_file:
                fold_paras_dict = yaml.load(paras_file, Loader=yaml.UnsafeLoader)
            self._train_history[fold_num] = fold_paras_dict["train_history"]

    def save_model(self, file_name=None, fold_num=None):
        """Saves trained model.

        Args:
            save_dir: str
            Path to save model.

        """
        # Define save path
        save_dir = self.get_model_save_dir(fold_num=fold_num)
        if file_name is None:
            # datestr = datetime.date.today().strftime("%Y-%m-%d")
            # file_name = self._model_name + "_" + self._model_label + "_" + datestr
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
        logger.debug(f"model: {self._model_name} has been saved to: {save_path}")
        self._model_is_saved = True

    def save_model_paras(self, file_name=None, fold_num=None):
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
            # datestr = datetime.date.today().strftime("%Y-%m-%d")
            # file_name = self._model_name + "_" + self._model_label + "_" + datestr
            file_name = self._model_name
        save_path = f"{save_dir}/{file_name}_paras.yaml"
        with open(save_path, "w") as write_file:
            yaml.dump(paras_dict, write_file, indent=2)
        logger.debug(f"model parameters has been saved to: {save_path}")

        norm_dict_path = pathlib.Path(rc.save_sub_dir) / "models" / "norm_dict.yaml"
        with open(norm_dict_path, "w") as norm_file:
            yaml.dump(self._feedbox.get_norm_dict(), norm_file, indent=2)


class Model_Sequential_Base(Model_Base):
    """Sequential model base.

    Note:
        This class should not be used directly
    """

    def __init__(self, job_config):
        """Initialize model."""
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

    def get_train_callbacks(self, fold_num: Optional[int] = None) -> list:
        train_callbacks = []
        tc = self._job_config.train
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
            path_pattern = f"{model_save_dir}/{self._model_name}_epoch{{epoch}}.h5"
            checkpoint = ModelCheckpoint(path_pattern, monitor="val_loss")
            train_callbacks.append(checkpoint)
        return train_callbacks

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

    def train(self):
        """Performs training."""

        # prepare config alias
        ic = self._job_config.input.clone()
        tc = self._job_config.train.clone()
        rc = self._job_config.run.clone()

        # Check
        if not self._model_is_compiled:
            logger.critical("DNN model is not yet compiled, recompiling")
            self.compile()
        if not self._array_prepared:
            logger.critical("Training data is not ready, pleas set up inputs")
            exit(1)

        # Train
        logger.info("-" * 40)
        logger.info("Loading inputs")
        ## get input
        input_df = self._feedbox.get_processed_df()
        cols = ic.selected_features

        train_index = input_df["is_train"] == True
        test_index = input_df["is_train"] == False
        x_train = input_df.loc[train_index, cols].values
        x_test = input_df.loc[test_index, cols].values
        y_train = input_df.loc[train_index, ["y"]].values
        y_test = input_df.loc[test_index, ["y"]].values
        wt_train = input_df.loc[train_index, "weight"].values
        wt_test = input_df.loc[test_index, "weight"].values
        del input_df

        if ic.rm_negative_weight_events == True:
            wt_train = wt_train.clip(min=0)
            wt_test = wt_test.clip(min=0)

        ## train
        train_index_list, validation_index_list = train_utils.get_train_val_indices(
            y_train, y_train, wt_train, tc.val_split, k_folds=tc.k_folds
        )
        for fold_num in range(self._num_folds):
            logger.info(f"Training start. Using model: {self._model_name}")
            logger.info(f"Model info: {self._model_note}")
            if self._num_folds >= 2:
                logger.info(
                    f"Performing k-fold training {fold_num + 1}/{self._num_folds}"
                )
            fold_model = self.get_model(fold_num=fold_num)
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
            history_obj = fold_model.fit(
                x_fold,
                y_fold,
                batch_size=tc.batch_size,
                epochs=tc.epochs,
                validation_data=val_fold,
                shuffle=True,
                class_weight={1: tc.sig_class_weight, 0: tc.bkg_class_weight},
                sample_weight=wt_fold,
                callbacks=self.get_train_callbacks(fold_num=fold_num),
                verbose=tc.verbose,
            )
            logger.info("Training finished.")
            # evaluation
            logger.info("Evaluate with test dataset:")
            score = fold_model.evaluate(
                x_test, y_test, verbose=tc.verbose, sample_weight=wt_test,
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
        train_test_dict = self._feedbox.get_train_test_df(
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
        for fold_num in range(self._num_folds):
            self.compile_single(fold_num=fold_num)
        self._model_is_compiled = True

    def compile_single(self, fold_num):
        """ Compile model, function to be changed in the future."""
        tc = self._job_config.train
        fold_model = self._model[fold_num]
        # Add layers
        for layer in range(tc.layers):
            # input layer
            if layer == 0:
                fold_model.add(
                    Dense(
                        tc.nodes,
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                        input_dim=self._model_input_dim,
                    )
                )
            # hidden layers
            else:
                fold_model.add(
                    Dense(
                        tc.nodes,
                        kernel_initializer="glorot_uniform",
                        activation="relu",
                    )
                )
            if tc.dropout_rate != 0:
                fold_model.add(Dropout(tc.dropout_rate))
        # output layer
        if tc.output_bkg_node_names:
            num_nodes_out = len(tc.output_bkg_node_names) + 1
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
        metrics = copy.deepcopy(tc.train_metrics)
        weighted_metrics = copy.deepcopy(tc.train_metrics_weighted)
        if metrics is not None and "plain_acc" in metrics:
            index = metrics.index("plain_acc")
            metrics[index] = plain_acc
        if weighted_metrics is not None and "plain_acc" in weighted_metrics:
            index = weighted_metrics.index("plain_acc")
            weighted_metrics[index] = plain_acc
        # compile model
        fold_model.compile(
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
