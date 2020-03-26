# -*- coding: utf-8 -*-
"""Model class for DNN training"""
import datetime
import glob
import json
import os
import time
import warnings

import eli5
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from eli5.sklearn import PermutationImportance
from keras import backend as K
from keras.callbacks import TensorBoard, callbacks
from keras.layers import Concatenate, Dense, Dropout, Input, Layer
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from matplotlib.ticker import FixedLocator, NullFormatter
from sklearn.metrics import auc, roc_curve

import ROOT
from HEPTools.plot_utils import plot_utils, th1_tools
from lfv_pdnn.common import array_utils, common_utils
from lfv_pdnn.data_io import get_arrays
from lfv_pdnn.train import train_utils


# self-defined metrics functions
def plain_acc(y_true, y_pred):
    return K.mean(K.less(K.abs(y_pred * 1. - y_true * 1.), 0.5))
    #return 1-K.mean(K.abs(y_pred-y_true))


class model_base(object):
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
        self.has_data = False
        self.is_mass_reset = False
        self.model_create_time = str(datetime.datetime.now())
        self.model_is_compiled = False
        self.model_is_loaded = False
        self.model_is_saved = False
        self.model_is_trained = False
        self.model_name = name
        self.model_save_path = None
        self.train_history = None


class model_sequential(model_base):
    """Sequential model base.

    Attributes:
        model_input_dim: int
        Number of input variables.
        model_num_node: int
        Number of nodes in each layer. 
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
    def __init__(self,
                 name,
                 input_dim,
                 num_node=300,
                 learn_rate=0.025,
                 decay=1e-6,
                 metrics=['plain_acc'],
                 weighted_metrics=['accuracy'],
                 selected_features=[]):
        """Initialize model."""
        model_base.__init__(self, name)
        # Model parameters
        self.model_input_dim = input_dim
        self.model_num_node = num_node
        self.model_learn_rate = learn_rate
        self.model_decay = decay
        self.model = Sequential()
        if 'plain_acc' in metrics:
            metrics[metrics.index('plain_acc')] = plain_acc
        if 'plain_acc' in weighted_metrics:
            weighted_metrics[weighted_metrics.index('plain_acc')] = plain_acc
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics
        # Arrays
        self.array_prepared = False
        self.selected_features = selected_features
        # Others
        self.norm_average = None
        self.norm_variance = None

    def calculate_auc(self, xs, xb, class_weight=None, shuffle_col=None):
        """Returns auc of given sig/bkg array."""
        x_plot, y_plot, y_pred = self.process_array(xs,
                                                    xb,
                                                    class_weight=class_weight,
                                                    shuffle_col=shuffle_col)
        fpr_dm, tpr_dm, _ = roc_curve(y_plot,
                                      y_pred,
                                      sample_weight=x_plot[:, -1])
        # Calculate auc and return
        auc_value = auc(fpr_dm, tpr_dm)
        return auc_value

    def compile(self):
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

    def get_corrcoef(self) -> dict:
        d_bkg = pd.DataFrame(data=self.xb_selected_original_mass,
                 columns=list(self.selected_features))
        bkg_matrix = d_bkg.corr()
        d_sig = pd.DataFrame(data=self.xs_selected_original_mass,
                 columns=list(self.selected_features))
        sig_matrix = d_sig.corr()
        corrcoef_matrix_dict = {}
        corrcoef_matrix_dict["bkg"] = bkg_matrix
        corrcoef_matrix_dict["sig"] = sig_matrix
        return corrcoef_matrix_dict

    def load_model(self,
                   dir,
                   model_name,
                   job_name='*',
                   model_class='*',
                   date='*',
                   version='*'):
        """Loads saved model."""
        # Search possible files
        search_pattern = dir + '/' + model_name + '*.h5'
        model_path_list = glob.glob(search_pattern)
        search_pattern = dir + '/' + date + '_' + job_name + '_' + version \
          + '/models/' + model_name + '_' + version + '*.h5'
        model_path_list += glob.glob(search_pattern)
        # Choose the newest one
        if len(model_path_list) < 1:
            raise FileNotFoundError(
                "Model file that matched the pattern not found.")
        model_path = model_path_list[-1]
        if len(model_path_list) > 1:
            print(
                "More than one valid model file found, try to specify more infomation."
            )
            print("Loading the last matched model path:", model_path)
        else:
            print("Loading model at:", model_path)
        self.model = keras.models.load_model(model_path,
                                             custom_objects={
                                                 'plain_acc': plain_acc
                                             })  # it's important to specify
        # custom objects
        self.model_is_loaded = True
        # Load parameters
        #try:
        paras_path = os.path.splitext(model_path)[0] + "_paras.json"
        self.load_model_parameters(paras_path)
        self.model_paras_is_loaded = True
        #except:
        #  warnings.warn("Model parameters not successfully loaded.")
        print("Model loaded.")

    def load_model_parameters(self, paras_path):
        """Retrieves model parameters from json file."""
        with open(paras_path, 'r') as paras_file:
            paras_dict = json.load(paras_file)
        # sorted by aphabet
        self.class_weight = common_utils.dict_key_strtoint(
            paras_dict['class_weight'])
        self.model_create_time = paras_dict['model_create_time']
        self.model_decay = paras_dict['model_decay']
        self.model_input_dim = paras_dict['model_input_dim']
        self.model_is_compiled = paras_dict['model_is_compiled']
        self.model_is_saved = paras_dict['model_is_saved']
        self.model_is_trained = paras_dict['model_is_trained']
        self.model_label = paras_dict['model_label']
        self.model_learn_rate = paras_dict['model_learn_rate']
        self.model_name = paras_dict['model_name']
        self.model_note = paras_dict['model_note']
        self.model_num_node = paras_dict['model_num_node']
        self.train_history_accuracy = paras_dict['train_history_accuracy']
        self.train_history_val_accuracy = paras_dict[
            'train_history_val_accuracy']
        self.train_history_loss = paras_dict['train_history_loss']
        self.train_history_val_loss = paras_dict['train_history_val_loss']

    def make_bar_plot(self,
                      ax,
                      datas,
                      labels,
                      weights,
                      bins,
                      range,
                      title=None,
                      x_lable=None,
                      y_lable=None,
                      x_unit=None,
                      x_scale=None,
                      density=False,
                      use_error=False):
        """Plot with verticle bar, can be used for data display.
    
            Note:
            According to ROOT: 
            "The error per bin will be computed as sqrt(sum of squares of weight) for each bin."

        """
        plt.ioff()
        # Check input
        data_1dim = np.array([])
        weight_1dim = np.array([])
        if type(datas) is list:
            for data, weight in zip(datas, weights):
                assert isinstance(data, np.ndarray), \
                  "datas element should be numpy array."
                assert isinstance(weight, np.ndarray), \
                  "weights element should be numpy array."
                assert data.shape == weight.shape, \
                  "Input weights should be None or have same type as arrays."
                data_1dim = np.concatenate((data_1dim, data))
                weight_1dim = np.concatenate((weight_1dim, weight))
        elif isinstance(datas, np.ndarray):
            assert isinstance(datas, np.ndarray), \
              "datas element should be numpy array."
            assert isinstance(weights, np.ndarray), \
              "weights element should be numpy array."
            assert datas.shape == weights.shape, \
              "Input weights should be None or have same type as arrays."
            data_1dim = datas
            weight_1dim = weights
        else:
            raise TypeError("Invalid arrays type.")

        # Scale x axis
        if x_scale is not None:
            data_1dim = data_1dim * x_scale

        # Make bar plot
        # get bin error and edges
        plot_ys, _ = np.histogram(data_1dim,
                                  bins=bins,
                                  range=range,
                                  weights=weight_1dim,
                                  density=density)
        sum_weight_squares, bin_edges = np.histogram(data_1dim,
                                                     bins=bins,
                                                     range=range,
                                                     weights=np.power(
                                                         weight_1dim, 2))
        errors = np.sqrt(sum_weight_squares)
        # Only plot ratio when bin is not 0.
        bin_centers = np.array([])
        bin_ys = np.array([])
        bin_yerrs = np.array([])
        for i, y1 in enumerate(plot_ys):
            if y1 != 0:
                ele_center = np.array(
                    [0.5 * (bin_edges[i] + bin_edges[i + 1])])
                bin_centers = np.concatenate((bin_centers, ele_center))
                ele_y = np.array([y1])
                bin_ys = np.concatenate((bin_ys, ele_y))
                ele_yerr = np.array([errors[i]])
                bin_yerrs = np.concatenate((bin_yerrs, ele_yerr))
        # plot bar
        bin_size = bin_edges[1] - bin_edges[0]
        if use_error:
            ax.errorbar(bin_centers,
                        bin_ys,
                        xerr=bin_size / 2.,
                        yerr=bin_yerrs,
                        fmt='.k',
                        label=labels)
        else:
            ax.errorbar(bin_centers,
                        bin_ys,
                        xerr=bin_size / 2.,
                        yerr=None,
                        fmt='.k',
                        label=labels)
        # Config
        if title is not None:
            ax.set_title(title)
        if x_lable is not None:
            if x_unit is not None:
                ax.set_xlabel(x_lable + '/' + x_unit)
            else:
                ax.set_xlabel(x_lable)
        else:
            if x_unit is not None:
                ax.set_xlabel(x_unit)
        if y_lable is not None:
            ax.set_ylabel(y_lable)
        if range is not None:
            ax.axis(xmin=range[0], xmax=range[1])
        ax.legend(loc="upper right")

    def plot_accuracy(self, ax):
        """Plots accuracy vs training epoch."""
        print("Plotting accuracy curve.")
        # Plot
        ax.plot(self.train_history_accuracy)
        ax.plot(self.train_history_val_accuracy)
        # Config
        ax.set_title('model accuracy')
        ax.set_ylabel('accuracy')
        #ax.set_ylim((0, 1))
        ax.set_xlabel('epoch')
        ax.legend(['train', 'val'], loc='lower left')
        ax.grid()

    def plot_auc_text(self, ax, titles, auc_values):
        """Plots auc information on roc curve."""
        auc_text = 'auc values:\n'
        for (title, auc_value) in zip(titles, auc_values):
            auc_text = auc_text + title + ": " + str(auc_value) + '\n'
        auc_text = auc_text[:-1]
        props = dict(boxstyle='round', facecolor='white', alpha=0.3)
        ax.text(0.5,
                0.6,
                auc_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props)

    def plot_correlation_matrix(self, ax, matrix_key="bkg"):
        corr_matrix_dict = self.get_corrcoef()
        # Get matrix
        corr_matrix = corr_matrix_dict[matrix_key]
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix,
                    mask=mask,
                    cmap=cmap,
                    vmax=.3,
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5},
                    ax=ax)

    def plot_feature_importance(self, ax):
        """Calculates importance of features and sort the feature.
    
        Definition of feature importance used here can be found in:
        https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data

        """
        print("Plotting feature importance.")
        # Prepare
        num_feature = len(self.selected_features)
        selected_feature_names = np.array(self.selected_features)
        feature_importance = np.zeros(num_feature)
        xs = self.xs_test_original_mass
        xb = self.xb_test_original_mass
        base_auc = self.calculate_auc(xs, xb, class_weight=self.class_weight)
        print("base auc:", base_auc)
        # Calculate importance
        for num, feature_name in enumerate(selected_feature_names):
            current_auc = self.calculate_auc(xs,
                                             xb,
                                             class_weight=self.class_weight,
                                             shuffle_col=num)
            #current_auc = 1
            feature_importance[num] = (1 - current_auc) / (1 - base_auc)
            print(feature_name, ":", feature_importance[num])
        # Sort
        sort_list = np.flip(np.argsort(feature_importance))
        sorted_importance = feature_importance[sort_list]
        sorted_names = selected_feature_names[sort_list]
        print("Feature importance rank:", sorted_names)
        # Plot
        if num_feature > 8:
            num_show = 8
        else:
            num_show = num_feature
        ax.bar(np.arange(num_show),
               sorted_importance[:num_show],
               align='center',
               alpha=0.5)
        ax.axhline(1, ls='--', color='r')
        ax.set_title("feature importance")
        ax.set_xticks(np.arange(num_show))
        ax.set_xticklabels(sorted_names[:num_show])

    def plot_final_roc(self, ax):
        """Plots roc curve for to check final training result on original samples.
        (backgound sample mass not reset according to signal)

        """
        print("Plotting final roc curve.")
        # Check
        if not self.model_is_trained:
            warnings.warn("Model is not trained yet.")
        # Make plots
        xs_plot = self.xs.copy()
        xb_plot = self.xb.copy()
        auc_value, _, _ = self.plot_roc(ax,
                                        xs_plot,
                                        xb_plot,
                                        class_weight=None)
        # Show auc value:
        self.plot_auc_text(ax, ['non-mass-reset auc'], [auc_value])
        # Extra plot config
        ax.grid()

    def plot_loss(self, ax):
        """Plots loss vs training epoch."""
        print("Plotting loss curve.")
        #Plot
        ax.plot(self.train_history_loss)
        ax.plot(self.train_history_val_loss)
        # Config
        ax.set_title('model loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(['train', 'val'], loc='lower left')
        ax.grid()

    def plot_roc(self, ax, xs, xb, class_weight=None):
        """Plots roc curve on given axes."""
        # Get data
        x_plot, y_plot, y_pred = self.process_array(xs,
                                                    xb,
                                                    class_weight=class_weight)
        fpr_dm, tpr_dm, _ = roc_curve(y_plot,
                                      y_pred,
                                      sample_weight=x_plot[:, -1])
        # Make plots
        ax.plot(fpr_dm, tpr_dm)
        ax.set_title("roc curve")
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.set_ylim(0.1, 1 - 1e-4)
        ax.set_yscale('logit')
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Calculate auc and return parameters
        auc_value = auc(fpr_dm, tpr_dm)
        return auc_value, fpr_dm, tpr_dm

    def plot_train_test_roc(self, ax):
        """Plots roc curve."""
        print("Plotting train/test roc curve.")
        # Check
        if not self.model_is_trained:
            warnings.warn("Model is not trained yet.")
        # First plot roc for train dataset
        auc_train, _, _ = self.plot_roc(ax, self.xs_train, self.xb_train)
        # Then plot roc for test dataset
        auc_test, _, _ = self.plot_roc(ax, self.xs_test, self.xb_test)
        # Then plot roc for train dataset without reseting mass
        auc_train_original, _, _ = self.plot_roc(ax,
                                                 self.xs_train_original_mass,
                                                 self.xb_train_original_mass)
        # Lastly, plot roc for test dataset without reseting mass
        auc_test_original, _, _ = self.plot_roc(ax, self.xs_test_original_mass,
                                                self.xb_test_original_mass)
        # Show auc value:
        self.plot_auc_text(
            ax, ['TV ', 'TE ', 'TVO', 'TEO'],
            [auc_train, auc_test, auc_train_original, auc_test_original])
        # Extra plot config
        ax.legend([
            'TV (train+val)', 'TE (test)', 'TVO (train+val original)',
            'TEO (test original)'
        ],
                  loc='lower right')
        ax.grid()

    def plot_test_scores(self,
                         ax,
                         title="test scores",
                         bins=50,
                         range=(-0.25, 1.25),
                         apply_data=False,
                         density=True,
                         log=True):
        """Plots training score distribution for siganl and background."""
        print("Plotting test scores.")
        self.plot_scores(ax,
                         self.xb_test_selected,
                         self.xb_test[:, -1],
                         self.xs_test_selected,
                         self.xs_test[:, -1],
                         selected_data=self.xd_selected,
                         data_weight=self.xd_norm[:, -1],
                         apply_data=apply_data,
                         title=title,
                         bins=bins,
                         range=range,
                         density=density,
                         log=log)

    def plot_test_scores_original_mass(self,
                                       ax,
                                       title="test scores (original mass)",
                                       bins=50,
                                       range=(-0.25, 1.25),
                                       apply_data=False,
                                       density=True,
                                       log=True):
        """Plots training score distribution for siganl and background."""
        print("Plotting test scores with original mass.")
        self.plot_scores(ax,
                         self.xb_test_selected_original_mass,
                         self.xb_test_original_mass[:, -1],
                         self.xs_test_selected_original_mass,
                         self.xs_test_original_mass[:, -1],
                         selected_data=self.xd_selected_original_mass.copy(),
                         data_weight=self.xd_norm[:, -1],
                         apply_data=apply_data,
                         title=title,
                         bins=bins,
                         range=range,
                         density=density,
                         log=log)

    def plot_train_scores(self,
                          ax,
                          title="train scores",
                          bins=50,
                          range=(-0.25, 1.25),
                          apply_data=False,
                          density=True,
                          log=True):
        """Plots training score distribution for siganl and background."""
        print("Plotting train scores.")
        self.plot_scores(ax,
                         self.xb_train_selected,
                         self.xb_train[:, -1],
                         self.xs_train_selected,
                         self.xs_train[:, -1],
                         selected_data=self.xd_selected,
                         data_weight=self.xd_norm[:, -1],
                         apply_data=apply_data,
                         title=title,
                         bins=bins,
                         range=range,
                         density=density,
                         log=log)

    def plot_train_scores_original_mass(self,
                                        ax,
                                        title="train scores (original mass)",
                                        bins=50,
                                        range=(-0.25, 1.25),
                                        apply_data=False,
                                        density=True,
                                        log=True):
        """Plots training score distribution for siganl and background."""
        print("Plotting train scores with original mass.")
        self.plot_scores(ax,
                         self.xb_train_selected_original_mass,
                         self.xb_train_original_mass[:, -1],
                         self.xs_train_selected_original_mass,
                         self.xs_train_original_mass[:, -1],
                         selected_data=self.xd_selected_original_mass.copy(),
                         data_weight=self.xd_norm[:, -1],
                         apply_data=apply_data,
                         title=title,
                         bins=bins,
                         range=range,
                         density=density,
                         log=log)

    def plot_scores(self,
                    ax,
                    selected_bkg,
                    bkg_weight,
                    selected_sig,
                    sig_weight,
                    selected_data=None,
                    data_weight=None,
                    apply_data=False,
                    title="scores",
                    bins=50,
                    range=(-0.25, 1.25),
                    density=True,
                    log=False):
        """Plots score distribution for siganl and background."""
        ax.hist(self.get_model().predict(selected_bkg),
                weights=bkg_weight,
                bins=bins,
                range=range,
                histtype='step',
                label='bkg',
                density=density,
                log=log)
        ax.hist(self.get_model().predict(selected_sig),
                weights=sig_weight,
                bins=bins,
                range=range,
                histtype='step',
                label='sig',
                density=density,
                log=log)
        if apply_data:
            """
      ax.hist(
      self.get_model().predict(selected_data),
      weights=data_weight,
      bins=bins, range=range, histtype='step', label='data', color="black",
      density=density, log=log)
      """
            self.make_bar_plot(ax,
                               self.get_model().predict(selected_data),
                               "data",
                               weights=np.reshape(data_weight, (-1, 1)),
                               bins=bins,
                               range=range,
                               density=density,
                               use_error=False)
        ax.set_title(title)
        ax.legend(loc='lower left')
        ax.set_xlabel("Output score")
        ax.set_ylabel("arb. unit")
        ax.grid()

    def plot_scores_separate(self,
                             ax,
                             bkg_dict,
                             bkg_plot_key_list,
                             selected_features,
                             sig_arr=None,
                             sig_weights=None,
                             apply_data=False,
                             data_arr=None,
                             data_weight=None,
                             plot_title='all input scores',
                             bins=50,
                             range=(-0.25, 1.25),
                             density=True,
                             log=False):
        """Plots training score distribution for different background.
    
        Note:
            bkg_plot_key_list can be used to adjust order of background sample 
            stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
            'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top

        """
        print("Plotting scores with bkg separated.")
        predict_arr_list = []
        predict_arr_weight_list = []
        # plot background
        for arr_key in bkg_plot_key_list:
            bkg_arr_temp = bkg_dict[arr_key].copy()
            bkg_arr_temp[:, 0:-2] = train_utils.norarray(
                bkg_arr_temp[:, 0:-2],
                average=self.norm_average,
                variance=self.norm_variance)
            selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
            predict_arr_list.append(
                np.array(self.get_model().predict(selected_arr)))
            predict_arr_weight_list.append(bkg_arr_temp[:, -1])
        try:
            ax.hist(np.transpose(predict_arr_list),
                    bins=bins,
                    range=range,
                    weights=np.transpose(predict_arr_weight_list),
                    histtype='bar',
                    label=bkg_plot_key_list,
                    density=density,
                    stacked=True)
        except:
            ax.hist(predict_arr_list[0],
                    bins=bins,
                    range=range,
                    weights=predict_arr_weight_list[0],
                    histtype='bar',
                    label=bkg_plot_key_list,
                    density=density,
                    stacked=True)
        # plot signal
        if sig_arr is None:
            sig_arr = self.xs_selected.copy()
            sig_weights = self.xs[:, -1]
        ax.hist(self.get_model().predict(sig_arr),
                bins=bins,
                range=range,
                weights=sig_weights,
                histtype='step',
                label='sig',
                density=density)
        # plot data
        if apply_data:
            if data_arr is None:
                data_arr = self.xd_selected_original_mass.copy()
                data_weight = self.xd[:, -1]
            self.make_bar_plot(ax,
                               self.get_model().predict(data_arr),
                               "data",
                               weights=np.reshape(data_weight, (-1, 1)),
                               bins=bins,
                               range=range,
                               density=density,
                               use_error=False)
        ax.set_title(plot_title)
        ax.legend(loc='upper right')
        ax.set_xlabel("Output score")
        ax.set_ylabel("arb. unit")
        ax.grid()
        if log is True:
            ax.set_yscale('log')
            ax.set_title(plot_title + "(log)")
        else:
            ax.set_title(plot_title + "(lin)")

    def plot_scores_separate_root(self,
                                  bkg_dict,
                                  bkg_plot_key_list,
                                  selected_features,
                                  sig_arr=None,
                                  sig_weights=None,
                                  apply_data=False,
                                  data_arr=None,
                                  data_weight=None,
                                  plot_title='all input scores',
                                  bins=50,
                                  range=(-0.25, 1.25),
                                  scale_sig=False,
                                  density=True,
                                  log_scale=False,
                                  save_plot=False,
                                  save_dir=None,
                                  save_file_name=None):
        """Plots training score distribution for different background with ROOT
    
        Note:
            bkg_plot_key_list can be used to adjust order of background sample 
            stacking. For example, if bkg_plot_key_list = ['top', 'zll', 'diboson']
            'top' will be put at bottom & 'zll' in the middle & 'diboson' on the top

        """
        print("Plotting scores with bkg separated with ROOT.")
        plot_canvas = ROOT.TCanvas(plot_title, plot_title, 800, 800)
        plot_pad_score = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        plot_pad_score.SetBottomMargin(0)
        plot_pad_score.SetGridx()
        plot_pad_score.Draw()
        plot_pad_score.cd()
        hist_list = []
        # plot background
        for arr_key in bkg_plot_key_list:
            bkg_arr_temp = bkg_dict[arr_key].copy()
            bkg_arr_temp[:, 0:-2] = train_utils.norarray(
                bkg_arr_temp[:, 0:-2],
                average=self.norm_average,
                variance=self.norm_variance)
            selected_arr = train_utils.get_valid_feature(bkg_arr_temp)
            predict_arr = np.array(self.get_model().predict(selected_arr))
            predict_weight_arr = bkg_arr_temp[:, -1]

            th1_temp = th1_tools.TH1FTool(arr_key,
                                          arr_key,
                                          nbin=bins,
                                          xlow=range[0],
                                          xup=range[1])
            th1_temp.fill_hist(predict_arr, predict_weight_arr)
            hist_list.append(th1_temp)
        hist_stacked_bkgs = th1_tools.THStackTool("bkg stack plot",
                                                  plot_title,
                                                  hist_list,
                                                  canvas=plot_pad_score)
        hist_stacked_bkgs.draw("pfc hist", log_scale=log_scale)
        hist_stacked_bkgs.get_hstack().GetYaxis().SetTitle("events/bin")
        # plot signal
        if sig_arr is None:
            selected_arr = self.xs_selected.copy()
            predict_arr = self.get_model().predict(selected_arr)
            predict_weight_arr = self.xs[:, -1]
        if scale_sig:
            sig_title = "sig"
        else:
            sig_title = "sig-scaled"
        hist_sig = th1_tools.TH1FTool("sig added",
                                      sig_title,
                                      nbin=bins,
                                      xlow=range[0],
                                      xup=range[1],
                                      canvas=plot_pad_score)
        hist_sig.fill_hist(predict_arr, predict_weight_arr)
        if scale_sig:
            total_weight = hist_stacked_bkgs.get_total_weights()
            scale_factor = total_weight / hist_sig.get_hist().GetSumOfWeights()
            hist_sig.get_hist().Scale(scale_factor)
        hist_sig.update_config("hist", "SetLineColor", ROOT.kRed)
        # set proper y range
        maximum_y = max(plot_utils.get_highest_bin_value(hist_list),
                        plot_utils.get_highest_bin_value(hist_sig))
        hist_stacked_bkgs.get_hstack().SetMaximum(1.2 * maximum_y)
        hist_stacked_bkgs.get_hstack().SetMinimum(0.1)
        hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelFont(43)
        hist_stacked_bkgs.get_hstack().GetYaxis().SetLabelSize(15)
        hist_sig.draw("same hist")
        # plot data if required
        if apply_data:
            if data_arr is None:
                selected_arr = self.xd_selected_original_mass.copy()
                predict_arr = self.get_model().predict(selected_arr)
                predict_weight_arr = self.xd[:, -1]
            hist_data = th1_tools.TH1FTool("data added",
                                           "data",
                                           nbin=bins,
                                           xlow=range[0],
                                           xup=range[1],
                                           canvas=plot_pad_score)
            hist_data.fill_hist(predict_arr, predict_weight_arr)
            hist_data.update_config("hist", "SetMarkerStyle", ROOT.kFullCircle)
            hist_data.update_config("hist", "SetMarkerColor", ROOT.kBlack)
            hist_data.update_config("hist", "SetMarkerSize", 0.8)
            hist_data.draw("same e1", log_scale=log_scale)
        else:
            hist_data = hist_sig
        hist_data.build_legend(0.4, 0.7, 0.6, 0.9)

        # ratio plot
        plot_canvas.cd()
        plot_pad_ratio = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
        plot_pad_ratio.SetTopMargin(0)
        plot_pad_ratio.SetGridx()
        plot_pad_ratio.Draw()
        plot_pad_ratio.cd()
        ## plot bkg error bar
        hist_bkg_total = hist_stacked_bkgs.get_added_hists()
        hist_bkg_err = hist_bkg_total.Clone()
        hist_bkg_err.Divide(hist_bkg_err.Clone())
        hist_bkg_err.SetDefaultSumw2()
        hist_bkg_err.SetMinimum(0.5)
        hist_bkg_err.SetMaximum(1.5)
        hist_bkg_err.SetStats(0)
        hist_bkg_err.SetTitle("")
        hist_bkg_err.SetFillColor(ROOT.kGray)
        hist_bkg_err.GetXaxis().SetTitle("DNN score")
        hist_bkg_err.GetXaxis().SetTitleSize(20)
        hist_bkg_err.GetXaxis().SetTitleFont(43)
        hist_bkg_err.GetXaxis().SetTitleOffset(4.)
        hist_bkg_err.GetXaxis().SetLabelFont(43)
        hist_bkg_err.GetXaxis().SetLabelSize(15)
        hist_bkg_err.GetYaxis().SetTitle("ratio")
        hist_bkg_err.GetYaxis().SetNdivisions(505)
        hist_bkg_err.GetYaxis().SetTitleSize(20)
        hist_bkg_err.GetYaxis().SetTitleFont(43)
        hist_bkg_err.GetYaxis().SetTitleOffset(1.55)
        hist_bkg_err.GetYaxis().SetLabelFont(43)
        hist_bkg_err.GetYaxis().SetLabelSize(15)
        hist_bkg_err.Draw("e3")
        ## plot base line
        base_line = ROOT.TF1("one", "1", 0, 1)
        base_line.SetLineColor(ROOT.kRed)
        base_line.Draw("same")
        ## plot ratio
        hist_ratio = hist_data.get_hist().Clone()
        hist_ratio.Divide(hist_bkg_total)
        hist_ratio.Draw("same")

        # save plot
        if save_plot:
            plot_canvas.SaveAs(save_dir + "/" + save_file_name + ".png")

    def prepare_array(self,
                      xs,
                      xb,
                      xd=None,
                      apply_data=False,
                      norm_array=True,
                      reset_mass=False,
                      reset_mass_name=None,
                      sig_weight=1000,
                      bkg_weight=1000,
                      data_weight=1000,
                      test_rate=0.2,
                      verbose=1):
        """Prepares array for training."""
        # normalize input variables if norm_array is True
        if norm_array:
            means, variances = train_utils.get_mean_var(xb[:, 0:-2],
                                                        axis=0,
                                                        weights=xb[:, -1])
            self.norm_average = means
            self.norm_variance = variances
            xs_norm_vars = xs.copy()
            xb_norm_vars = xb.copy()
            xs_norm_vars[:, 0:-2] = train_utils.norarray(xs_norm_vars[:, 0:-2],
                                                         average=means,
                                                         variance=variances)
            xb_norm_vars[:, 0:-2] = train_utils.norarray(xb_norm_vars[:, 0:-2],
                                                         average=means,
                                                         variance=variances)
            self.xs = xs_norm_vars
            self.xb = xb_norm_vars
        else:
            self.xs = xs.copy()
            self.xb = xb.copy()
        rdm_seed = int(time.time())
        # get bkg array with mass reset
        if reset_mass:
            reset_mass_id = self.selected_features.index(reset_mass_name)
            self.xb_reset_mass = array_utils.modify_array(
                self.xb,
                reset_mass=reset_mass,
                reset_mass_array=self.xs,
                reset_mass_id=reset_mass_id)
            self.is_mass_reset = True
        else:
            self.xb_reset_mass = self.xb
            self.is_mass_reset = False
        # normalize total weight
        self.xs_norm = array_utils.modify_array(self.xs,
                                                norm=True,
                                                sumofweight=sig_weight)
        self.xb_norm = array_utils.modify_array(self.xb,
                                                norm=True,
                                                sumofweight=bkg_weight)
        self.xb_norm_reset_mass = array_utils.modify_array(
            self.xb_reset_mass, norm=True, sumofweight=bkg_weight)
        # get train/test data set, split with ratio=test_rate
        self.x_train, self.x_test, self.y_train, self.y_test,\
          self.xs_train, self.xs_test, self.xb_train, self.xb_test =\
          train_utils.split_and_combine(self.xs_norm, self.xb_norm_reset_mass,
          test_rate=test_rate, shuffle_seed=rdm_seed)
        self.x_train_original_mass, self.x_test_original_mass,\
          self.y_train_original_mass, self.y_test_original_mass,\
          self.xs_train_original_mass, self.xs_test_original_mass,\
          self.xb_train_original_mass, self.xb_test_original_mass =\
          train_utils.split_and_combine(self.xs_norm, self.xb_norm,
            test_rate=test_rate, shuffle_seed=rdm_seed)
        # select features used for training
        self.x_train_selected = train_utils.get_valid_feature(self.x_train)
        self.x_test_selected = train_utils.get_valid_feature(self.x_test)
        self.xs_train_selected = train_utils.get_valid_feature(self.xs_train)
        self.xb_train_selected = train_utils.get_valid_feature(self.xb_train)
        self.xs_test_selected = train_utils.get_valid_feature(self.xs_test)
        self.xb_test_selected = train_utils.get_valid_feature(self.xb_test)
        self.xs_selected = train_utils.get_valid_feature(self.xs_norm)
        self.xb_selected = train_utils.get_valid_feature(
            self.xb_norm_reset_mass)
        self.x_train_selected_original_mass = train_utils.get_valid_feature(
            self.x_train_original_mass)
        self.x_test_selected_original_mass = train_utils.get_valid_feature(
            self.x_test_original_mass)
        self.xs_train_selected_original_mass = train_utils.get_valid_feature(
            self.xs_train_original_mass)
        self.xb_train_selected_original_mass = train_utils.get_valid_feature(
            self.xb_train_original_mass)
        self.xs_test_selected_original_mass = train_utils.get_valid_feature(
            self.xs_test_original_mass)
        self.xb_test_selected_original_mass = train_utils.get_valid_feature(
            self.xb_test_original_mass)
        self.xs_selected_original_mass = train_utils.get_valid_feature(
            self.xs_norm)
        self.xb_selected_original_mass = train_utils.get_valid_feature(
            self.xb_norm)
        # prepare data to apply model when apply_data is True
        if apply_data == True:
            if norm_array:
                xd_norm_vars = xd.copy()
                xd_norm_vars[:, 0:-2] = train_utils.norarray(
                    xd_norm_vars[:, 0:-2],
                    average=self.norm_average,
                    variance=self.norm_variance)
                self.xd = xd_norm_vars
            else:
                self.xd = xd
            if reset_mass:
                reset_mass_id = self.selected_features.index(reset_mass_name)
                self.xd_reset_mass = array_utils.modify_array(
                    self.xd,
                    reset_mass=reset_mass,
                    reset_mass_array=self.xs,
                    reset_mass_id=reset_mass_id)
            else:
                self.xd_reset_mass = xd
            self.xd_norm = array_utils.modify_array(self.xd,
                                                    norm=True,
                                                    sumofweight=data_weight)
            self.xd_norm_reset_mass = array_utils.modify_array(
                self.xd_reset_mass, norm=True, sumofweight=data_weight)
            self.xd_selected = train_utils.get_valid_feature(
                self.xd_norm_reset_mass)
            self.xd_selected_original_mass = train_utils.get_valid_feature(
                self.xd_norm)
            self.has_data = True
        self.array_prepared = True
        if verbose == 1:
            print("Training array prepared.")
            print("> signal shape:", self.xs_selected.shape)
            print("> background shape:", self.xb_selected.shape)

    def process_array(self, xs, xb, class_weight=None, shuffle_col=None):
        """Process sig/bkg arrays in the same way for training arrays."""
        # Get data
        xs_proc = xs.copy()
        xb_proc = xb.copy()
        if class_weight is not None:
            xs_proc[:, -1] = xs_proc[:, -1] * class_weight[1]
            xb_proc[:, -1] = xb_proc[:, -1] * class_weight[0]
        x_proc = np.concatenate((xs_proc, xb_proc))
        if shuffle_col is not None:
            x_proc = array_utils.reset_col(x_proc, x_proc, shuffle_col)
        x_proc_selected = train_utils.get_valid_feature(x_proc)
        y_proc = np.concatenate(
            (np.ones(xs_proc.shape[0]), np.zeros(xb_proc.shape[0])))
        y_pred = self.get_model().predict(x_proc_selected)
        return x_proc, y_proc, y_pred

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
            file_name = self.model_name + '_' + self.model_label + '_' + datestr
        # Check path
        path_pattern = save_dir + '/' + file_name + '_v{}.h5'
        save_path = common_utils.get_newest_file_version(path_pattern)['path']
        version_id = common_utils.get_newest_file_version(
            path_pattern)['ver_num']
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        # Save
        self.model.save(save_path)
        self.model_save_path = save_path
        print("model:", self.model_name, "has been saved to:", save_path)
        # update path for json
        path_pattern = save_dir + '/' + file_name + '_v{}_paras.json'
        save_path = common_utils.get_newest_file_version(
            path_pattern, ver_num=version_id)['path']
        self.save_model_paras(save_path)
        print("model parameters has been saved to:", save_path)
        self.model_is_saved = True

    def save_model_paras(self, save_path):
        """Save model parameters to json file."""
        # sorted by aphabet
        paras_dict = {}
        paras_dict['class_weight'] = self.class_weight
        paras_dict['model_create_time'] = self.model_create_time
        paras_dict['model_decay'] = self.model_decay
        paras_dict['model_input_dim'] = self.model_input_dim
        paras_dict['model_is_compiled'] = self.model_is_compiled
        paras_dict['model_is_saved'] = self.model_is_saved
        paras_dict['model_is_trained'] = self.model_is_trained
        paras_dict['model_label'] = self.model_label
        paras_dict['model_learn_rate'] = self.model_learn_rate
        paras_dict['model_name'] = self.model_name
        paras_dict['model_note'] = self.model_note
        paras_dict['model_num_node'] = self.model_num_node
        paras_dict['train_history_accuracy'] = self.train_history_accuracy
        paras_dict[
            'train_history_val_accuracy'] = self.train_history_val_accuracy
        paras_dict['train_history_loss'] = self.train_history_loss
        paras_dict['train_history_val_loss'] = self.train_history_val_loss
        with open(save_path, 'w') as write_file:
            json.dump(paras_dict, write_file, indent=2)

    def show_input_distributions(self,
                                 apply_data=False,
                                 figsize=(8, 6),
                                 style_cfg_path=None,
                                 save_fig=False,
                                 save_dir=None,
                                 save_format="png"):
        """Plots input distributions comparision plots for sig/bkg/data"""
        print("Plotting input distributions.")
        config = {}
        if style_cfg_path is not None:
            with open(style_cfg_path) as plot_config_file:
                config = json.load(plot_config_file)

        for feature_id, feature in enumerate(self.selected_features):
            # prepare background histogram
            hist_bkg = th1_tools.TH1FTool(feature + "_bkg",
                                          feature + "_bkg",
                                          nbin=100,
                                          xlow=-20,
                                          xup=20)
            hist_bkg.fill_hist(
                np.reshape(self.xb_selected_original_mass[:, feature_id],
                           (-1, 1)), np.reshape(self.xb_norm[:, -1], (-1, 1)))
            hist_bkg.set_config(config)
            hist_bkg.update_config("hist", "SetLineColor", 4)
            hist_bkg.apply_config()
            hist_bkg.draw()
            hist_bkg.save(save_dir=save_dir,
                          save_file_name=feature + "_bkg",
                          save_format=save_format)
            # prepare signal histogram
            hist_sig = th1_tools.TH1FTool(feature + "_sig",
                                          feature + "_sig",
                                          nbin=100,
                                          xlow=-20,
                                          xup=20)
            hist_sig.fill_hist(
                np.reshape(self.xs_selected_original_mass[:, feature_id],
                           (-1, 1)), np.reshape(self.xs_norm[:, -1], (-1, 1)))
            hist_sig.set_config(config)
            hist_sig.update_config("hist", "SetLineColor", 2)
            hist_sig.apply_config()
            hist_sig.draw()
            hist_sig.save(save_dir=save_dir,
                          save_file_name=feature + "_sig",
                          save_format=save_format)
            # prepare sig vs bkg comparison plots
            hist_col = th1_tools.HistCollection([hist_bkg, hist_sig],
                                                name=feature,
                                                title=feature)
            hist_col.draw(config_str="hist", draw_norm=True)
            hist_col.save(save_dir=save_dir,
                          save_file_name=feature,
                          save_format=save_format)

    def show_performance(self,
                         apply_data=False,
                         figsize=(16, 24),
                         show_fig=True,
                         save_fig=False,
                         save_path=None,
                         job_type="train"):
        """Shortly reports training result.

        Args:
            figsize: tuple
                Defines plot size.

        """
        # Check input
        assert isinstance(self, model_base)
        print("Model performance:")
        # Plots
        if job_type == "train" and self.is_mass_reset == True:
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=figsize)
            self.plot_accuracy(ax[0, 0])
            self.plot_loss(ax[0, 1])
            self.plot_train_test_roc(ax[1, 0])
            self.plot_feature_importance(ax[1, 1])
            #following plots shoud use density plot because bkg (train/test) arrays
            # are part of full arrays and data arrays are full arrays
            self.plot_train_scores(ax[2, 0],
                                   bins=50,
                                   apply_data=apply_data,
                                   density=True)
            self.plot_test_scores(ax[2, 1],
                                  bins=50,
                                  apply_data=apply_data,
                                  density=True)
            self.plot_train_scores_original_mass(ax[3, 0],
                                                 bins=50,
                                                 apply_data=apply_data,
                                                 density=True)
            self.plot_test_scores_original_mass(ax[3, 1],
                                                bins=50,
                                                apply_data=apply_data,
                                                density=True)
        else:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
            self.plot_accuracy(ax[0, 0])
            self.plot_loss(ax[0, 1])
            self.plot_train_test_roc(ax[1, 0])
            self.plot_feature_importance(ax[1, 1])
            self.plot_train_scores_original_mass(ax[2, 0],
                                                 bins=50,
                                                 apply_data=apply_data)
            self.plot_test_scores_original_mass(ax[2, 1],
                                                bins=50,
                                                apply_data=apply_data)

        fig.tight_layout()
        if show_fig:
            plt.show()
        else:
            print("(Plots-show skipped according to settings)")
        if save_fig:
            fig.savefig(save_path)

    def train(self):
        pass


class Model_1002(model_sequential):
    """Sequential model optimized with old ntuple at Sep. 9th 2019.
  
    Major modification based on 0913 model:
        1. Optimized to train on full mass range. (Used to be on bkg samples with
        cut to have similar mass range as signal.)
        2. Use normalized data for training.
    
    """
    def __init__(self,
                 name,
                 input_dim,
                 num_node=400,
                 learn_rate=0.02,
                 decay=1e-6,
                 metrics=['plain_acc'],
                 weighted_metrics=['accuracy'],
                 selected_features=[],
                 save_tb_logs=False,
                 tb_logs_path=None,
                 use_early_stop=False,
                 early_stop_paras={}):
        super().__init__(name,
                         input_dim,
                         num_node=num_node,
                         learn_rate=learn_rate,
                         decay=decay,
                         metrics=metrics,
                         weighted_metrics=weighted_metrics,
                         selected_features=selected_features)
        self.model_label = "mod1002"
        self.model_note = "Sequential model optimized with old ntuple"\
                          + " at Oct. 2rd 2019"\
                          + " to deal with training with full bkg mass."
        self.save_tb_logs = save_tb_logs
        self.tb_logs_path = tb_logs_path
        self.use_early_stop = use_early_stop
        self.early_stop_paras = early_stop_paras

    def compile(self):
        """ Compile model, function to be changed in the future."""
        # Add layers
        # input
        self.model.add(
            Dense(self.model_num_node,
                  kernel_initializer='uniform',
                  input_dim=self.model_input_dim))

        # hidden 1
        #self.model.add(BatchNormalization())
        self.model.add(
            Dense(self.model_num_node,
                  kernel_initializer="glorot_normal",
                  activation="relu"))
        # hidden 2
        #self.model.add(BatchNormalization())
        self.model.add(
            Dense(self.model_num_node,
                  kernel_initializer="glorot_normal",
                  activation="relu"))
        # hidden 3
        #self.model.add(BatchNormalization())
        self.model.add(
            Dense(self.model_num_node,
                  kernel_initializer="glorot_normal",
                  activation="relu"))

        # hidden 4
        #self.model.add(BatchNormalization())
        #self.model.add(Dense(self.model_num_node,
        #                     kernel_initializer="glorot_normal",
        #                     activation="relu"))
        # hidden 5
        #self.model.add(BatchNormalization())
        #self.model.add(Dense(self.model_num_node,
        #                     kernel_initializer="glorot_normal",
        #                     activation="relu"))

        # output
        #self.model.add(BatchNormalization())
        self.model.add(
            Dense(1, kernel_initializer="glorot_uniform",
                  activation="sigmoid"))
        # Compile
        self.model.compile(loss="binary_crossentropy",
                           optimizer=SGD(lr=self.model_learn_rate,
                                         decay=self.model_decay),
                           metrics=self.metrics,
                           weighted_metrics=self.weighted_metrics)
        self.model_is_compiled = True

    def train(
        self,
        batch_size=128,
        epochs=20,
        val_split=0.25,
        sig_class_weight=1.,
        bkg_class_weight=1.,
        verbose=1,
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
                warnings.warn("TensorBoard logs path not specified, \
          set path to: {}".format(self.tb_logs_path))
            tb_callback = TensorBoard(log_dir=self.tb_logs_path,
                                      histogram_freq=1)
            train_callbacks.append(tb_callback)
        if self.use_early_stop:
            early_stop_callback = callbacks.EarlyStopping(
                monitor=self.early_stop_paras["monitor"],
                min_delta=self.early_stop_paras["min_delta"],
                patience=self.early_stop_paras["patience"],
                mode=self.early_stop_paras["mode"],
                restore_best_weights=self.
                early_stop_paras["restore_best_weights"])
            train_callbacks.append(early_stop_callback)
        self.train_history = self.get_model().fit(
            self.x_train_selected,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            class_weight=self.class_weight,
            sample_weight=self.x_train[:, -1],
            callbacks=train_callbacks,
            verbose=verbose)
        print("Training finished.")
        # Quick evaluation
        print("Quick evaluation:")
        score = self.get_model().evaluate(self.x_test_selected,
                                          self.y_test,
                                          verbose=verbose,
                                          sample_weight=self.x_test[:, -1])
        print('> test loss:', score[0])
        print('> test accuracy:', score[1])

        # Save train history
        # save accuracy history
        self.train_history_accuracy = [
            float(ele) for ele in self.train_history.history['accuracy']
        ]
        try:
            self.train_history_accuracy = [float(ele) for ele in\
              self.train_history.history['acc']]
            self.train_history_val_accuracy = [float(ele) for ele in\
              self.train_history.history['val_acc']]
        except:  # updated for tensorflow2.0
            self.train_history_accuracy = [float(ele) for ele in\
              self.train_history.history['accuracy']]
            self.train_history_val_accuracy = [float(ele) for ele in\
              self.train_history.history['val_accuracy']]
        # save loss history/
        self.train_history_loss = [float(ele) for ele in\
            self.train_history.history['loss']]
        self.train_history_val_loss = [float(ele) for ele in\
            self.train_history.history['val_loss']]

        self.model_is_trained = True


class Model_1016(Model_1002):
    """Sequential model optimized with old ntuple at Sep. 9th 2019.
  
    Major modification based on 1002 model:
        1. Change structure to make quantity of nodes decrease with layer num.
    
    """
    def __init__(self,
                 name,
                 input_dim,
                 num_node=400,
                 learn_rate=0.02,
                 decay=1e-6,
                 dropout_rate=0.3,
                 metrics=['plain_acc'],
                 weighted_metrics=['accuracy'],
                 selected_features=[],
                 save_tb_logs=False,
                 tb_logs_path=None,
                 use_early_stop=False,
                 early_stop_paras={}):
        super().__init__(name,
                         input_dim,
                         num_node=num_node,
                         learn_rate=learn_rate,
                         decay=decay,
                         metrics=metrics,
                         weighted_metrics=weighted_metrics,
                         selected_features=selected_features,
                         save_tb_logs=save_tb_logs,
                         tb_logs_path=tb_logs_path,
                         use_early_stop=use_early_stop,
                         early_stop_paras=early_stop_paras)
        self.model_label = "mod1016"
        self.model_note = "New model structure based on 1002's model." \
          + "Created at Oct. 16th 2019 to deal with training with full bkg mass." \
          + "Modified at Mar. 20th 2020 to add dropout layers."
        self.dropout_rate = dropout_rate

    def compile(self):
        """ Compile model, function to be changed in the future."""
        # Add layers
        # input
        self.model.add(
            Dense(100,
                  kernel_initializer='uniform',
                  input_dim=self.model_input_dim))
        self.model.add(Dropout(self.dropout_rate))
        # hidden 1
        self.model.add(
            Dense(100, kernel_initializer="glorot_normal", activation="relu"))
        self.model.add(Dropout(self.dropout_rate))
        # hidden 2
        self.model.add(
            Dense(100, kernel_initializer="glorot_normal", activation="relu"))
        self.model.add(Dropout(self.dropout_rate))
        # output
        self.model.add(
            Dense(1, kernel_initializer="glorot_uniform",
                  activation="sigmoid"))
        # Compile
        self.model.compile(loss="binary_crossentropy",
                           optimizer=SGD(lr=self.model_learn_rate,
                                         decay=self.model_decay,
                                         momentum=0.5,
                                         nesterov=True),
                           metrics=self.metrics,
                           weighted_metrics=self.weighted_metrics)
        self.model_is_compiled = True
