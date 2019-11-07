# -*- coding: utf-8 -*-
"""Model class for DNN training"""
import datetime
import glob
import json
import os
import time
import warnings

import eli5
from eli5.sklearn import PermutationImportance
import keras
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, Input, Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, SGD, RMSprop, Adam
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import matplotlib.pyplot as plot_test_scores
from matplotlib.ticker import NullFormatter, FixedLocator
import numpy as np
from sklearn.metrics import roc_curve, auc

from ..common.common_utils import *
from . import train_utils 
from .train_utils import *

# self-difined metrics functions
def plain_acc(y_true, y_pred):
  return K.mean(K.less(K.abs(y_pred*1. - y_true*1.), 0.5))
  #return 1-K.mean(K.abs(y_pred-y_true))

class model_base(object):
  """Base model of deep neural network for pdnn training.
  
  Attributes:
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
      Column id of input array of features that will be used for training.
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
    >>> selected_features = [1, 3, 5, 7, 9]
    >>> model_deep = model.model_0913(model_name, len(selected_features))
    Then compile model:
    >>> model_deep.compile()
    Prepare array for training:
    >>> xs_emu = np.load('path/to/numpy/signal/array.npy')
    >>> xb_emu = np.load('path/to/numpy/background/array.npy')
    >>> model_deep.prepare_array(xs_emu, xb_emu, selected_features)
    Perform training:
    >>> model_deep.train(epochs = epochs, val_split = 0.1, verbose = 0)
    Make plots to shoe training performance:
    >>> model_deep.show_performance()
  
  """

  def __init__(
    self, name,
    input_dim, num_node=300,
    learn_rate=0.025, 
    decay=1e-6,
    metrics=['plain_acc'],
    weighted_metrics=['accuracy']
    ):
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
    self.x_train = np.array([])
    self.x_test = np.array([])
    self.y_train = np.array([])
    self.y_test = np.array([])
    self.xs_test = np.array([])
    self.xb_test = np.array([])
    self.selected_features = []
    self.x_train_selected = np.array([])
    self.x_test_selected = np.array([])
    self.xs_test_selected = np.array([])
    self.xb_test_selected = np.array([])
    self.xs_selected = np.array([])
    self.xb_selected = np.array([])
    self.xs_train_original_mass = np.array([])
    self.xs_test_original_mass = np.array([])
    self.xb_train_original_mass = np.array([])
    self.xb_test_original_mass = np.array([])
    self.xs_train_original_mass = np.array([])
    self.xs_test_original_mass = np.array([])
    self.xb_train_original_mass = np.array([])
    self.xb_test_original_mass = np.array([])
  
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

  def load_model(self, model_name, dir='models', model_class='*', date='*',
                 version='*'):
    """Loads saved model."""
    # Search possible files
    search_pattern = dir + '/' + model_name + '_' + model_class + '_' + date\
                     + '*' + version + '.h5'
    model_path_list = glob.glob(search_pattern)
    # Choose the newest one
    if len(model_path_list) < 1:
      raise FileNotFoundError("Model file that matched the pattern not found.")
    model_path = model_path_list[-1]
    if len(model_path_list) > 1:
      print("More than one valid model file found, try to specify more infomation.")
      print("Loading the last matched model path:", model_path)
    else:
      print("Loading model at:", model_path)
    self.model = keras.models.load_model(model_path,
      custom_objects={'plain_acc': plain_acc})  # it's important to specify
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
    self.class_weight = dict_key_strtoint(paras_dict['class_weight'])
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
    self.train_history_val_accuracy = paras_dict['train_history_val_accuracy']
    self.train_history_loss = paras_dict['train_history_loss']
    self.train_history_val_loss = paras_dict['train_history_val_loss']

  def plot_accuracy(self, ax):
    """Plots accuracy vs training epoch."""
    # Plot
    ax.plot(self.train_history_accuracy)
    ax.plot(self.train_history_val_accuracy)
    # Config
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_ylim((0, 1))
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
    ax.text(0.5, 0.6, auc_text, transform=ax.transAxes, fontsize=10,
      verticalalignment='top', bbox=props)

  def plot_final_roc(self, ax):
    """Plots roc curve for to check final training result on original samples.
    (backgound sample mass not reset according to signal)

    """
     # Check
    if not self.model_is_trained:
      warnings.warn("Model is not trained yet.")
    # Make plots
    xs_plot = self.xs.copy()
    xb_plot = self.xb.copy()
    auc_value, _, _ = self.plot_roc(
      ax, xs_plot, xb_plot,
      class_weight=None
      )
    # Show auc value:
    self.plot_auc_text(ax, ['non-mass-reset auc'], [auc_value])
    # Extra plot config
    ax.grid()

  def plot_loss(self, ax):
    """Plots loss vs training epoch."""
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
    xs_plot = xs.copy()
    xb_plot = xb.copy()
    if class_weight is not None:
      xs_plot[:,-1] = xs_plot[:,-1] * class_weight[1]
      xb_plot[:,-1] = xb_plot[:,-1] * class_weight[0]
    xs_selected = get_part_feature(xs, self.selected_features)
    xb_selected = get_part_feature(xb, self.selected_features)
    x_plot = np.concatenate((xs_plot, xb_plot))
    x_plot_selected = np.concatenate((xs_selected, xb_selected))
    y_plot = np.concatenate((np.ones(xs_plot.shape[0]), np.zeros(xb_plot.shape[0])))
    y_pred = self.get_model().predict(x_plot_selected)
    fpr_dm, tpr_dm, threshold = roc_curve(y_plot, y_pred,
      sample_weight=x_plot[:,-1])
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
    # Check
    if not self.model_is_trained:
      warnings.warn("Model is not trained yet.")
    # First plot roc for train dataset
    auc_train, _, _ = self.plot_roc(ax, self.xs_train, self.xb_train)
    # Then plot roc for test dataset
    auc_test, _, _ = self.plot_roc(ax, self.xs_test, self.xb_test)
    # Then plot roc for train dataset without reseting mass
    auc_train_original, _, _ = self.plot_roc(
      ax, self.xs_train_original_mass, self.xb_train_original_mass
      )
    # Lastly, plot roc for test dataset without reseting mass
    auc_test_original, _, _ = self.plot_roc(
      ax, self.xs_test_original_mass, self.xb_test_original_mass
    )
    # Show auc value:
    self.plot_auc_text(
      ax,
      ['TV ', 'TE ', 'TVO', 'TEO'],
      [auc_train, auc_test, auc_train_original, auc_test_original]
      )
    # Extra plot config
    ax.legend(
      ['TV (train+val)', 'TE (test)', 'TVO (train+val original)', 'TEO (test original)'],
      loc='lower right'
      )
    ax.grid()

  def plot_test_scores(
    self, ax,
    bins=100, range=(-0.25, 1.25), density=True, log=True
    ):
    """Plots training score distribution for siganl and background."""
    self.plot_scores(
      ax,
      self.xb_test_selected, self.xb_test[:, -1],
      self.xs_test_selected, self.xs_test[:, -1],
      title = "test scores", bins=bins, range=range,
      density=density, log=log
    )

  def plot_test_scores_original_mass(
    self, ax,
    title = "test scores (original mass)", bins=100, range=(-0.25, 1.25),
    density=True, log=True
    ):
    """Plots training score distribution for siganl and background."""
    self.plot_scores(
      ax,
      self.xb_test_selected_original_mass, self.xb_test_original_mass[:, -1],
      self.xs_test_selected_original_mass, self.xs_test_original_mass[:, -1],
      bins=bins, range=range, density=density, log=log
    )

  def plot_train_scores(
    self, ax,
    title = "train scores", bins=100, range=(-0.25, 1.25),
    density=True, log=True
    ):
    """Plots training score distribution for siganl and background."""
    self.plot_scores(
      ax,
      self.xb_train_selected, self.xb_train[:, -1],
      self.xs_train_selected, self.xs_train[:, -1],
      bins=bins, range=range, density=density, log=log
    )

  def plot_train_scores_original_mass(
    self, ax,
    title = "train scores (original mass)", bins=100, range=(-0.25, 1.25),
    density=True, log=True
    ):
    """Plots training score distribution for siganl and background."""
    self.plot_scores(
      ax,
      self.xb_train_selected_original_mass, self.xb_train_original_mass[:, -1],
      self.xs_train_selected_original_mass, self.xs_train_original_mass[:, -1],
     bins=bins, range=range, density=density, log=log
    )

  def plot_scores(
    self, ax,
    selected_bkg, bkg_weight,
    selected_sig, sig_weight,
    title = "scores", bins=100, range=(-0.25, 1.25),
    density=True, log=False
    ):
    """Plots score distribution for siganl and background."""
    ax.hist(
      self.get_model().predict(selected_bkg),
      weights=bkg_weight,
      bins=bins, range=range, histtype='step', label='bkg',
      density=True, log=log
      )
    ax.hist(
      self.get_model().predict(selected_sig),
      weights=sig_weight,
      bins=bins, range=range, histtype='step', label='sig',
      density=True, log=log
      )
    ax.set_title('train scores')
    ax.legend(loc='lower left')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()

  def plot_scores_separate(self, ax, arr_dict, key_list, selected_features,
                           sig_arr=None, sig_weights=None, plot_title='all input scores',
                           bins=40, range=(-0.25, 1.25), density=True, log=False):
    """Plots training score distribution for different background."""
    predict_arr_list = []
    predict_arr_weight_list = []

    average=self.norm_average
    variance=self.norm_variance

    for arr_key in key_list:
      bkg_arr_temp = arr_dict[arr_key].copy()
      average=self.norm_average
      variance=self.norm_variance
      #average, variance = get_mean_var(bkg_arr_temp[:, 0:-2], axis=0, weights=bkg_arr_temp[:, -1])
      bkg_arr_temp[:, 0:-2] = norarray(bkg_arr_temp[:, 0:-2], average=average, variance=variance)

      selected_arr = get_part_feature(bkg_arr_temp, selected_features)
      #print("debug", self.get_model().predict(selected_arr))
      predict_arr_list.append(np.array(self.get_model().predict(selected_arr)))
      predict_arr_weight_list.append(bkg_arr_temp[:, -1])

    try:
      ax.hist(np.transpose(predict_arr_list), bins=bins, 
              range=range, weights=np.transpose(predict_arr_weight_list), histtype='bar', label=key_list, density=True, 
              stacked=True)
    except:
      ax.hist(predict_arr_list[0], bins=bins, 
              range=range, weights=predict_arr_weight_list[0], histtype='bar', label=key_list, density=True, 
              stacked=True)

    if sig_arr is None:
      sig_arr=self.xs_selected.copy()
      sig_weights=self.xs[:,-1]

    ax.hist(self.get_model().predict(sig_arr), bins=bins, range=range,
            weights=sig_weights, histtype='step', label='sig', density=True)
    
    ax.set_title(plot_title)
    ax.legend(loc='upper center')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    if log is True:
      ax.set_yscale('log')
      ax.set_title(plot_title+"(log)")
    else:
      ax.set_title(plot_title+"(lin)")

  def prepare_array(self, xs, xb, selected_features, norm_array=True,
                    sig_weight=1000, bkg_weight=1000, test_rate=0.2,
                    verbose=1):
    """Prepares array for training."""
    self.xs = xs
    self.xb = xb
    rdm_seed = int(time.time())
    # get bkg array with mass reset
    xb_reset_mass = modify_array(xb, weight_id=-1, reset_mass=True,
                      reset_mass_array=xs, reset_mass_id=0)
    # normalize total weight
    self.xs_norm = modify_array(xs, weight_id=-1, norm=True,
                                sumofweight=sig_weight)
    self.xb_norm = modify_array(xb, weight_id=-1, norm=True,
                                sumofweight=bkg_weight)
    self.xb_norm_reset_mass = modify_array(xb_reset_mass, weight_id=-1,
                                norm=True, sumofweight=bkg_weight)
    # get train/test data set, split with ratio=test_rate
    self.x_train, self.x_test, self.y_train, self.y_test,\
    self.xs_train, self.xs_test, self.xb_train, self.xb_test =\
    split_and_combine(self.xs_norm, self.xb_norm_reset_mass,
    test_rate=test_rate, shuffle_seed=rdm_seed)
    self.xs_train_original_mass, self.xs_test_original_mass,\
      self.xb_train_original_mass, self.xb_test_original_mass,\
      self.xs_train_original_mass, self.xs_test_original_mass,\
      self.xb_train_original_mass, self.xb_test_original_mass =\
      split_and_combine(self.xs_norm, self.xb_norm,
        test_rate=test_rate, shuffle_seed=rdm_seed)
    # select features used for training
    self.selected_features = selected_features
    self.x_train_selected = get_part_feature(self.x_train, selected_features)
    self.x_test_selected = get_part_feature(self.x_test, selected_features)
    self.xs_train_selected = get_part_feature(self.xs_train, selected_features)
    self.xb_train_selected = get_part_feature(self.xb_train, selected_features)
    self.xs_test_selected = get_part_feature(self.xs_test, selected_features)
    self.xb_test_selected = get_part_feature(self.xb_test, selected_features)
    self.xs_selected = get_part_feature(self.xs, selected_features)
    self.xb_selected = get_part_feature(self.xb, selected_features)
    self.x_train_selected_original_mass = get_part_feature(self.x_train, selected_features)
    self.x_test_selected_original_mass = get_part_feature(self.x_test, selected_features)
    self.xs_train_selected_original_mass = get_part_feature(self.xs_train, selected_features)
    self.xb_train_selected_original_mass = get_part_feature(self.xb_train, selected_features)
    self.xs_test_selected_original_mass = get_part_feature(self.xs_test, selected_features)
    self.xb_test_selected_original_mass = get_part_feature(self.xb_test, selected_features)
    self.xs_selected_original_mass = get_part_feature(self.xs, selected_features)
    self.xb_selected_original_mass = get_part_feature(self.xb, selected_features)

    self.array_prepared = True
    if verbose == 1:
      print("Training array prepared.")
      print("> signal shape:", self.xs_selected.shape)
      print("> background shape:", self.xb_selected.shape)

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
    save_path = get_newest_file_version(path_pattern)['path']
    version_id = get_newest_file_version(path_pattern)['ver_num']
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))

    # Save
    self.model.save(save_path)
    self.model_save_path = save_path
    print("model:", self.model_name, "has been saved to:", save_path)
    # update path for json
    path_pattern = save_dir + '/' + file_name + '_v{}_paras.json'
    save_path = get_newest_file_version(
      path_pattern,
      ver_num=version_id
      )['path']
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
    paras_dict['train_history_val_accuracy'] = self.train_history_val_accuracy
    paras_dict['train_history_loss'] = self.train_history_loss
    paras_dict['train_history_val_loss'] = self.train_history_val_loss
    with open(save_path, 'w') as write_file:
      json.dump(paras_dict, write_file, indent=2)

  def show_performance(
    self, figsize=(16, 12), show_fig=True,
    save_fig=False, save_path=None
    ):
    """Shortly reports training result.

    Args:
      figsize: tuple
        Defines plot size.

    """
    # Check input
    assert isinstance(self, model_base)
    print("Model performance:")
    # Plots
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    self.plot_accuracy(ax[0, 0])
    self.plot_loss(ax[0, 1])
    self.plot_train_scores(ax[1, 0])
    self.plot_test_scores(ax[1, 1])
    self.plot_train_test_roc(ax[1, 2])
    self.plot_train_scores_original_mass(ax[2,0])
    self.plot_test_scores_original_mass(ax[2,1])
    
    #self.plot_final_roc(ax[1, 2])
    '''
    train_bkg_dict = {'bkg': self.xb_train}
    test_bkg_dict = {'bkg': self.xb_test}
    bkg_list = ['bkg']
    self.plot_scores_separate(ax[2, 0], train_bkg_dict, bkg_list, 
      self.selected_features, sig_arr=self.xs_train_selected, 
      sig_weights=self.xs_train[:,-1], plot_title='train scores', bins=80, range=(-0.1, 1.1))
    self.plot_scores_separate(ax[2, 1], test_bkg_dict,  bkg_list,
      self.selected_features, sig_arr=self.xs_test_selected,
      sig_weights=self.xs_test[:,-1],  plot_title='test scores',  bins=80, range=(-0.1, 1.1))
    '''
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
  def __init__(
    self, name,
    input_dim, num_node=400,
    learn_rate=0.02, decay=1e-6,
    metrics=['plain_acc'],
    weighted_metrics=['accuracy'],
    save_tb_logs=False,
    tb_logs_path=None
    ):
    model_sequential.__init__(
      self, name,
      input_dim, num_node=num_node, 
      learn_rate=learn_rate, decay=decay,
      metrics=metrics,
      weighted_metrics=weighted_metrics
      )
    self.model_label = "mod1002"
    self.model_note = "Sequential model optimized with old ntuple"\
                      + " at Oct. 2rd 2019"\
                      + " to deal with training with full bkg mass."
    self.save_tb_logs=save_tb_logs
    self.tb_logs_path=tb_logs_path

  def compile(self):
    """ Compile model, function to be changed in the future."""
    # Add layers
    # input
    self.model.add(Dense(self.model_num_node, kernel_initializer='uniform', 
                         input_dim = self.model_input_dim))

    # hidden 1
    #self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 2
    #self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 3
    #self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
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
    self.model.add(Dense(1, kernel_initializer="glorot_uniform", 
                         activation="sigmoid"))
    # Compile
    self.model.compile(loss="binary_crossentropy", 
      optimizer=SGD(
        lr=self.model_learn_rate, 
        decay=self.model_decay),
        metrics=self.metrics,
        weighted_metrics=self.weighted_metrics
        )
    self.model_is_compiled = True


  def prepare_array(self, xs, xb, selected_features, channel_id,
                    sig_weight=1000, bkg_weight=1000, test_rate=0.2,
                    verbose=1):
    """Prepares array for training.
    
    Note:
      Use normalized array for training.

    """
    # Stadard input normalization for training.
    means, vars = get_mean_var(xb[:, 0:-2], axis=0, weights=xb[:, -1])
    self.norm_average = means
    self.norm_variance = vars
    xs_norm = xs.copy()
    xb_norm = xb.copy()
    xs_norm[:, 0:-2] = norarray(xs[:, 0:-2], average=means, variance=vars)
    xb_norm[:, 0:-2] = norarray(xb[:, 0:-2], average=means, variance=vars)
    model_sequential.prepare_array(self, xs_norm, xb_norm, selected_features,
                                   sig_weight=sig_weight, bkg_weight=bkg_weight, 
                                   test_rate=test_rate, verbose=verbose)


  def train(self, weight_id=-1, batch_size=128, epochs=20, val_split=0.25,
            sig_class_weight=1., bkg_class_weight=1., verbose=1, ):
    """Performs training."""
    # Check
    if self.model_is_compiled == False:
      raise ValueError("DNN model is not yet compiled")
    if self.array_prepared == False:
      raise ValueError("Training data is not ready.")
    # Train
    print("Training start. Using model:", self.model_name)
    print("Model info:", self.model_note)
    self.class_weight = {1:sig_class_weight, 0:bkg_class_weight}
    train_callbacks = []
    if self.save_tb_logs:
      if self.tb_logs_path is None:
        self.tb_logs_path = "temp_logs/{}".format(self.model_label)
        warnings.warn("TensorBoard logs path not specified, \
          set path to: {}".format(self.tb_logs_path))
      tb_callback = TensorBoard(log_dir=self.tb_logs_path, histogram_freq=1)
      train_callbacks.append(tb_callback)
    self.train_history = self.get_model().fit(
      self.x_train_selected,
      self.y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=val_split,
      class_weight=self.class_weight,
      sample_weight=self.x_train[:, weight_id],
      callbacks=train_callbacks,
      verbose=verbose
      )
    print("Training finished.")
    # Quick evaluation
    print("Quick evaluation:")
    score = self.get_model().evaluate(self.x_test_selected, 
                                      self.y_test, verbose = verbose, 
                                      sample_weight = self.x_test[:, -1])
    print('> test loss:', score[0])
    print('> test accuracy:', score[1])

    # Save train history
    # save accuracy history
    self.train_history_accuracy = [float(ele) for ele in self.train_history.history['accuracy']]
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
  def __init__(
    self, name,
    input_dim, num_node=400,
    learn_rate=0.02, decay=1e-6,
    metrics=['plain_acc'],
    weighted_metrics=['accuracy'],
    save_tb_logs=False,
    tb_logs_path=None
    ):
    Model_1002.__init__(
      self, name,
      input_dim, num_node=num_node,
      learn_rate=learn_rate, decay=decay,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      save_tb_logs=save_tb_logs,
      tb_logs_path=tb_logs_path)
    self.model_label = "mod1016"
    self.model_note = "New model structure based on 1002's model"\
                      + " at Oct. 16th 2019"\
                      + " to deal with training with full bkg mass."

  def compile(self):
    """ Compile model, function to be changed in the future."""
    # Add layers
    # input
    self.model.add(Dense(500, kernel_initializer='uniform', 
                         input_dim = self.model_input_dim))
    # hidden 1
    #self.model.add(BatchNormalization())
    self.model.add(Dense(400, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 2
    #self.model.add(BatchNormalization())
    self.model.add(Dense(300, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 3
    #self.model.add(BatchNormalization())
    self.model.add(Dense(200, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 4
    #self.model.add(BatchNormalization())
    self.model.add(Dense(100, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 5
    #self.model.add(BatchNormalization())
    #self.model.add(Dense(self.model_num_node, 
    #                     kernel_initializer="glorot_normal", 
    #                     activation="relu"))
    # output
    #self.model.add(BatchNormalization())
    self.model.add(Dense(1, kernel_initializer="glorot_uniform", 
                         activation="sigmoid"))
    # Compile
    self.model.compile(
      loss="binary_crossentropy", 
      optimizer=SGD(
        lr=self.model_learn_rate, 
        decay=self.model_decay,
        momentum=0.5, nesterov=True
        ),
      metrics=self.metrics,
      weighted_metrics=self.weighted_metrics
      )
    self.model_is_compiled = True
