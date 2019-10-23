# -*- coding: utf-8 -*-
"""Model class for DNN training"""
import os
import time

from datetime import datetime
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Dense, Input, Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, SGD, RMSprop, Adam
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings

from ..common import print_helper
from . import train_utils 
from .train_utils import *

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
    self.model_create_time = datetime.now()
    self.model_is_compiled = False
    self.model_name = name
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

  def __init__(self, name, input_dim, num_node = 300, learn_rate = 0.025, 
               decay = 1e-6):
    """Initialize model."""
    model_base.__init__(self, name)
    # Model parameters
    self.model_input_dim = input_dim
    self.model_num_node = num_node
    self.model_learn_rate = learn_rate
    self.model_decay = decay
    self.model = Sequential()
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

  def plot_accuracy(self, ax):
    """Plots accuracy vs training epoch."""
    # Plot
    try:
      ax.plot(self.get_train_history().history['acc'])
      ax.plot(self.get_train_history().history['val_acc'])
    except KeyError:  # updated for tensorflow2.0
      ax.plot(self.get_train_history().history['accuracy'])
      ax.plot(self.get_train_history().history['val_accuracy'])
    
    # Config
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_ylim((0, 1))
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper center')
    ax.grid()
    return ax

  def get_auc_train(self):
    return auc(self.fpr_dm_train, self.tpr_dm_train)

  def get_auc_test(self):
    return auc(self.fpr_dm_test, self.tpr_dm_test)

  def plot_loss(self, ax):
    """Plots loss vs training epoch."""
    #Plot
    ax.plot(self.get_train_history().history['loss'])
    ax.plot(self.get_train_history().history['val_loss'])
    # Config
    ax.set_title('model loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['train', 'val'], loc='upper center')
    ax.grid()
    return ax

  def plot_roc(self, ax):
    """Plots roc curve."""
    # Check
    if not self.model_is_compiled:
      warnings.warn("Model is not compiled")
    # Plot
    ax.plot(self.fpr_dm_train, self.tpr_dm_train)
    ax.plot(self.fpr_dm_test, self.tpr_dm_test)
    # Config
    ax.set_title("roc curve")
    ax.set_xlabel('tpr')
    ax.set_ylabel('fpr')
    ax.legend(['train', 'test'], loc='lower right')
    ax.grid()
    return ax

  def plot_scores(self, ax, bins=100, range=None, density=True, log=False):
    """Plots training score distribution for siganl and background."""
    ax.hist(self.get_model().predict(self.xs_test_selected), bins=bins, 
            range=range, weights=self.xs_test[:,-1], histtype='step', label='signal', density=True, 
            log=log)
    ax.hist(self.get_model().predict(self.xb_test_selected), bins=bins, 
            range=range, weights=self.xb_test[:,-1], histtype='step', label='background', density=True, 
            log=log)
    ax.set_title('training scores')
    ax.legend(loc='upper center')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    return ax

  def plot_scores_separate(self, ax, arr_dict, key_list, selected_features,
                           sig_arr=None, sig_weights=None, plot_title='training scores',
                           bins=40, range=(-0.25, 1.25), density=True, log=False):
    """Plots training score distribution for different background."""
    predict_arr_list = []
    predict_arr_weight_list = []

    average=self.norm_average
    variance=self.norm_variance

    for arr_key in key_list:
      bkg_arr_temp = arr_dict[arr_key]
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
              stacked=True, log=log)
    except:
      ax.hist(predict_arr_list[0], bins=bins, 
              range=range, weights=predict_arr_weight_list[0], histtype='bar', label=key_list, density=True, 
              stacked=True, log=log)

    if sig_arr is None:
      sig_arr=self.xs_selected
      sig_weights=self.xs[:,-1]

    ax.hist(self.get_model().predict(sig_arr), bins=bins, 
            range=range, weights=sig_weights, histtype='step', label='signal', density=True, 
            log=log)
    
    ax.set_title(plot_title)
    ax.legend(loc='upper center')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    return ax

  def prepare_array(self, xs, xb, selected_features, test_rate=0.2, verbose=1):
    """Prepares array for training."""
    self.xs = xs
    self.xb = xb
    rdm_seed = int(time.time())
    # get train data set, reset bkg mass first
    xb_reset_mass = xb_norm = modify_array(xb, weight_id=-1,
      reset_mass=True, reset_mass_array=xs, reset_mass_id=0)
    self.x_train, _, self.y_train, _, self.xs_train, _, self.xb_train, _ = \
      split_and_combine(xs, xb_reset_mass, test_rate=test_rate,
      shuffle_seed=rdm_seed)
    # get test data set, use same random seed
    _, self.x_test, _, self.y_test, _, self.xs_test, _, self.xb_test = \
      split_and_combine(xs, xb_reset_mass, test_rate=test_rate,
      shuffle_seed=rdm_seed)
    # select features wanted
    self.selected_features = selected_features
    self.x_train_selected = get_part_feature(self.x_train, selected_features)
    self.x_test_selected = get_part_feature(self.x_test, selected_features)

    self.xs_train_selected = get_part_feature(self.xs_train, selected_features)
    self.xb_train_selected = get_part_feature(self.xb_train, selected_features)

    self.xs_test_selected = get_part_feature(self.xs_test, selected_features)
    self.xb_test_selected = get_part_feature(self.xb_test, selected_features)
    self.xs_selected = get_part_feature(self.xs, selected_features)
    self.xb_selected = get_part_feature(self.xb, selected_features)
    self.array_prepared = True
    if verbose == 1:
      print("Training array prepared.")
      print("> signal shape:", self.xs_selected.shape)
      print("> background shape:", self.xb_selected.shape)


  def save_model(self, save_path):
    """Saves trained model.
    
    Args:
      save_path: str
        Path to save model.

    """
    # Check path
    parent_path = os.path.split(save_path)[0]
    if not os.path.exists(parent_path):
      os.makedirs(parent_path)
    # Save
    self.model.save(save_path)
    print("model:", self.model_name, "has been saved to:", save_path)

  def show_performance(self, figsize=(15, 12)):
    """Shortly reports training result.

    Args:
      figsize: tuple
        Defines plot size.

    """
    # Check input
    assert isinstance(self, model_base)
    print("Model performance:")
    # Plots
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    self.plot_scores(ax[0, 0])
    self.plot_roc(ax[0, 1])
    self.plot_accuracy(ax[1, 0])
    self.plot_loss(ax[1, 1])

    train_bkg_dict = {'bkg': self.xb_train}
    test_bkg_dict = {'bkg': self.xb_test }
    bkg_list = ['bkg']
    self.plot_scores_separate(ax[2, 0], train_bkg_dict, bkg_list, 
      self.selected_features, sig_arr=self.xs_train_selected, 
      sig_weights=self.xs_train[:,-1], plot_title='train scores', bins=80, range=(-0.1, 1.1))
    self.plot_scores_separate(ax[2, 1], test_bkg_dict,  bkg_list,
      self.selected_features, sig_arr=self.xs_test_selected,
      sig_weights=self.xs_test[:,-1],  plot_title='test scores',  bins=80, range=(-0.1, 1.1))
    
    fig.tight_layout()
    plt.show()
    print("> auc for train:", self.get_auc_train())
    print("> auc for test: ", self.get_auc_test())

  def train(self):
    pass


class Model_0913(model_sequential):
  """Sequential model optimized with old ntuple at Sep. 9th 2019."""
  def __init__(self, name, input_dim, num_node = 300, learn_rate = 0.025, 
               decay = 1e-6):
    model_sequential.__init__(self, name, input_dim, num_node = 300, 
                              learn_rate = 0.025, decay = 1e-6)
    self.model_note = "Sequential model optimized with old ntuple"\
                      + " at Sep. 9th 2019"

  def compile(self):
    """ Compile model, function to be changed in the future."""
    # Add layers
    # input
    self.model.add(Dense(self.model_num_node, kernel_initializer='uniform', 
                         input_dim = self.model_input_dim))
    # hidden 1
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 2
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 3
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 4
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # hidden 5
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.model_num_node, 
                         kernel_initializer="glorot_normal", 
                         activation="relu"))
    # output
    self.model.add(BatchNormalization())
    self.model.add(Dense(1, kernel_initializer="glorot_uniform", 
                         activation="sigmoid"))
    # Compile
    self.model.compile(loss="binary_crossentropy", 
                       optimizer=SGD(lr=self.model_learn_rate, 
                       decay=self.model_decay), metrics=["accuracy"])
    self.model_is_compiled = True

  def train(self, weight_id = -1, batch_size = 100, epochs = 20, 
            val_split = 0.25, verbose = 1):
    """Performs training."""
    # Check
    if self.model_is_compiled == False:
      raise ValueError("DNN model is not yet compiled")
    if self.array_prepared == False:
      raise ValueError("Training data is not ready.")
    # Train
    print("Training start. Using model:", self.model_name)
    print("Model info:", self.model_note)
    self.train_history = self.get_model().fit(self.x_train_selected, 
                         self.y_train, batch_size = batch_size, 
                         epochs = epochs, validation_split = val_split, 
                         sample_weight = self.x_train[:, weight_id],
                         verbose = verbose)
    print("Training finished.")
    # Quick evaluation
    print("Quick evaluation:")
    score = self.get_model().evaluate(self.x_test_selected, 
                                      self.y_test, verbose = verbose, 
                                      sample_weight = self.x_test[:, -1])
    print('> test loss:', score[0])
    print('> test accuracy:', score[1])
    # Save train history
    # for train sample
    predictions_dm_train = self.get_model().predict(self.x_train_selected)
    self.fpr_dm_train, self.tpr_dm_train, self.threshold_train = \
      roc_curve(self.y_train, predictions_dm_train)
    # for test sample
    predictions_dm_test = self.get_model().predict(self.x_test_selected)
    self.fpr_dm_test, self.tpr_dm_test, self.threshold_test = \
      roc_curve(self.y_test, predictions_dm_test)

class Model_1002(model_sequential):
  """Sequential model optimized with old ntuple at Sep. 9th 2019.
  
  Major modification based on 0913 model:
    1. Optimized to train on full mass range. (Used to be on bkg samples with
       cut to have similar mass range as signal.)
    2. Use normalized data for training.
  
  """
  def __init__(self, name, input_dim, num_node=400, learn_rate=0.02, 
               decay=1e-6):
    model_sequential.__init__(self, name, input_dim, num_node=num_node, 
                              learn_rate=learn_rate, decay=decay)
    self.model_note = "Sequential model optimized with old ntuple"\
                      + " at Oct. 2rd 2019"\
                      + " to deal with training with full bkg mass."

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
                       optimizer=SGD(lr=self.model_learn_rate, 
                       decay=self.model_decay), metrics=["accuracy", "mean_squared_error"])
    self.model_is_compiled = True

  def prepare_array(self, xs, xb, selected_features, channel_id,
                    rm_neg_weight=True, bs_weight_ratio=1., test_rate=0.2,
                    verbose=1):
    """Prepares array for training.
    
    Note:
      Use normalized array for training.

    """
    # first make basic selection
    xs_norm = modify_array(xs, weight_id=-1, select_channel=True,
      channel_id=channel_id, norm=True, sumofweight=1000, shuffle=True,
      shuffle_seed = int(time.time()))
    
    xb_norm = modify_array(xb, weight_id=-1,
      remove_negative_weight=rm_neg_weight, select_channel=True,
      channel_id=channel_id, norm=True,
      sumofweight=1000*bs_weight_ratio, shuffle=True,
      shuffle_seed=int(time.time()))
    # stadard input normalization for training.
    average, variance = get_mean_var(xb_norm[:, 0:-2], axis=0, weights=xb_norm[:, -1])
    self.norm_average = average
    self.norm_variance = variance
    xs_norm[:, 0:-2] = norarray(xs_norm[:, 0:-2], average=average, variance=variance)
    xb_norm[:, 0:-2] = norarray(xb_norm[:, 0:-2], average=average, variance=variance)
    model_sequential.prepare_array(self, xs_norm, xb_norm, selected_features, 
                                   test_rate=test_rate, verbose=verbose)


  def train(self, weight_id = -1, batch_size = 128, epochs = 20, 
            val_split = 0.25, verbose = 1):
    """Performs training."""
    # Check
    if self.model_is_compiled == False:
      raise ValueError("DNN model is not yet compiled")
    if self.array_prepared == False:
      raise ValueError("Training data is not ready.")
    # Train
    print("Training start. Using model:", self.model_name)
    print("Model info:", self.model_note)
    self.train_history = self.get_model().fit(self.x_train_selected, 
                         self.y_train, batch_size = batch_size, 
                         epochs = epochs, validation_split = val_split, 
                         sample_weight = self.x_train[:, weight_id],
                         verbose = verbose)
    print("Training finished.")
    # Quick evaluation
    print("Quick evaluation:")
    score = self.get_model().evaluate(self.x_test_selected, 
                                      self.y_test, verbose = verbose, 
                                      sample_weight = self.x_test[:, -1])
    print('> test loss:', score[0])
    print('> test accuracy:', score[1])
    # Save train history
    # for train sample
    predictions_dm_train = self.get_model().predict(self.x_train_selected)
    self.fpr_dm_train, self.tpr_dm_train, self.threshold_train = \
      roc_curve(self.y_train, predictions_dm_train, sample_weight=self.x_train[:,-1])
    # for test sample
    predictions_dm_test = self.get_model().predict(self.x_test_selected)
    self.fpr_dm_test, self.tpr_dm_test, self.threshold_test = \
      roc_curve(self.y_test, predictions_dm_test, sample_weight=self.x_test[:,-1])

class Model_1016(model_sequential):
  """Sequential model optimized with old ntuple at Sep. 9th 2019.
  
  Major modification based on 1002 model:
    1. Change structure to make quantity of nodes decrease with layer num.
  
  """
  def __init__(self, name, input_dim, num_node=400, learn_rate=0.02, 
               decay=1e-6):
    model_sequential.__init__(self, name, input_dim, num_node=num_node, 
                              learn_rate=learn_rate, decay=decay)
    self.model_note = "Sequential model optimized with old ntuple"\
                      + " at Oct. 2rd 2019"\
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
    self.model.compile(loss="binary_crossentropy", 
                       optimizer=SGD(lr=self.model_learn_rate, 
                       decay=self.model_decay, momentum=0.5, nesterov=True), metrics=["accuracy"])
    self.model_is_compiled = True

  def prepare_array(self, xs, xb, selected_features, channel_id,
                    rm_neg_weight=True, bs_weight_ratio=1., test_rate=0.2,
                    verbose=1):
    """Prepares array for training.
    
    Note:
      Use normalized array for training.

    """
    # first make basic selection
    xs_norm = modify_array(xs, weight_id=-1, select_channel=True,
      channel_id=channel_id, norm=True, sumofweight=1000, shuffle=True,
      shuffle_seed = int(time.time()))
    
    xb_norm = modify_array(xb, weight_id=-1,
      remove_negative_weight=rm_neg_weight, select_channel=True,
      channel_id=channel_id, norm=True,
      sumofweight=1000*bs_weight_ratio, shuffle=True,
      shuffle_seed=int(time.time()))
    # stadard input normalization for training.
    average, variance = get_mean_var(xb_norm[:, 0:-2], axis=0, weights=xb_norm[:, -1])
    self.norm_average = average
    self.norm_variance = variance
    xs_norm[:, 0:-2] = norarray(xs_norm[:, 0:-2], average=average, variance=variance)
    xb_norm[:, 0:-2] = norarray(xb_norm[:, 0:-2], average=average, variance=variance)
    model_sequential.prepare_array(self, xs_norm, xb_norm, selected_features, 
                                   test_rate=test_rate, verbose=verbose)


  def train(self, weight_id = -1, batch_size = 128, epochs = 20, 
            val_split = 0.25, verbose = 1):
    """Performs training."""
    # Check
    if self.model_is_compiled == False:
      raise ValueError("DNN model is not yet compiled")
    if self.array_prepared == False:
      raise ValueError("Training data is not ready.")
    # Train
    print("Training start. Using model:", self.model_name)
    print("Model info:", self.model_note)
    self.train_history = self.get_model().fit(self.x_train_selected, 
                         self.y_train, batch_size = batch_size, 
                         epochs = epochs, validation_split = val_split, 
                         sample_weight = self.x_train[:, weight_id],
                         verbose = verbose)
    print("Training finished.")
    # Quick evaluation
    print("Quick evaluation:")
    score = self.get_model().evaluate(self.x_test_selected, 
                                      self.y_test, verbose = verbose, 
                                      sample_weight = self.x_test[:, -1])
    print('> test loss:', score[0])
    print('> test accuracy:', score[1])
    # Save train history
    # for train sample
    predictions_dm_train = self.get_model().predict(self.x_train_selected)
    self.fpr_dm_train, self.tpr_dm_train, self.threshold_train = \
      roc_curve(self.y_train, predictions_dm_train, sample_weight=self.x_train[:,-1])
    # for test sample
    predictions_dm_test = self.get_model().predict(self.x_test_selected)
    self.fpr_dm_test, self.tpr_dm_test, self.threshold_test = \
      roc_curve(self.y_test, predictions_dm_test, sample_weight=self.x_test[:,-1])
