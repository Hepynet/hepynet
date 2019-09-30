# -*- coding: utf-8 -*-
"""Model class for DNN training"""
import os

from datetime import datetime
from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, SGD, RMSprop, Adam
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings

from ..common import print_helper
import train_utils 
from train_utils import split_and_combine, get_part_feature

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
    ax.plot(self.get_train_history().history['acc'])
    ax.plot(self.get_train_history().history['val_acc'])
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
            range=range, histtype='step', label='signal', density=True, 
            log=log)
    ax.hist(self.get_model().predict(self.xb_test_selected), bins=bins, 
            range=range, histtype='step', label='background', density=True, 
            log=log)
    ax.set_title('training scores')
    ax.legend(loc='upper center')
    ax.set_xlabel("Output score")
    ax.set_ylabel("arb. unit")
    ax.grid()
    return ax

  def prepare_array(self, xs, xb, selected_features, test_rate = 0.2):
    """Prepares array for training."""
    self.xs = xs
    self.xb = xb
    self.x_train, self.x_test, self.y_train, self.y_test, self.xs_test, \
    self.xb_test = split_and_combine(xs, xb, test_rate = test_rate)
    # select features wanted
    self.selected_features = selected_features
    self.x_train_selected = get_part_feature(self.x_train, selected_features)
    self.x_test_selected = get_part_feature(self.x_test, selected_features)
    self.xs_test_selected = get_part_feature(self.xs_test, selected_features)
    self.xb_test_selected = get_part_feature(self.xb_test, selected_features)
    self.xs_selected = get_part_feature(self.xs, selected_features)
    self.xb_selected = get_part_feature(self.xb, selected_features)
    self.array_prepared = True
    print "Training array prepared."
    print "> signal shape:", self.xs_selected.shape
    print "> background shape:", self.xb_selected.shape

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
    print "model:", self.model_name, "has been saved to:", save_path

  def show_performance(self, figsize=(16, 9)):
    """Shortly reports training result.

    Args:
      figsize: tuple
        Defines plot size.

    """
    # Check input
    assert isinstance(self, model_base)
    print "Model performance:"
    # Plots
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    self.plot_scores(ax[0, 0])
    self.plot_roc(ax[0, 1])
    self.plot_accuracy(ax[1, 0])
    self.plot_loss(ax[1, 1])
    fig.tight_layout()
    plt.show()
    print "> auc for train:", self.get_auc_train()
    print "> auc for test: ", self.get_auc_test()

  def train(self):
    pass


class model_0913(model_sequential):
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
                       optimizer=SGD(lr=0.025, decay=1e-6), 
                       metrics=["accuracy"])
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
    print "Training start. Using model:", self.model_name
    print "Model info:", self.model_note
    self.train_history = self.get_model().fit(self.x_train_selected, 
                         self.y_train, batch_size = batch_size, 
                         epochs = epochs, validation_split = val_split, 
                         sample_weight = self.x_train[:, weight_id],
                         verbose = verbose)
    print "Training finished."
    # Quick evaluation
    print "Quick evaluation:"
    score = self.get_model().evaluate(self.x_test_selected, 
                                      self.y_test, verbose = verbose, 
                                      sample_weight = self.x_test[:, -1])
    print '> test loss:', score[0]
    print '> test accuracy:', score[1]
    # Save train history
    # for train sample
    predictions_dm_train = self.get_model().predict(self.x_train_selected)
    self.fpr_dm_train, self.tpr_dm_train, self.threshold_train = \
      roc_curve(self.y_train, predictions_dm_train)
    # for test sample
    predictions_dm_test = self.get_model().predict(self.x_test_selected)
    self.fpr_dm_test, self.tpr_dm_test, self.threshold_test = \
      roc_curve(self.y_test, predictions_dm_test)
