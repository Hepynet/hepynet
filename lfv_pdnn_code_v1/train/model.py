from datetime import datetime
from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, SGD, RMSprop, Adam
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import warnings

from lfv_pdnn_code_v1.common import print_helper
from lfv_pdnn_code_v1.train import train_utils 
from train_utils import split_and_combine, get_part_feature

class model_base(object):
  """base model of deep neural network for pdnn training"""
  def __init__(self, name):
    """ initialize model

    Args: 
      name: str, name of the model
    """
    self.model_create_time = datetime.now()
    self.model_is_compiled = False
    self.model_name = name
    self.train_history = ''

class model_sequential(model_base):
  """Sequential model base"""
  def __init__(self, name, input_dim, num_node = 300, learn_rate = 0.025, 
               decay = 1e-6):
    model_base.__init__(self, name)
    self.model_input_dim = input_dim
    self.model_num_node = num_node
    self.model_learn_rate = learn_rate
    self.model_decay = decay
    self.model = Sequential()
  
  def compile(self):
    pass

  def get_model(self):
    """Returns model"""
    if not self.model_is_compiled:
      warnings.warn("model is not compiled")
    return self.model


class model_0913(model_sequential):
  """Sequential model optimized with old ntuple at Sep. 9th 2019"""
  def __init__(self, name, input_dim, num_node = 300, learn_rate = 0.025, 
               decay = 1e-6):
    model_sequential.__init__(self, name, input_dim, num_node = 300, 
                              learn_rate = 0.025, decay = 1e-6)
    self.array_prepared = False
    self.x_train = np.array([])
    self.x_test = np.array([])
    self.y_train = np.array([])
    self.y_test = np.array([])
    self.xs_test = np.array([])
    self.xb_test = np.array([])
    self.model_note = "Sequential model optimized with old ntuple"\
                      + " at Sep. 9th 2019"

  def compile(self):
    """ Compile model, function to be changed in the future"""
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

  def prepare_array(self, xs, xb, selected_features, test_rate = 0.2):
    # get training data
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

  def train(self, weight_id = -1, batch_size = 100, epochs = 20, 
            val_split = 0.25, verbose = 1):
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
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

  def get_model(self):
    """Returns model"""
    return super(model_0913, self).get_model()
