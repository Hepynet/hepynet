import datetime

from configparser import ConfigParser
import json

from lfv_pdnn_code_v1.data_io.get_arrays import *
from lfv_pdnn_code_v1.train import model
from lfv_pdnn_code_v1.train.train_utils import get_input_array

class job_executor(object):
  """Core class to execute a pdnn job based on given cfg file."""

  def __init__(self, input_path):
    """Initialize executor."""
    # general
    self.job_create_time = str(datetime.datetime.now())
    self.cfg_path = input_path
    self.cfg_is_collected = False
    self.array_is_loaded = False
    # [array] section
    self.arr_version = None
    self.bkg_dict_path = None
    self.bkg_key = None
    self.bkg_sumofweight = None
    self.sig_dict_path = None
    self.sig_key = None
    self.sig_sumofweight = None
    self.selected_features = None
    self.input_dim = None
    self.channel_id = None
    # [model] section
    self.model_name = None
    self.model_class = None
    self.test_rate = None
    self.batch_size = None
    self.epochs = None
    self.val_split = None
    self.sig_class_weight = None
    self.bkg_class_weight = None
    # [report] section
    self.plot_bkg_list = None
    self.verbose = None

  def execute_dnn(self):
    """Execute DNN training with given configuration."""
    if not self.cfg_is_collected:
      self.get_config()
    if not self.array_is_loaded:
      self.load_arrays()
    xs, xb = get_input_array(
      self.sig_dict, '500GeV',
      self.bkg_dict, 'all', -4
      ) # -4 for emu
    self.model = getattr(model, self.model_class)(
      self.model_name, self.input_dim
      )
    self.model.prepare_array(
      xs, xb, self.selected_features,
      self.channel_id,
      sig_weight=self.sig_sumofweight,
      bkg_weight=self.bkg_sumofweight,
      test_rate=self.test_rate,
      verbose=self.verbose
      )
    self.model.compile()
    self.model.train(
      batch_size=self.batch_size,
      epochs=self.epochs,
      val_split=self.val_split,
      sig_class_weight=self.sig_class_weight,
      bkg_class_weight=self.bkg_class_weight,
      verbose=self.verbose
      )

  def get_config(self, path=None):
    """Retrieves configurations from ini file."""
    # Set parser
    if path is None:
      ini_path = self.cfg_path
    else:
      ini_path = path
    config = ConfigParser()
    config.read(ini_path)
    # Check whether need to import other (default) ini file first
    default_ini_path = None
    try:
      default_ini_path = config.get('config', 'include')
    except:
      pass
    if default_ini_path is not None:
      self.get_config(default_ini_path)
    # Load [array] section
    self.try_parse_str('arr_version', config, 'array', 'arr_version')
    self.try_parse_str('bkg_dict_path', config, 'array', 'bkg_dict_path')
    self.try_parse_str('bkg_key', config, 'array', 'bkg_key')
    self.try_parse_float('bkg_sumofweight', config, 'array', 'bkg_sumofweight')
    self.try_parse_str('sig_dict_path', config, 'array', 'sig_dict_path')
    self.try_parse_str('sig_key', config, 'array', 'sig_key')
    self.try_parse_float('sig_sumofweight', config, 'array', 'sig_sumofweight')
    self.try_parse_list('selected_features', config, 'array', 'selected_features')
    if self.selected_features is not None:
      self.input_dim = len(self.selected_features)
    else:
      self.input_dim = None
    self.try_parse_int('channel_id', config, 'array', 'channel_id')
    # Load [model] section
    self.try_parse_str('model_name', config, 'model', 'model_name')
    self.try_parse_str('model_class', config, 'model', 'model_class')
    self.try_parse_float('test_rate', config, 'model', 'test_rate')
    self.try_parse_int('batch_size', config, 'model', 'batch_size')
    self.try_parse_int('epochs', config, 'model', 'epochs')
    self.try_parse_float('val_split', config, 'model', 'val_split')
    self.try_parse_float('sig_class_weight', config, 'model', 'sig_class_weight')
    self.try_parse_float('bkg_class_weight', config, 'model', 'bkg_class_weight')
    # Load [report] section
    self.try_parse_list('plot_bkg_list', config, 'report', 'plot_bkg_list')
    self.try_parse_int('verbose', config, 'report', 'verbose')

    self.cfg_is_collected = True

  def load_arrays(self):
    """Get training arrays."""
    if self.arr_version == 'old':
      self.bkg_dict = get_old_bkg(self.bkg_dict_path)
      self.sig_dict = get_old_sig(self.sig_dict_path)
    self.array_is_loaded = True

  def try_parse_float(self, parsed_val, config_parser, section, val_name):
    try:
      setattr(self, parsed_val, config_parser.getfloat(section, val_name))
    except:
      pass

  def try_parse_int(self, parsed_val, config_parser, section, val_name):
    try:
      setattr(self, parsed_val, config_parser.getint(section, val_name))
    except:
      pass

  def try_parse_str(self, parsed_val, config_parser, section, val_name):
    try:
      setattr(self, parsed_val, config_parser.get(section, val_name))
    except:
      pass

  def try_parse_list(self, parsed_val, config_parser, section, val_name):
    try:
      setattr(self, parsed_val, json.loads(config_parser.get(section, val_name)))
    except:
      pass
