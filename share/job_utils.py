import datetime
import itertools
import os
import re

from configparser import ConfigParser
import json
import matplotlib.pyplot as plt
import platform
import re
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Image, PageBreak, Paragraph, Spacer, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

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
    self.scan_para_dict = {}
    # [job] section
    self.job_name = None
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
    self.learn_rate = None
    self.learn_rate_decay = None
    self.test_rate = None
    self.batch_size = None
    self.epochs = None
    self.val_split = None
    self.sig_class_weight = None
    self.bkg_class_weight = None
    self.save_model = None
    self.save_model_path = None
    # [para_scan] section
    self.perform_para_scan = None
    self.para_scan_cfg = None
    self.scan_learn_rate = None
    # [report] section
    self.plot_bkg_list = None
    self.show_report = None
    self.save_pdf_report = None
    self.save_pdf_path = None
    self.verbose = None

    # [scanned_para] section
    self.scan_learn_rate = []

  def execute_jobs(self):
    """Execute all planned jobs."""
    # Execute single job if parameter scan is not needed
    if self.perform_para_scan is not True:
      self.execute_single_job()
      return  # single job executed
    # Perform scan as specified
    print('*' * 80)
    print("Executing parameters scanning.")
    scan_list = self.get_scan_para_list()
    for scan_set in scan_list:
      print('*' * 80)
      print("Scanning parameter set:", scan_set)
      keys = list(scan_set.keys())
      for key in keys:
        setattr(self, key, scan_set[key])
      self.execute_single_job()

  def execute_single_job(self):
    """Execute single DNN training with given configuration."""
    if not self.cfg_is_collected:
      self.get_config()
    if not self.array_is_loaded:
      self.load_arrays()
    xs, xb = get_input_array(
      self.sig_dict, '500GeV',
      self.bkg_dict, 'all', -4
      ) # -4 for emu
    self.model = getattr(model, self.model_class)(
      self.model_name, self.input_dim,
      learn_rate=self.learn_rate,
      decay=self.learn_rate_decay
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
    if self.show_report or self.save_pdf_report:
      # Performance plots
      self.fig_performance_path = None
      self.fig_non_mass_reset_path = None
      self.report_path = None
      # setup save parameters if reports need to be saved
      fig_save_path = None
      datestr = datetime.date.today().strftime("%Y-%m-%d")

      """
      save_dir = self.save_pdf_path + '/' + datestr + '_' + self.job_name
      if self.save_pdf_report:
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)
        fig_save_path = save_dir + '/' + self.job_name \
          + '_performance_' + datestr + '.png'
      """
      path_pattern = None
      save_dir = None
      if self.perform_para_scan:
        path_pattern = self.save_pdf_path + '/' + datestr + '_' + self.job_name\
          + '_scan{}'
        save_dir = get_newest_file_version(path_pattern, n_digit=3)['path']
      else:
        path_pattern = self.save_pdf_path + '/' + datestr + '_' + self.job_name\
          + '_v{}'
        save_dir = get_newest_file_version(path_pattern)['path']
      
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)
      fig_save_path = save_dir + '/' + self.job_name \
        + '_performance_' + datestr + '.png'

      self.fig_performance_path = fig_save_path
      # show and save according to setting
      self.model.show_performance(
        show_fig=self.show_report,
        save_fig=self.save_pdf_report,
        save_path=fig_save_path
        )
      # Extra plots (use model on non-mass-reset arrays)
      fig, ax = plt.subplots(ncols=2, figsize=(16, 4))
      self.model.plot_scores_separate(
        ax[0], self.plot_bkg_dict, self.plot_bkg_list, self.selected_features,
        plot_title='training scores (lin)', bins=40, range=(-0.25, 1.25),
        density=True, log=False
        )
      self.model.plot_scores_separate(
        ax[1], self.plot_bkg_dict, self.plot_bkg_list, self.selected_features,
        plot_title='training scores (log)', bins=40, range=(-0.25, 1.25),
        density=True, log=True
        )
      if self.save_model:
        mod_save_path = self.save_model_path
        self.model.save_model(save_dir=mod_save_path)
      if self.save_pdf_report:
        fig_save_path = save_dir + '/' + self.job_name \
            + '_non-mass-reset_' + datestr + '.png'
        fig.savefig(fig_save_path)
        self.fig_non_mass_reset_path = fig_save_path
        pdf_save_path = save_dir + '/' + self.job_name \
          + '_report_' + datestr + '.pdf'
        self.generate_report(pdf_save_path=pdf_save_path)
        self.report_path = pdf_save_path

  def get_config(self, path=None):
    """Retrieves configurations from ini file."""
    # Set parser
    if path is None:
      ini_path = self.cfg_path
    else:
      ini_path = path
    if not os.path.isfile(ini_path):
      raise ValueError("No vallid configuration file path provided.")
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
    # Load [job] section
    self.try_parse_str('job_name', config, 'job', 'job_name')
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
    self.try_parse_float('learn_rate', config, 'model', 'learn_rate')
    self.try_parse_float('learn_rate_decay', config, 'model', 'learn_rate_decay')
    self.try_parse_float('test_rate', config, 'model', 'test_rate')
    self.try_parse_int('batch_size', config, 'model', 'batch_size')
    self.try_parse_int('epochs', config, 'model', 'epochs')
    self.try_parse_float('val_split', config, 'model', 'val_split')
    self.try_parse_float('sig_class_weight', config, 'model', 'sig_class_weight')
    self.try_parse_float('bkg_class_weight', config, 'model', 'bkg_class_weight')
    self.try_parse_bool('save_model', config, 'model', 'save_model')
    self.try_parse_str('save_model_path', config, 'model', 'save_model_path')
    # Load [para_scan]
    self.try_parse_bool('perform_para_scan', config, 'para_scan', 'perform_para_scan')
    self.try_parse_str('para_scan_cfg', config, 'para_scan', 'para_scan_cfg')
    # Load [report] section
    self.try_parse_list('plot_bkg_list', config, 'report', 'plot_bkg_list')
    self.try_parse_bool('show_report', config, 'report', 'show_report')
    self.try_parse_bool('save_pdf_report', config, 'report', 'save_pdf_report')
    self.try_parse_str('save_pdf_path', config, 'report', 'save_pdf_path')
    self.try_parse_int('verbose', config, 'report', 'verbose')

    if self.perform_para_scan:
      self.get_config_scan()

    self.cfg_is_collected = True

  def get_config_scan(self):
    """Load parameters scan configuration file."""
    config = ConfigParser()
    valid_cfg_path = get_valid_cfg_path(self.para_scan_cfg)
    config.read(valid_cfg_path)
    # get available scan variables:
    self.try_parse_list('scan_learn_rate', config, 'scanned_para', 'scan_learn_rate')

  def get_scan_para_list(self):
    """Gets a list of dictionary to reset parameters for new scan job."""
    scan_para_names = ['scan_learn_rate']
    used_para_lists = []
    used_para_names = []
    for para_name in scan_para_names:
      scanned_values = getattr(self, para_name)
      if len(scanned_values) > 0:
        used_para_names.append(para_name)
        used_para_lists.append(scanned_values)
    combs = list(itertools.product(*used_para_lists))
    scan_list = []
    for comb in combs:
      scan_dict_single = {}
      for (key, value) in zip(used_para_names, comb):
        scan_dict_single[key] = value
        scan_list.append(scan_dict_single)
    print("[debug]: scan_list:", scan_list)
    if len(scan_list) < 1:
      raise ValueError("Empty scan parameter list, please check .ini file.")
    return scan_list

  def generate_report(self, pdf_save_path=None):
    """Generate a brief report to show how is the model."""
    # Initalize
    if pdf_save_path is None:
      datestr = datetime.date.today().strftime("%Y-%m-%d")
      save_dir = self.save_pdf_path + '/' + datestr + '_' + self.job_name
      pdf_save_path = save_dir + '/' + self.job_name \
        + '_report_' + datestr + '.pdf'
    doc = SimpleDocTemplate(
      pdf_save_path, pagesize=letter,
      rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18
      )
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    reports=[]
    # Reports
    # head
    ptext = "JOB NAME: " + self.job_name
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "DATE TIME: " + self.job_create_time
    reports.append(Paragraph(ptext, styles["Justify"]))
    reports.append(Spacer(1, 12))
    # machine info
    ptext = "MACHINE INFO:"
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "-" * 80
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = 'machine:' + platform.machine()
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = 'version:' + platform.version()
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = 'platform:' + platform.platform()
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = 'system:' + platform.system()
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = 'processor:' + platform.processor()
    reports.append(Paragraph(ptext, styles["Justify"]))
    reports.append(Spacer(1, 12))
    # plots
    ptext = "PERFORMANCE PLOTS:"
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "-" * 80
    reports.append(Paragraph(ptext, styles["Justify"]))
    fig = self.fig_performance_path
    im = Image(fig, 6.4*inch, 3.6*inch)
    reports.append(im)
    fig = self.fig_non_mass_reset_path
    im = Image(fig, 6.4*inch, 1.6*inch)
    reports.append(im)
    # parameters
    reports.append(PageBreak())
    ptext = "KEY PARAMETERS:"
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "-" * 80
    reports.append(Paragraph(ptext, styles["Justify"]))
    reports.append(Spacer(1, 12))
    ptext = "config file location: " + re.sub('[\s+]', '', self.cfg_path)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "[array]"
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "array version           :    " + self.arr_version
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "channel id              :    " + self.interpret_channel_id(self.channel_id)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "selected features id    :    " + str(self.selected_features)
    reports.append(Paragraph(ptext, styles["Justify"]))
    reports.append(Spacer(1, 12))
    ptext = "bkg arrays path         :    " + self.bkg_dict_path
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "bkg arrays used         :    " + self.bkg_key
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "bkg total weight set    :    " + str(self.bkg_sumofweight)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "sig arrays path         :    " + self.sig_dict_path
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "sig arrays used         :    " + self.sig_key
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "sig total weight        :    " + str(self.sig_sumofweight)
    reports.append(Paragraph(ptext, styles["Justify"]))
    reports.append(Spacer(1, 12))
    ptext = "[model]"
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "name                    :    " + self.model_name
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "class                   :    " + self.model_class
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "learn rate              :    " + str(self.learn_rate)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "learn decay             :    " + str(self.learn_rate_decay)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "test ratio              :    " + str(self.test_rate)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "validation split        :    " + str(self.val_split)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "batch size              :    " + str(self.batch_size)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "epochs                  :    " + str(self.epochs)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "signal class weight     :    " + str(self.sig_class_weight)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "background class weight :    " + str(self.bkg_class_weight)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "model saved             :    " + str(self.save_model)
    reports.append(Paragraph(ptext, styles["Justify"]))
    ptext = "model saved path        :    " + str(self.model.model_save_path)
    reports.append(Paragraph(ptext, styles["Justify"]))

    # build/save
    doc.build(reports)

  def interpret_channel_id(self, channel_id, interpret_dict=None):
    """Gives channel id real meaning."""
    if interpret_dict is None:
      interpret_dict = {
        -4: 'emu',
        -3: 'etau',
        -2: 'mutau'
      }
    try:
      return interpret_dict[channel_id]
    except:
      raise ValueError("Invalid channel key detected.")

  def load_arrays(self):
    """Get training arrays."""
    if self.arr_version == 'old':
      self.bkg_dict = get_old_bkg(self.bkg_dict_path)
      self.sig_dict = get_old_sig(self.sig_dict_path)
    self.array_is_loaded = True
    if self.show_report or self.save_pdf_report:
      self.plot_bkg_dict = {key:self.bkg_dict[key] for key in self.plot_bkg_list}

  def set_para(self, parsed_val, data_type, config_parser, section, val_name):
    """Sets parameters for training manually."""
    if data_type == 'bool':
      pass
    elif data_type == 'float':
      float_temp = config_parser.getfloat(section, val_name)
      setattr(self, parsed_val, float_temp)

  def try_parse_bool(self, parsed_val, config_parse, section, val_name):
    try:
      setattr(self, parsed_val, config_parse.getboolean(section, val_name))
    except:
      pass

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


def get_valid_cfg_path(path):
  """Finds valid path for cfg file in /share folder.
  
  If path is already valid:
    Nothing will be done and original path will be returned.
  If path is not valid:
    Try to add share folder path before to see whether we can get a valid path.
    Otherwise, raise error to ask configuration correction.

  """
  # Check path:
  if os.path.isfile(path):
    return path
  # Check try add share folder prefix
  curren_dir = os.getcwd()
  main_dirs = re.findall(r".*pDNN-Code-for-LFV-v1\.0", curren_dir)
  share_dir = None
  for temp in main_dirs:
    share_dir_temp = temp + '/share'
    if os.path.isdir(share_dir_temp):
      share_dir = share_dir_temp
      break
  if share_dir is None:
    raise ValueError('No valid path found, please check .ini file.')
  if os.path.isfile(share_dir + '/' + path):
    return share_dir + '/' + path
  elif os.path.isfile(share_dir + path):
    return share_dir + '/' + path
  else:
    raise ValueError('No valid path found, please check .ini file.')
