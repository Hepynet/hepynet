import datetime

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
    self.test_rate = None
    self.batch_size = None
    self.epochs = None
    self.val_split = None
    self.sig_class_weight = None
    self.bkg_class_weight = None
    self.save_model = None
    self.save_model_path = None
    # [report] section
    self.plot_bkg_list = None
    self.show_report = None
    self.save_pdf_report = None
    self.save_pdf_path = None
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
    if self.show_report or self.save_pdf_report:
      # Performance plots
      self.fig_performance_path = None
      self.fig_non_mass_reset_path = None
      self.report_path = None
      # setup save parameters if reports need to be saved
      fig_save_path = None
      datestr = datetime.date.today().strftime("%Y-%m-%d")
      save_dir = self.save_pdf_path + '/' + datestr + '_' + self.job_name
      if self.save_pdf_report:
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
    self.try_parse_float('test_rate', config, 'model', 'test_rate')
    self.try_parse_int('batch_size', config, 'model', 'batch_size')
    self.try_parse_int('epochs', config, 'model', 'epochs')
    self.try_parse_float('val_split', config, 'model', 'val_split')
    self.try_parse_float('sig_class_weight', config, 'model', 'sig_class_weight')
    self.try_parse_float('bkg_class_weight', config, 'model', 'bkg_class_weight')
    self.try_parse_bool('save_model', config, 'model', 'save_model')
    self.try_parse_str('save_model_path', config, 'model', 'save_model_path')
    # Load [report] section
    self.try_parse_list('plot_bkg_list', config, 'report', 'plot_bkg_list')
    self.try_parse_bool('show_report', config, 'report', 'show_report')
    self.try_parse_bool('save_pdf_report', config, 'report', 'save_pdf_report')
    self.try_parse_str('save_pdf_path', config, 'report', 'save_pdf_path')
    self.try_parse_int('verbose', config, 'report', 'verbose')

    self.cfg_is_collected = True

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
