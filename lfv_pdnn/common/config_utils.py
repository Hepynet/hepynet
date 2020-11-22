import argparse
import logging
import os
from typing import Any

import lfv_pdnn
import yaml


DEFAULT_CFG = {"input": {"region": ""}, "train": {"output_bkg_node_names": []}}


class Hepy_Config(object):
    """Helper class to handle job configs"""

    def __init__(self, config: dict) -> None:
        # define supported sections to avoid lint error
        self.config = Hepy_Config_Section({})
        self.job = Hepy_Config_Section({})
        self.input = Hepy_Config_Section({})
        self.train = Hepy_Config_Section({})
        self.apply = Hepy_Config_Section({})
        self.para_scan = Hepy_Config_Section({})
        self.report = Hepy_Config_Section({})
        self.run = Hepy_Config_Section({})
        # set default
        self.update(DEFAULT_CFG)
        # initialize config
        for key, value in config.items():
            if type(value) is dict:
                getattr(self, key).update(value)
            elif value is None:
                pass
            else:
                logging.critical(
                    f"Expect section {key} must has dict type value or None, please check the input."
                )
                exit(1)

    def update(self, config: dict) -> None:
        """Updates configs with given config dict, overwrite if exists

        Args:
            config (dict): two level dictionary of configs
        """
        for key, value in config.items():
            if type(value) is dict:
                if key in self.__dict__.keys():
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, Hepy_Config_Section(value))
            elif value is None:
                pass
            else:
                logging.critical(
                    f"Expect section {key} has dict type value or None, please check the input."
                )
                exit(1)

    def print(self) -> None:
        """Shows all configs
        """
        logging.info("")
        logging.info("Config details >>> ")
        for key, value in self.__dict__.items():
            logging.info(f"[{key}]")
            value.print()
        logging.info("Config ends <<< ")


class Hepy_Config_Section(object):
    """Helper class to handle job configs in a section"""

    def __init__(self, section_config_dict: dict) -> None:
        for key, value in section_config_dict.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        """Called when an attribute lookup has not found the attribute in the usual places"""
        return None

    def update(self, cfg_dict: dict) -> None:
        """Updates the section config dict with new dict, overwrite if exists

        Args:
            cfg_dict (dict): new section config dict for update
        """
        for key, value in cfg_dict.items():
            setattr(self, key, value)

    def print(self) -> None:
        """Shows all section configs
        """
        for key, value in self.__dict__.items():
            logging.info(f"    {key} : {value}")


def load_pc_meta() -> dict:
    """Loads platform meta dict for different path setup in different machines.

    Returns:
        dict: meta information dictionary
    """
    lfv_pdnn_dir = os.path.dirname(lfv_pdnn.__file__)
    logging.debug(f"Found lfv_pdnn root directory at: {lfv_pdnn_dir}")
    pc_meta_cfg_path = f"{lfv_pdnn_dir}/../share/cross_platform/pc_meta.yaml"
    try:
        with open(pc_meta_cfg_path) as pc_meta_file:
            pc_meta_dict = yaml.load(pc_meta_file, Loader=yaml.FullLoader)
            logging.debug(f"pc_meta config loaded: \n {pc_meta_dict}")
            return pc_meta_dict
    except:
        logging.critical(
            "Can't load pc_meta config file, please check: share/cross_platform/pc_meta.yaml"
        )
        exit(1)


def load_yaml_dict(yaml_path) -> dict:
    try:
        yaml_file = open(yaml_path, "r")
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
    except:
        logging.critical(f"Can't open yaml config: {yaml_path}")
        exit(1)
