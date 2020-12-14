import argparse
import collections
import copy
import logging
import os
from typing import Any

import lfv_pdnn
import yaml
from lfv_pdnn.common import common_utils

logger = logging.getLogger("lfv_pdnn")

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
                logger.critical(
                    f"Expect section {key} must has dict type value or None, please check the input."
                )
                raise ValueError

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
                logger.critical(
                    f"Expect section {key} has dict type value or None, please check the input."
                )
                raise ValueError

    def print(self) -> None:
        """Shows all configs
        """
        logger.info("")
        logger.info("Config details " + ">" * 80)
        for key, value in self.__dict__.items():
            logger.info(f"[{key}]")
            value.print()
        logger.info("Config ends " + "<" * 83)
        logger.info("")


class Hepy_Config_Section(object):
    """Helper class to handle job configs in a section"""

    def __init__(self, section_config_dict: dict) -> None:
        self._config_dict = section_config_dict
        for key, value in section_config_dict.items():
            if type(value) is dict:
                setattr(self, key, Hepy_Config_Section(value))
            else:
                setattr(self, key, value)

    def __deepcopy__(self, memo):
        clone_obj = Hepy_Config_Section(self.get_config_dict())
        return clone_obj

    def __getattr__(self, item):
        """Called when an attribute lookup has not found the attribute in the usual places"""
        return None

    def clone(self):
        return copy.deepcopy(self)

    def get_config_dict(self) -> dict:
        """ Returns config in dict format """
        return self._config_dict

    def update(self, cfg_dict: dict) -> None:
        """Updates the section config dict with new dict, overwrite if exists

        Args:
            cfg_dict (dict): new section config dict for update
        """
        dict_merge(self._config_dict, cfg_dict)
        for key, value in cfg_dict.items():
            if type(value) is dict:
                if key in self.__dict__.keys() and key != "_config_dict":
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, Hepy_Config_Section(value))
            else:
                setattr(self, key, value)

    def print(self, tabs=0) -> None:
        """Shows all section configs
        """
        for key, value in self.__dict__.items():
            if key != "_config_dict":
                if isinstance(value, Hepy_Config_Section):
                    logger.info(f"{' '*4*tabs}    {key} :")
                    for sub_key, sub_value in value.__dict__.items():
                        if sub_key != "_config_dict":
                            if isinstance(sub_value, Hepy_Config_Section):
                                sub_value.print(tabs=tabs + 1)
                            else:
                                logger.info(
                                    f"{' '*4*tabs}        {sub_key} : {sub_value}"
                                )
                elif isinstance(value, list):
                    logger.info(f"{' '*4*tabs}    {key} :")
                    for ele in value:
                        logger.info(f"{' '*4*tabs}        - {ele}")
                else:
                    logger.info(f"{' '*4*tabs}    {key} : {value}")


def dict_merge(my_dict, merge_dict):
    """ Recursive dict merge """
    for key in merge_dict.keys():
        if (
            key in my_dict
            and isinstance(my_dict[key], dict)
            and isinstance(merge_dict[key], collections.Mapping)
        ):
            dict_merge(my_dict[key], merge_dict[key])
        else:
            my_dict[key] = merge_dict[key]


def load_current_platform_meta() -> dict:
    """Loads meta data for current platform

    Returns:
        dict: meta data dict of current platform
    """
    platform_meta = load_pc_meta()["platform_meta"]
    current_hostname = common_utils.get_current_hostname()
    current_platform = common_utils.get_current_platform_name()
    if current_hostname in platform_meta:
        if current_platform in platform_meta[current_hostname]:
            return platform_meta[current_hostname][current_platform]
    logger.critical(
        f"No meta data found for current host {current_hostname} with platform {current_platform}, please update the config at share/cross_platform/pc_meta.yaml"
    )
    raise KeyError


def load_pc_meta() -> dict:
    """Loads platform meta dict for different path setup in different machines.

    Returns:
        dict: meta information dictionary
    """
    lfv_pdnn_dir = os.path.dirname(lfv_pdnn.__file__)
    logger.debug(f"Found lfv_pdnn root directory at: {lfv_pdnn_dir}")
    pc_meta_cfg_path = f"{lfv_pdnn_dir}/../share/cross_platform/pc_meta.yaml"
    try:
        with open(pc_meta_cfg_path) as pc_meta_file:
            pc_meta_dict = yaml.load(pc_meta_file, Loader=yaml.FullLoader)
            logger.debug(f"pc_meta config loaded: \n {pc_meta_dict}")
            return pc_meta_dict
    except:
        logger.critical(
            "Can't load pc_meta config file, please check: share/cross_platform/pc_meta.yaml"
        )
        raise FileNotFoundError


def load_yaml_dict(yaml_path) -> dict:
    try:
        yaml_file = open(yaml_path, "r")
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
    except:
        logger.critical(f"Can't open yaml config: {yaml_path}")
        raise FileNotFoundError
