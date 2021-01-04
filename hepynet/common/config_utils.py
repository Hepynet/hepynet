import argparse
import collections
import copy
import logging
import pathlib
import re
import sys
from typing import Any

import yaml

import hepynet
from hepynet.common import common_utils
from hepynet.common.config_default import DEFAULT_CFG

logger = logging.getLogger("hepynet")


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

    def clone(self):
        return copy.deepcopy(self)

    def get_config_dict(self) -> dict:
        """ Returns config in dict format """
        out_dict = {}
        out_dict["config"] = self.config.get_config_dict()
        out_dict["job"] = self.job.get_config_dict()
        out_dict["input"] = self.input.get_config_dict()
        out_dict["train"] = self.train.get_config_dict()
        out_dict["apply"] = self.apply.get_config_dict()
        out_dict["para_scan"] = self.para_scan.get_config_dict()
        out_dict["run"] = self.run.get_config_dict()
        return out_dict

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
        config_dict = dict()
        for key, value in self.__dict__.items():
            if key != "_config_dict":
                if isinstance(value, Hepy_Config_Section):
                    config_dict[key] = value.get_config_dict()
                else:
                    config_dict[key] = value
        return config_dict

    def update(self, cfg_dict: dict) -> None:
        """Updates the section config dict with new dict, overwrite if exists

        Args:
            cfg_dict (dict): new section config dict for update
        """
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
                    value.print(tabs=tabs + 1)
                elif isinstance(value, list):
                    logger.info(f"{' '*4*tabs}    {key} :")
                    for ele in value:
                        logger.info(f"{' '*4*tabs}        - {ele}")
                else:
                    logger.info(f"{' '*4*tabs}    {key} : {value}")


def load_current_platform_meta() -> dict:
    """Loads meta data for current platform

    Returns:
        dict: meta data dict of current platform
    """
    platform_meta = load_pc_meta()["platform_meta"]
    current_hostname = common_utils.get_current_hostname()
    current_platform = common_utils.get_current_platform_name()
    for my_host in platform_meta.keys():
        if re.match(f"({my_host})", current_hostname):
            for my_platform in platform_meta[my_host]:
                if re.match(f"({my_platform})", current_platform):
                    return platform_meta[my_host][my_platform]
    logger.critical(
        f"No meta data found for current host {current_hostname} with platform {current_platform}, please update the config at share/cross_platform/pc_meta.yaml"
    )
    exit(1)


def load_pc_meta() -> dict:
    """Loads platform meta dict for different path setup in different machines.

    Returns:
        dict: meta information dictionary
    """
    pc_meta_cfg_path = f"{pathlib.Path().parent}/share/cross_platform/pc_meta.yaml"
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
        logger.critical(
            f"Can't read yaml config: {yaml_path}, please check whether input yaml config exists and the syntax is correct"
        )
        raise IOError
