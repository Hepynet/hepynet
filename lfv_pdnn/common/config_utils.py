import os

import lfv_pdnn
import yaml
from lfv_pdnn.common.logging_cfg import *


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
