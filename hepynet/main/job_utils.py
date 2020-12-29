import csv
import logging
import os

import hepynet

logger = logging.getLogger("hepynet")

MAIN_DIR_NAMES = ["pdnn-lfv", "work"]


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
    hepynet_dir = os.path.dirname(os.path.dirname(hepynet.__file__))
    logger.debug(f"Get hepynet dir: {hepynet_dir}")
    cfg_path = f"{hepynet_dir}/{path}"
    if os.path.isfile(cfg_path):
        return cfg_path
    elif os.path.isdir(cfg_path):
        logger.critical(
            f"Expect a config file path but get directory path: {cfg_path}, please check .ini file."
        )
        exit()
    else:
        logger.critical(
            f"No valid config file found at path {cfg_path}, please check .ini file."
        )
        exit()


def make_table(data, save_dir, num_para=1):
    """Makes table for scan meta data and so on.

    Input example:
        data = [
            ["col-1", "col-2", "col-3", "col-4" ],
            [1,2,3,4],
            ["a", "b", "c", "d"]
        ]
    """
    # save csv format
    save_path = save_dir + "/scan_meta_report.csv"
    with open(save_path, "w", newline="") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for single_list in data:
            wr.writerow(single_list)
