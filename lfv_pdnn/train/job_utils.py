import os
import csv

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
    current_dir = os.getcwd()
    main_dirs = []
    for main_dir_name in MAIN_DIR_NAMES:
        try:
            found_dirs = re.findall(".*" + main_dir_name, current_dir)
            main_dirs += found_dirs
        except:
            pass
    share_dir = None
    for temp in main_dirs:
        share_dir_temp = temp + "/share"
        if os.path.isdir(share_dir_temp):
            share_dir = share_dir_temp
            break
    if share_dir is None:
        raise ValueError(
            "No valid path found for {}, please check .ini file.".format(share_dir)
        )
    if os.path.isfile(share_dir + "/train/" + path):
        return share_dir + "/train/" + path
    elif os.path.isfile(share_dir + "/" + path):
        return share_dir + "/" + path
    else:
        raise ValueError("No valid path found, please check .ini file.")


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
