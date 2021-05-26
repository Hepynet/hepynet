import glob
import logging
import platform
import re
import socket
from typing import Any

logger = logging.getLogger("hepynet")


def get_current_platform_name() -> str:
    """Returns the name of the current platform.

    Returns:
        str: name of current platform
    """
    return platform.platform()


def get_current_hostname() -> str:
    """Returns the hostname of current machine

    Returns:
        str: current hostname
    """
    return socket.gethostname()


def get_default_if_none(input_var: Any, default_value: Any):
    if input_var is None:
        return default_value
    else:
        return input_var


def get_newest_file_version(
    path_pattern: str,
    n_digit: int = 2,
    ver_num: int = None,
    use_existing: bool = False,
):
    """Check existed file and return last available file path with version.

    Version range 00 -> 99 (or 999)
    If reach limit, last available version will be used. 99 (or 999)

    """
    # Return file path if ver_num is given
    if ver_num is not None:
        return {
            "ver_num": ver_num,
            "path": path_pattern.format(str(ver_num).zfill(n_digit)),
        }
    # Otherwise try to find ver_num
    path_list = glob.glob(path_pattern.format("*"))
    path_list = sorted(path_list)
    if len(path_list) < 1:
        if use_existing:
            logger.debug(
                f"Can't find existing file with path pattern: {path_pattern}, returning empty."
            )
            return {}
        else:
            ver_num = 0
            path = path_pattern.format(str(0).zfill(n_digit))
    else:
        path = path_list[-1]  # Choose the last match
        version_tag_search = re.compile("v(" + "\d" * n_digit + ")")
        ver_num = int(version_tag_search.search(path).group(1))
        if not use_existing:
            ver_num += 1
            path = path_pattern.format(str(ver_num).zfill(n_digit))

    return {
        "ver_num": ver_num,
        "path": path,
    }


def get_significant_digits(number: float, n_digits: int):
    if round(number) == number:
        m = len(str(number)) - 1 - n_digits
        if number / (10 ** m) == 0.0:
            return number
        else:
            return float(int(number) / (10 ** m) * (10 ** m))
    if len(str(number)) > n_digits + 1:
        return round(number, n_digits - len(str(int(number))))
    else:
        return number
