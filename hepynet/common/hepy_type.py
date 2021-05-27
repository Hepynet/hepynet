# defines type hints in hepynet
import os
from typing import List, Tuple, Union

import matplotlib.axes._axes as axes

from hepynet.common import config_utils

# from hepynet.train import hep_model  # TODO: this may cause circular import issue

# matplotlib objects
ax = axes.Axes

# python types combination
bound = Union[Tuple[float, float], List[float]]
pathlike = Union[str, os.PathLike]

# hepynet objects
config = config_utils.Hepy_Config
sub_config = config_utils.Hepy_Config_Section
# model = hep_model.Model_Base
