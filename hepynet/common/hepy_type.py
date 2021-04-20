# defines type hints in hepynet
import os
from typing import Tuple, Union
import matplotlib.axes._axes as axes
import numpy as np
from hepynet.train import hep_model

from hepynet.common import config_utils

# matplotlib objects
ax = axes.Axes

# python types combination
bound = Tuple[float, float]
pathlike = Union[str, os.PathLike]

# hepynet objects
config = config_utils.Hepy_Config
sub_config = config_utils.Hepy_Config_Section
model = hep_model.Model_Base
