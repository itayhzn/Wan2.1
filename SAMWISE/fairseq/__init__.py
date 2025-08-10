# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from SAMWISE.fairseq.X import Y`
from SAMWISE.fairseq.distributed import utils as distributed_utils
from SAMWISE.fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

# initialize hydra
from SAMWISE.fairseq.dataclass.initialize import hydra_init

hydra_init()

import SAMWISE.fairseq.criterions  # noqa
import SAMWISE.fairseq.distributed  # noqa
import SAMWISE.fairseq.models  # noqa
import SAMWISE.fairseq.modules  # noqa
import SAMWISE.fairseq.optim  # noqa
# import SAMWISE.fairseq.optim.lr_scheduler  # noqa
import SAMWISE.fairseq.pdb  # noqa
import SAMWISE.fairseq.scoring  # noqa
import SAMWISE.fairseq.tasks  # noqa
import SAMWISE.fairseq.token_generation_constraints  # noqa

#import SAMWISE.fairseq.benchmark  # noqa
#import SAMWISE.fairseq.model_parallel  # noqa
