"""
Note: These tests will fail if you have not first trained the model.
"""

import sys, os
from pathlib import Path

# windows changes
# sys.path.append(str(Path(__file__).parent.parent))
# Linux changes
file = Path(__file__).resolve()
print("file:", file)
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

print("paraen:", file.parents[1])
print("paraen:", file.parent)

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import (
    Mapper,
    WeathersitImputer,
    WeekdayImputer,
)
