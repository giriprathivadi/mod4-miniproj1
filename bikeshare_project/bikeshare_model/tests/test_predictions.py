"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score

from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
   
    # TDB
    pass
