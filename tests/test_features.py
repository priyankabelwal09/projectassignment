
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from attrition_model.config.core import config
from attrition_model.processing.features import Mapper


def test_martialsts_variable_mapper(sample_input_data):
    print(sample_input_data[0].head())    
    
    # Given
    #mapper = Mapper(variables = config.model_config_.jobrole_var, 
                   # mappings = config.model_config_.jobrole_mappings)
    #assert sample_input_data[0].loc[50, 'jobrole'] == 'Sales Executive'

    # When
    #subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    #assert subject.loc[50, 'jobrole'] == 0