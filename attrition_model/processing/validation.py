import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from attrition_model.config.core import config
from attrition_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    age: Optional[int]
    #attrition: Optional[str]
    businesstravel: Optional[str]
    dailyrate : Optional[int]
    department: Optional[str]
    distancefromhome: Optional[int]
    education: Optional[int]
    educationfield : Optional[str]
    environmentsatisfaction : Optional[int]
    gender: Optional[str]
    hourlyrate: Optional[int]
    jobinvolvement: Optional[int]
    joblevel: Optional[int]
    jobrole: Optional[str]
    jobsatisfaction  : Optional[int]
    maritalstatus: Optional[str]
    monthlyincome : Optional[int]
    monthlyrate : Optional[int]
    numcompaniesworked: Optional[int]
    overtime  : Optional[str]
    percentsalaryhike : Optional[int]
    performancerating: Optional[int]
    relationshipsatisfaction : Optional[int]
    stockoptionlevel : Optional[int]
    totalworkingyears : Optional[int]
    trainingtimeslastyear : Optional[int]
    worklifebalance : Optional[int]
    yearsatcompany : Optional[int]
    yearsincurrentrole : Optional[int]
    yearssincelastpromotion : Optional[int]
    yearswithcurrmanager : Optional[int]
     
     
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
