import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from attrition_model import __version__ as model_version
from attrition_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {

            "age":41,
            "businesstravel":"Travel_Rarely",
            "dailyrate":1102,
            "department":"Sales",
            "distancefromhome":1,
            "education":2,
            "educationfield":"Life Sciences",
            "environmentsatisfaction":2,
            "gender":"Female",
            "hourlyrate":94,
            "jobinvolvement":3,              
            "joblevel":"2",
            "jobrole":"Sales Executive",
            "jobsatisfaction":4,
            "maritalstatus":"Single",             
            "monthlyincome":5993,
            "monthlyrate":19479,
            "numcompaniesworked":8,
            "overtime":"Yes",
            "percentsalaryhike":11,
            "performancerating":3,            
            "relationshipsatisfaction":1,
            "stockoptionlevel":0,
            "totalworkingyears":8,
            "trainingtimeslastyear":0,
            "worklifebalance":1,
            "yearsatcompany":6,
            "yearsincurrentrole":4,
            "yearssincelastpromotion":0,
            "yearswithcurrmanager":5
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Employee Attrition prediction with the attrition_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

