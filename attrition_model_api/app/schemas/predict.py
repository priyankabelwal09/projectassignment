from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]

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

