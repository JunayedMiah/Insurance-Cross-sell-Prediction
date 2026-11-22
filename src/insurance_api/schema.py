"""Request and response models for the API.

The input fields follow the Kaggle "Health Insurance Cross Sell Prediction" dataset.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Customer(BaseModel):
    """One customer, described by the same columns the model was trained on."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Age": 44,
                "Driving_License": 1,
                "Region_Code": 28,
                "Previously_Insured": 0,
                "Vehicle_Age": "> 2 Years",
                "Vehicle_Damage": "Yes",
                "Annual_Premium": 40454.0,
                "Policy_Sales_Channel": 26,
                "Vintage": 217,
            }
        }
    )

    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=18, le=100)
    Driving_License: Literal[0, 1]
    Region_Code: int = Field(ge=0, le=60)
    Previously_Insured: Literal[0, 1]
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(gt=0, le=1_000_000)
    Policy_Sales_Channel: int = Field(ge=0, le=200)
    Vintage: int = Field(ge=0, le=400, description="Days the customer has been with the company")


class Contribution(BaseModel):
    feature: str
    value: str
    shap_value: float = Field(description="Positive pushes towards 'interested', negative away from it")


class Prediction(BaseModel):
    probability: float = Field(ge=0, le=1)
    interested: bool
    threshold: float
    top_reasons: list[Contribution]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_version: str
    trained_at: str
    threshold: float
    metrics: dict[str, float]
    n_train_rows: int
