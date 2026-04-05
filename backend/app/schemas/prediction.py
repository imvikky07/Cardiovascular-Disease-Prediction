from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years (1–120)")
    gender: GenderEnum = Field(..., description="Gender: Male or Female")
    cholesterol: int = Field(..., ge=100, le=400, description="Total cholesterol in mg/dL")
    blood_pressure: int = Field(..., ge=50, le=250, description="Systolic blood pressure in mmHg")
    glucose: int = Field(..., ge=50, le=400, description="Fasting glucose in mg/dL")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status: 0=No, 1=Yes")
    alcohol: int = Field(..., ge=0, le=1, description="Alcohol intake: 0=No, 1=Yes")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    physical_activity: int = Field(..., ge=0, le=1, description="Physical activity: 0=No, 1=Yes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "gender": "Male",
                "cholesterol": 240,
                "blood_pressure": 140,
                "glucose": 110,
                "smoking": 1,
                "alcohol": 0,
                "bmi": 28.5,
                "physical_activity": 0,
            }
        }
    }


class RiskLevel(str, Enum):
    low = "Low Risk"
    high = "High Risk"


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 0=Low Risk, 1=High Risk")
    risk_level: RiskLevel = Field(..., description="Human-readable risk label")
    risk_probability: float = Field(..., description="Probability of CVD risk (0.0–1.0)")
    risk_percentage: float = Field(..., description="Risk probability as percentage")
    confidence: str = Field(..., description="Confidence level: Low / Medium / High")
    message: str = Field(..., description="Descriptive result message")
    input_summary: PredictionRequest = Field(..., description="Echo of input data")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    service: str
