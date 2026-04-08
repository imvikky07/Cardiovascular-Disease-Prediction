from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import PredictionRequest, PredictionResponse, RiskLevel
from app.ml.model_loader import model_loader
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_confidence(probability: float) -> str:
    if probability < 0.35 or probability > 0.65:
        return "High"
    elif probability < 0.45 or probability > 0.55:
        return "Medium"
    return "Low"


def get_message(prediction: int, probability: float) -> str:
    pct = round(probability * 100, 1)
    if prediction == 0:
        return (
            f"Your cardiovascular risk assessment shows a {pct}% probability of CVD. "
            "Your risk level appears low. Maintain a healthy lifestyle to keep it that way."
        )
    return (
        f"Your cardiovascular risk assessment shows a {pct}% probability of CVD. "
        "Your risk level is elevated. We recommend consulting a healthcare professional promptly."
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict CVD Risk",
    description="Submit patient data and receive a cardiovascular disease risk prediction.",
)
async def predict(request: PredictionRequest):
    try:
        # if not model_loader.is_loaded:
        #     model_loader.load()

        features = {
            "age": request.age,
            "gender": request.gender.value,
            "cholesterol": request.cholesterol,
            "blood_pressure": request.blood_pressure,
            "glucose": request.glucose,
            "smoking": request.smoking,
            "alcohol": request.alcohol,
            "bmi": request.bmi,
            "physical_activity": request.physical_activity,
        }

        prediction, probability = model_loader.predict(features)

        risk_level = RiskLevel.high if prediction == 1 else RiskLevel.low
        confidence = get_confidence(probability)
        message = get_message(prediction, probability)

        return PredictionResponse(
            prediction=prediction,
            risk_level=risk_level,
            risk_probability=round(probability, 4),
            risk_percentage=round(probability * 100, 2),
            confidence=confidence,
            message=message,
            input_summary=request,
        )

    except ValueError as e:
        logger.error(f"Validation error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(e)}",
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )
