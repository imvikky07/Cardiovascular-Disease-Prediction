from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = "CVD Risk Detection API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ]

    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/ml/model.joblib")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "app/ml/scaler.joblib")
    ENCODER_PATH: str = os.getenv("ENCODER_PATH", "app/ml/label_encoder.joblib")
    FEATURE_NAMES_PATH: str = os.getenv("FEATURE_NAMES_PATH", "app/ml/feature_names.joblib")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
