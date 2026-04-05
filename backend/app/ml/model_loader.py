import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    _instance = None
    _model = None
    _scaler = None
    _label_encoder = None
    _feature_names = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        try:
            self._model = joblib.load(settings.MODEL_PATH)
            self._scaler = joblib.load(settings.SCALER_PATH)
            self._label_encoder = joblib.load(settings.ENCODER_PATH)
            self._feature_names = joblib.load(settings.FEATURE_NAMES_PATH)
            self._loaded = True
            logger.info("ML model and artifacts loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Model artifact not found: {e}")
            raise RuntimeError(
                f"Model artifacts missing. Run scripts/train_model.py first. Error: {e}"
            )

    def predict(self, features: Dict[str, Any]) -> Tuple[int, float]:
        if not self._loaded:
            self.load()

        gender_encoded = int(self._label_encoder.transform([features["gender"]])[0])

        row = {
            "age":               features["age"],
            "gender":            gender_encoded,
            "cholesterol":       features["cholesterol"],
            "blood_pressure":    features["blood_pressure"],
            "glucose":           features["glucose"],
            "smoking":           features["smoking"],
            "alcohol":           features["alcohol"],
            "bmi":               features["bmi"],
            "physical_activity": features["physical_activity"],
        }
        feature_vector = pd.DataFrame([row], columns=self._feature_names)

        scaled = self._scaler.transform(feature_vector)
        prediction = int(self._model.predict(scaled)[0])
        probability = float(self._model.predict_proba(scaled)[0][1])

        return prediction, probability

    @property
    def is_loaded(self) -> bool:
        return self._loaded


model_loader = ModelLoader()
