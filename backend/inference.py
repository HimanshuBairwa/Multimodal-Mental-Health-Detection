"""
PHQ-9 Depression Risk Inference Module

This module loads a trained XGBoost model (saved as JSON)
and exposes a clean Python interface for predicting
depression risk from PHQ-9 responses.

Author: Himanshu Bairwa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from xgboost import XGBClassifier


@dataclass
class PHQ9PredictionResult:
    """Structured output for a PHQ-9 prediction."""
    probability: float        # Predicted probability of depression (0–1)
    label: str                # "low_risk" / "high_risk" / etc.
    threshold: float          # Threshold used to map probability -> label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "label": self.label,
            "threshold": self.threshold,
        }


class PHQ9DepressionModel:
    """
    Wrapper around an XGBoost model trained on PHQ-9 features.

    Expected input:
        - phq9_scores: list of 9 integers/floats (0–3 each), or
                       the exact feature vector used during training.

    NOTE:
        The actual model file is stored locally under `models/`
        and is intentionally git-ignored (not pushed to GitHub).
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_phq9_model.json",
        positive_class_index: int = 1,
    ) -> None:
        self.model_path = model_path
        self.positive_class_index = positive_class_index
        self.model = self._load_model()

    def _load_model(self) -> XGBClassifier:
        model = XGBClassifier()
        model.load_model(self.model_path)
        return model

    def predict_proba(self, phq9_scores: List[float]) -> float:
        """
        Returns the probability of being in the positive (depressed) class.

        Args:
            phq9_scores: list of PHQ-9 item scores or feature vector.

        Returns:
            Probability (float between 0 and 1).
        """
        x = np.array(phq9_scores, dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(x)[0][self.positive_class_index]
        return float(proba)

    def predict_label(
        self,
        phq9_scores: List[float],
        threshold: float = 0.5,
    ) -> PHQ9PredictionResult:
        """
        Converts probability into a discrete risk label.

        Args:
            phq9_scores: list of scores/features.
            threshold: cutoff for high-risk vs low-risk.

        Returns:
            PHQ9PredictionResult with probability and label.
        """
        p = self.predict_proba(phq9_scores)

        if p >= threshold:
            label = "high_risk"
        else:
            label = "low_risk"

        return PHQ9PredictionResult(
            probability=p,
            label=label,
            threshold=threshold,
        )


if __name__ == "__main__":
    # Example manual test: replace with real PHQ-9 scores (0–3 each)
    example_scores = [1, 2, 1, 0, 2, 1, 0, 1, 2]

    model = PHQ9DepressionModel(
        model_path="models/xgboost_phq9_model.json"
    )

    result = model.predict_label(example_scores, threshold=0.5)
    print("PHQ-9 Example Prediction:", result.to_dict())
