# Backend – Multimodal Mental Health Detection

This folder contains the **server-side / model-side code** for the project.

## Modules

### 1. `phq9/` – Clinical Survey (PHQ-9) Inference

- Loads a trained **XGBoost** model stored locally under `models/`.
- Exposes a clean Python interface:

  ```python
  from backend.phq9 import PHQ9DepressionModel

  model = PHQ9DepressionModel(model_path="models/xgboost_phq9_model.json")
  result = model.predict_label(phq9_scores=[1, 2, 0, 1, 3, 1, 0, 2, 1])
  print(result.to_dict())
