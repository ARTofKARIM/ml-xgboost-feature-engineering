# XGBoost Feature Engineering Pipeline

Advanced feature engineering pipeline with XGBoost classification, SHAP explainability, and automated feature creation.

## Architecture
```
ml-xgboost-feature-engineering/
├── src/
│   ├── data_loader.py        # Data loading
│   ├── feature_engineer.py   # Interactions, polynomial, target/freq encoding
│   ├── model.py              # XGBoost with CV and tuning
│   └── visualization.py      # Feature importance, SHAP plots
├── config/config.yaml
├── tests/test_features.py
└── main.py
```

## Installation
```bash
git clone https://github.com/mouachiqab/ml-xgboost-feature-engineering.git
cd ml-xgboost-feature-engineering && pip install -r requirements.txt
python main.py
```

## Technologies
- Python 3.9+, XGBoost, SHAP, scikit-learn, category-encoders















