"""XGBoost model with tuning."""
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

class XGBoostPipeline:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.03),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            use_label_encoder=False, eval_metric="mlogloss", random_state=42, n_jobs=-1)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
        return self.model

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def feature_importance(self, feature_names=None):
        imp = self.model.feature_importances_
        if feature_names:
            return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
        return imp

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred, average="weighted")}
