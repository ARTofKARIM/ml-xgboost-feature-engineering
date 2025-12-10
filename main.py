"""Main pipeline for XGBoost with feature engineering."""
import yaml
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import XGBoostPipeline
from src.visualization import XGBVisualizer

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    loader = DataLoader()
    df = loader.load()
    X_train, X_test, y_train, y_test = loader.split(df)
    fe = FeatureEngineer()
    numeric_cols = X_train.select_dtypes(include=["number"]).columns[:5].tolist()
    X_train = fe.create_interactions(X_train, numeric_cols)
    X_train = fe.create_statistical_features(X_train, numeric_cols)
    X_test = fe.create_interactions(X_test, numeric_cols)
    X_test = fe.create_statistical_features(X_test, numeric_cols)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    pipeline = XGBoostPipeline(config["model"])
    pipeline.train(X_train, y_train, X_test, y_test)
    results = pipeline.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

if __name__ == "__main__":
    main()
