"""Data loading and exploration."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    def load(self, path=None):
        return pd.read_csv(path or self.config["data"]["path"])
    def split(self, df):
        target = self.config["data"]["target"]
        X, y = df.drop(columns=[target]), df[target]
        return train_test_split(X, y, test_size=self.config["data"]["test_size"], random_state=42, stratify=y if y.nunique() < 20 else None)
