"""Advanced feature engineering pipeline."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from itertools import combinations

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.freq_maps = {}
        self.target_means = {}

    def create_interactions(self, df, columns=None, top_n=5):
        df = df.copy()
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns[:top_n].tolist()
        new_features = {}
        for c1, c2 in combinations(numeric_cols, 2):
            new_features[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            new_features[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)
        return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    def create_polynomial(self, df, columns, degree=2):
        df = df.copy()
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[columns])
        names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=names, index=df.index)
        new_cols = [c for c in poly_df.columns if c not in df.columns]
        return pd.concat([df, poly_df[new_cols]], axis=1)

    def frequency_encoding(self, df, columns):
        df = df.copy()
        for col in columns:
            freq = df[col].value_counts(normalize=True)
            self.freq_maps[col] = freq.to_dict()
            df[f"{col}_freq"] = df[col].map(freq)
        return df

    def target_encoding(self, df, columns, target, smoothing=10):
        df = df.copy()
        global_mean = target.mean()
        for col in columns:
            means = df.groupby(col).apply(lambda x: target.loc[x.index].mean())
            counts = df[col].value_counts()
            smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
            self.target_means[col] = smooth_means.to_dict()
            df[f"{col}_target_enc"] = df[col].map(smooth_means)
        return df

    def create_statistical_features(self, df, columns):
        df = df.copy()
        df["row_mean"] = df[columns].mean(axis=1)
        df["row_std"] = df[columns].std(axis=1)
        df["row_max"] = df[columns].max(axis=1)
        df["row_min"] = df[columns].min(axis=1)
        df["row_skew"] = df[columns].skew(axis=1)
        return df
