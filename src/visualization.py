"""Visualization for XGBoost analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class XGBVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_feature_importance(self, importances, top_n=20, save=True):
        top = importances[:top_n]
        names, values = zip(*top)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.invert_yaxis()
        if save:
            fig.savefig(f"{self.output_dir}feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_shap_summary(self, shap_values, X, save=True):
        try:
            import shap
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            if save:
                plt.savefig(f"{self.output_dir}shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass
