import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

sns.set_theme(style="whitegrid", palette="muted")


class DataFormatter:
    def __init__(self, df):
        self.df = df

    def get_data_types(self):
        """Return value counts of data types"""
        return self.df.dtypes.value_counts()

    def get_missing_counts(self, drop_zero=True):
        """Return missing value counts per column"""
        missing_counts = self.df.isnull().sum()
        if drop_zero:
            missing_counts = missing_counts[missing_counts > 0]
        return missing_counts.sort_values(ascending=False)

    def get_numeric_df(self) -> pd.DataFrame:
        return self.df.select_dtypes(include=["number"])
   
    def get_correlation(self) -> pd.DataFrame:
        num_df = self.get_numeric_df()
        return num_df.corr(numeric_only=True)

    def get_variances(self) -> pd.Series:
        num_df = self.get_numeric_df()
        return num_df.var(numeric_only=True).sort_values(ascending=False)

    def get_low_variance_features(self, threshold: float = 0.0) -> list[str]:
        variances = self.get_variances()
        return variances[variances <= threshold].index.tolist()

    def get_high_corr_pairs(self, thresh: float = 0.9) -> pd.DataFrame:
        corr = self.get_correlation().abs()
        # upper triangle mask to avoid duplicates/self-pairs
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_vals = corr.where(mask).stack()
        hits = corr_vals[corr_vals >= thresh].sort_values(ascending=False)
        return hits.reset_index().rename(columns={"level_0": "feature_1",
                                                  "level_1": "feature_2",
                                                  0: "abs_corr"})


class DataVisualizer:
    def __init__(self, df, vis_dir="data/visualization"):
        self.df = df
        self.formatter = DataFormatter(df)
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def visualizeDataTypes(self):
        data_types = self.formatter.get_data_types()

        plt.figure(figsize=(8, 6))
        ax = data_types.plot(
            kind="bar",
            color=sns.color_palette("Set2")
        )
        plt.title("Column Data Types", fontsize=14, weight="bold")
        plt.xlabel("Data Type", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=10, color="black"
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "column_data_types.png"), dpi=300)
        plt.close()

    def visualizeMissingValues(self):
        missing_counts = self.formatter.get_missing_counts()

        plt.figure(figsize=(max(12, len(missing_counts) * 0.6), 6))
        ax = missing_counts.plot(
            kind="bar",
            color=sns.color_palette("deep"),
            width=0.7
        )
        plt.title("Missing Values per Column", fontsize=14, weight="bold")
        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Missing Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for p in ax.patches:
            ax.annotate(
                str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=9, color="black", rotation=90
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "missing_values_bar.png"), dpi=300)
        plt.close()


    
    def visualizeCorrelationHeatmap(self):
        corr = self.formatter.get_correlation()
        if corr.empty:
            print("No numeric columns for correlation heatmap.")
            return
        # mask the upper triangle for cleaner view
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(min(18, 1.0 + 0.6 * corr.shape[1]), 12))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap (Numeric Features)", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "correlation_heatmap.png"), dpi=300)
        plt.close()

        # corr.to_csv(os.path.join(self.vis_dir, "correlation_matrix.csv"))

        # And top correlated pairs (abs >= 0.9 by default)
        top_pairs = self.formatter.get_high_corr_pairs(thresh=0.9)
        if not top_pairs.empty:
            top_pairs.to_csv(os.path.join(self.vis_dir, "high_corr_pairs.csv"), index=False)

    # Might not be as Important as it captures features that are important for injury prevention
    def visualizeVariances(self, threshold):
        variances = self.formatter.get_variances()
        if variances.empty:
            print("No numeric columns for variance plot.")
            return

        plt.figure(figsize=(max(12, len(variances) * 0.5), 6))
        ax = variances.plot(kind="bar", width=0.7)
        plt.title("Feature Variance (Numeric)", fontsize=14, weight="bold")
        plt.xlabel("Features"); plt.ylabel("Variance")
        plt.xticks(rotation=45, ha="right")

        if threshold is not None:
            plt.axhline(y=threshold, linestyle="--", linewidth=1.2)
            # annotate how many will be dropped
            low_count = (variances <= threshold).sum()
            plt.text(0.99, 0.95,
                     f"Threshold={threshold:.4g} → drop {low_count} feature(s)",
                     transform=ax.transAxes, ha="right", va="top", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "feature_variances.png"), dpi=300)
        plt.close()
