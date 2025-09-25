import os
import matplotlib.pyplot as plt
import seaborn as sns

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
