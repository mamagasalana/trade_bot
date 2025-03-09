
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class HELPER:

    @staticmethod
    def get_unit_ccy(pairs:list):
        currencies = set()
        for pair in pairs:
            currencies.update([pair[:3], pair[3:]])  # First 3 letters and last 3 letters
        return list(sorted(currencies))


    @staticmethod
    def return_weekday_only(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to keep only weekday (Monday-Friday) entries based on the index.

        Parameters:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.

        Returns:
        pd.DataFrame: A DataFrame containing only weekday rows.
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Filter only weekdays (Monday=0, Sunday=6)
        return df[df.index.dayofweek < 5]

    @staticmethod
    def find_na_threshold(df : pd.DataFrame):
        # Percentage of missing values per column
        missing_percentage = df.isna().mean() * 100

        # Plot histogram
        plt.figure(figsize=(8,5))
        plt.hist(missing_percentage, bins=20, edgecolor='k')
        plt.xlabel("Percentage of Missing Values")
        plt.ylabel("Number of Columns")
        plt.title("Distribution of Missing Data")
        plt.show()

        thresholds = np.arange(0.5, 1.0, 0.01)  # Test from 50% to 100%
        remaining_columns = [df.dropna(thresh=df.shape[0] * t, axis=1).shape[1] for t in thresholds]

        # Plot threshold vs. remaining columns
        plt.plot(thresholds, remaining_columns, marker='o')
        plt.xlabel("Threshold for Non-NaN Values")
        plt.ylabel("Remaining Columns")
        plt.title("Optimal Threshold Selection")
        plt.grid(True)
        plt.show()

        diffs = np.diff(remaining_columns)
        elbow_index = np.argmax(diffs < np.mean(diffs))  # First stable point
        optimal_threshold = thresholds[elbow_index]

        print(f"Recommended threshold: {optimal_threshold:.2f}")
        return optimal_threshold