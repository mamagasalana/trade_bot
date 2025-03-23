
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    
    @staticmethod
    def get_scaled(df):
        scaler = StandardScaler()
        if isinstance(df, pd.Series):
            prescale = df.to_frame()
        else:
            prescale = df
            
        return pd.DataFrame(
            scaler.fit_transform(prescale),
            index=prescale.index,
            columns=prescale.columns)
    
    @classmethod
    def plot_chart(cls, f1: pd.DataFrame, f2: pd.DataFrame=None, scale=False):
        """ plot chart

        Args:
            f1 (pd.DataFrame): this is to be plotted on axes 1
            f2 (pd.DataFrame): this is to be plotted on axes 2
            scale: apply standard scaler for f2
        """
        f1_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        f2_colors = ['gray', 'tab:olive', 'tab:cyan', 'tab:gray', 'black', 'magenta', 'gold']

        fig, ax1 = plt.subplots(figsize=(13, 4))
        # Plot actual and modeled oil prices on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        f1.plot(ax=ax1, legend=False, color=f1_colors)
        ax1.tick_params(axis='y', labelcolor=color)

        if f2 is not None:
        # Create a second y-axis 

            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('spread', color=color) 
            if scale:
                df_scaled = cls.get_scaled(f2)
                df_scaled.plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')
                ax2.axhline(-2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
                ax2.axhline(2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
            else:
                f2.plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
