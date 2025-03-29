
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
    def get_scaled_mean_std(df: pd.DataFrame, train_ratio=0.7):
        """
        Extract the mean and standard deviation using a StandardScaler from a portion of the data.

        Args:
            df (pd.Series): The time series data.
            train_ratio (float): Ratio of data to use for training (default 0.7).
            test_only (bool): Whether to use only the test data.

        Returns:
            tuple: (mean, std) of the selected portion of the data.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if df.empty:
            return None, None

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        if train_df.notnull().sum().sum() < 1000:
            return None, None

        scaler = StandardScaler()
        _ = scaler.fit_transform(train_df)

        mean = scaler.mean_[0]
        std = np.sqrt(scaler.var_[0])

        return mean, std

    @staticmethod
    def get_scaled(df: pd.DataFrame, train_ratio: float = 0.8, test_only=False) -> pd.DataFrame:
        """
        Scale a DataFrame using StandardScaler fit on training data only.
        
        Args:
            df (pd.DataFrame or pd.Series): Input data to scale
            train_ratio (float): Fraction of data to use as training set

        Returns:
            pd.DataFrame: Scaled version of full input, using training stats
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        if train_df.notnull().sum().sum() < 1000:
            return
        test_df = df.iloc[split_idx:]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        if not test_df.empty:
            test_scaled = scaler.transform(test_df)
        else:
            test_scaled = np.empty((0, train_scaled.shape[1]))

        if test_only:
            return pd.DataFrame(
                np.vstack([test_scaled]),
                index=test_df.index,
                columns=test_df.columns
            )
        else:
            return pd.DataFrame(
                np.vstack([train_scaled, test_scaled]),
                index=df.index,
                columns=df.columns
            )

    
    @classmethod
    def plot_chart(cls, f1: pd.DataFrame, f2: pd.DataFrame=None, scale=False,
                   hline=None, title=None, train_ratio=0.7):
        """ plot chart

        Args:
            f1 (pd.DataFrame): this is to be plotted on axes 1
            f2 (pd.DataFrame): this is to be plotted on axes 2
            scale: apply standard scaler for f2
            hline: if scale, show hline on chart, else show mean +- 2 * std in chart
            title: Chart title
            train_ratio: use in scale, ignore if scale is False
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
                df_scaled = cls.get_scaled(f2, train_ratio=train_ratio)
                df_scaled.plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')
                if hline is not None:
                    ax2.axhline(-hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
                    ax2.axhline(hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
            else:
                f2.plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')
                
                if hline is not None:
                    mean, std = cls.get_scaled_mean_std(f2, train_ratio=train_ratio)
                    if mean is not None and std is not None:
                        ax2.axhline(mean-2*std, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=hline
                        ax2.axhline(mean, color='gray', linewidth=0.6, linestyle='--')  # Adding a horizontal line at y=hline
                        ax2.axhline(mean+2*std, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=hline

            ax2.tick_params(axis='y', labelcolor=color)

        fig.legend(loc="upper left", bbox_to_anchor=(1.02,1), bbox_transform=ax1.transAxes)
        if title:
            plt.title(title)
        plt.show()