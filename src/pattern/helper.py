
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from enum import Enum

class SCALE(Enum):
    NO_SCALE = 0
    SCALE = 1
    OTHER = 2

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
    def plot_chart(cls, f1: pd.DataFrame, f2: pd.DataFrame=None, scale: SCALE= SCALE.NO_SCALE,
                   hline=None, title=None, train_ratio=0.7, window=(0,None)):
        """
        Plot a comparative chart of two datasets using dual y-axes, with optional scaling and horizontal reference lines.

        This function visualizes the primary data (`f1`) on the left y-axis and optionally plots a secondary data (`f2`)
        on the right y-axis. The secondary data can be scaled using a Standard Scaler or visualized as is. Additionally,
        horizontal lines can be plotted for reference, either using fixed values or scaled mean and standard deviation.

        Args:
            f1 (pd.DataFrame): 
                Primary data to be plotted on the left y-axis. This is often the actual observed data.
            f2 (pd.DataFrame, optional): 
                Secondary data to be plotted on the right y-axis. This can be modeled data or a spread. Defaults to None.
            scale (SCALE, optional): 
                Scaling option for `f2`. Supports:
                - SCALE.NO_SCALE: Plot without scaling.
                - SCALE.SCALE: Apply standard scaling using the training ratio.
                - SCALE.OTHER: Plot using mean and standard deviation with a ±2 std reference line.
                Defaults to SCALE.NO_SCALE.
            hline (float, optional): 
                A horizontal reference line at `±hline`. If `SCALE.OTHER` is used, it will plot using ±2 std deviations
                from the mean. Defaults to None.
            title (str, optional): 
                Chart title. Defaults to None.
            train_ratio (float, optional): 
                Ratio used for splitting the data into training and testing when scaling is applied. 
                Only applicable when `scale=SCALE.SCALE` or `SCALE.OTHER`. Defaults to 0.7.
            window (tuple, optional): 
                A tuple specifying the range of data to visualize using start and end indices `(start, end)`. 
                Use `(0, None)` to display the entire dataset. Defaults to (0, None).
        
        Behavior:
            - `f1` is plotted on the primary y-axis using a solid line.
            - If `f2` is provided, it is plotted on a secondary y-axis using a dashed line (`:`).
            - Horizontal lines are drawn using the specified `hline` or using ±2 standard deviations if scaled with `SCALE.OTHER`.
            - A legend is displayed on the right side of the figure.

        Example:
            >>> HELPER.plot_chart(f1=data1, f2=data2, scale=SCALE.SCALE, hline=2, title="Comparison Chart")

        Notes:
            - Ensure `f1` and `f2` have the same time index for meaningful comparisons.
            - For SCALE.OTHER, mean and standard deviation are calculated using the training data.

        """
        f1_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        f2_colors = ['gray', 'tab:olive', 'tab:cyan', 'tab:gray', 'black', 'magenta', 'gold']

        fig, ax1 = plt.subplots(figsize=(13, 4))
        # Plot actual and modeled oil prices on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        f1.iloc[window[0]: window[-1]].plot(ax=ax1, legend=False, color=f1_colors)
        ax1.tick_params(axis='y', labelcolor=color)

        if f2 is not None:
            # Create a second y-axis 
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('spread', color=color) 
            
            if scale == SCALE.SCALE:
                df_scaled = cls.get_scaled(f2, train_ratio=train_ratio)
                df_scaled.plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')

            else:
                f2.iloc[window[0]: window[-1]].plot( color=f2_colors, ax=ax2, legend=False, linestyle=':')
                
            if hline is not None:
                if scale == SCALE.SCALE:
                    ax2.axhline(-hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
                    ax2.axhline(hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
                    
                elif scale == SCALE.OTHER:
                    mean, std = cls.get_scaled_mean_std(f2, train_ratio=train_ratio)
                    if mean is not None and std is not None:
                        ax2.axhline(mean-2*std, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=hline
                        ax2.axhline(mean, color='gray', linewidth=0.6, linestyle='--')  # Adding a horizontal line at y=hline
                        ax2.axhline(mean+2*std, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=hline
                elif scale ==  SCALE.NO_SCALE:
                    ax2.axhline(-hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
                    ax2.axhline(hline, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0

           # Add a vertical line for the train-test split
            
            split_index = max(0, int(len(f2) * train_ratio) -1)
            if window[-1] is None or (window[0] < split_index < window[-1]):
                split_date = f2.index[split_index]
                ax2.axvline(split_index, color='red', linestyle='--', linewidth=1.2, label='Train-Test Split')

                ax2.text(split_index, ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.05,
                        f'{split_date}', 
                        color='red', 
                        fontsize=10, 
                        ha='center', 
                        va='bottom')
            ax2.tick_params(axis='y', labelcolor=color)

        fig.legend(loc="upper left", bbox_to_anchor=(1.02,1), bbox_transform=ax1.transAxes)
        if title:
            plt.title(title)
        plt.show()

    @classmethod
    def plot_pivot(cls, df, index, value, column, batches=1):
        """
        Plot a line graph of specified values over a given index, 
        grouped by a specified column, with optional batching for clearer visualization.

        This function is useful for visualizing how a particular metric (e.g., standard deviation)
        evolves over a range of windows or other time-series indices, for different categories
        (e.g., currencies). The data is plotted in separate figures with a clear, non-overlapping legend.

        Args:
            df (pd.DataFrame): 
                Input DataFrame containing the data to plot. 
                It must contain the specified index, value, and column.
            index (str, optional): 
                The column name to use for the x-axis (e.g., 'window'). 
                Defaults to 'window'.
            value (str, optional): 
                The column name representing the values to plot on the y-axis (e.g., 'std'). 
                Defaults to 'std'.
            column (str, optional): 
                The column representing different categories or groups (e.g., 'ccy'). 
                Defaults to 'ccy'.
            batches (int, optional): 
                The number of batches to split the unique categories into. 
                Useful for reducing clutter in the plots. 
                Defaults to 1, meaning no batching.

        Example:
            >>> plot_pivot(df, index='window', value='std', column='ccy', batches=3)
            This will plot the standard deviation ('std') over different windows,
            with currencies ('ccy') divided into 3 batches for better readability.

        Notes:
            - The function automatically splits the categories into the specified number of batches using `numpy.array_split()`.
            - Each batch is plotted in a separate figure.
            - The legend for each figure is displayed outside the plot to avoid overlapping.
        """
        unique_ccys = df[column].unique()
        batches = np.array_split(unique_ccys, batches)

        for i, batch in enumerate(batches):
            fig, ax = plt.subplots(figsize=(13, 4))
            all_lines = []
            all_labels = []

            for ccy in batch:
                subset = df[df[column] == ccy]
                line, = ax.plot(subset[index], subset[value], label=ccy)
                all_lines.append(line)
                all_labels.append(ccy)

            ax.set_xlabel(index)
            ax.set_ylabel(value)
            ax.set_title(f'{value} Over Different Windows - Batch {i+1}')
            ax.grid(True)

            # Add a consolidated legend outside the plot
            fig.legend(all_lines, all_labels, loc="upper left", bbox_to_anchor=(1.02, 1), bbox_transform=ax.transAxes)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

        plt.show()

