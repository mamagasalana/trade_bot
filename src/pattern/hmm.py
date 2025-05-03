import numpy as np
import pandas as pd
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from src.pattern.ccy_strength import CCY_STR, CCY_STR_DEBUG, SMOOTHING_METHOD
from src.pattern.helper import HELPER
import matplotlib.pyplot as plt  # Make sure this import is at the top
import itertools
import seaborn as sns

class HMM:

    def __init__(self, windows=[300, 400, 500], method=SMOOTHING_METHOD.ROLLING_ZSCORE):
        self.c = CCY_STR(['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS'])
        self.windows= windows
        data = []
        ranks = []
        for window in windows:
            data.append(self.c.smoothing_method(window, method=method).add_suffix(f'_{window}'))
            ranks.append(data[-1].rank(axis=1, method='min', ascending=False))

        self.data = pd.concat(data, axis=1)
        self.ranks = pd.concat(ranks, axis=1)
        self.prices = self.c.get_all_pairs()

    def plot(self, pairs=['EUR', 'USD'], diffs_shift=[], diffs=[],use_rank=False, n_components=3, train_ratio=0.7, window_size=300, verbose=False):
        """plot hmm

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            verbose(bool, optional): used by HMM to display debug message
        """

        pairs_windows =list(itertools.product(pairs, self.windows))

        pw2 = ['_'.join(map(str, pw)) for pw in pairs_windows]

        pairs_strength_diffs_shift = []
        for i in diffs_shift:
            pairs_strength_diffs_shift.append(self.data[pw2].diff(1).shift(i).add_suffix(f'_shift{i}'))

        pairs_strength_diffs = []
        for i in diffs:
            pairs_strength_diffs.append(self.data[pw2].diff(i).add_suffix(f'_diff{i}'))

        if use_rank:
            combined = pd.concat([self.data[pw2], self.ranks[pw2].add_suffix('_rank')] +pairs_strength_diffs+pairs_strength_diffs_shift, axis=1)
        else:
            combined = pd.concat([self.data[pw2]] + pairs_strength_diffs+pairs_strength_diffs_shift, axis=1)
        
        split_idx = int(len(combined) *train_ratio)
        split_dt = combined.index[split_idx]
        print(split_dt)

        combined.dropna(inplace=True)
        X_full = combined.values.astype(np.float64)  # shape: (n_days, 2 * n_assets)
        step_size = 1  # can increase to skip less overlapping windows
        windows = []
        rolling_index = []
        for i in range(0, len(X_full) - window_size + 1, step_size):
            w = X_full[i:i + window_size]
            windows.append(w)
            rolling_index.append(combined.index[i + window_size - 1])

        X_rolling = np.stack(windows)  
        split_idx = rolling_index.index(split_dt)

        model = DenseHMM([Normal() for _ in range(n_components)], max_iter=100, verbose=verbose)

        model.fit([X_rolling[:split_idx]])
        
        ret = model.predict_proba(X_rolling)
        last_probs = ret[:, -1, :].numpy() 
        
        df_probs = pd.DataFrame(last_probs, columns=[f"regime_{i}" for i in range(last_probs.shape[1])])
        df_probs.index = pd.to_datetime(rolling_index)  # restore original datetime index

        if pairs[0] + pairs[1] in self.prices.columns:
            pair2 = pairs[0] + pairs[1]
        else:
            pair2 = pairs[1] + pairs[0]
        price = self.prices[pair2].loc[rolling_index]
        # Add vertical line at the train/test split index
        fig, ax1 = plt.subplots(figsize=(12, 4))

        colors = ['tab:red', 'tab:orange', 'yellow', 'tab:green', 'tab:blue', 'tab:pink']
        df_probs.plot(ax=ax1 , color=colors)
        ax1.axvline(x=df_probs.index[split_idx], color='red', linestyle='--', label='Train/Test Split')
        ax1.set_ylabel("Regime Probabilities")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # Plot EURUSD price on second axis
        ax2 = ax1.twinx()
        ax2.plot(df_probs.index, price, color='black', label=f'{pair2} Price', alpha=0.4, linewidth=2.5)
        ax2.set_ylabel(pair2)
        ax2.legend(loc="upper right")

        plt.title(f"HMM Regime Probabilities with {pair2} Overlay")
        plt.tight_layout()
        plt.show()

        return combined, model
    
    def heat_map(self, model, columns):
        
        mean_data = []
        var_data = []
        states = []
        for i, dist in enumerate(model.distributions):
            log_weight, mean_tensor, cov_tensor = dist.parameters()
            mean = mean_tensor.numpy()
            cov = cov_tensor.numpy()
            mean_data.append(mean)
            var_data.append(np.diag(cov))
            states.append('State %s' % i)

        pairs= sorted(list(set(x[:3] for x in columns)))
        total_pairs = len(pairs)
        # Create DataFrames
        df_mean = pd.DataFrame(mean_data, index=states, columns=columns)
        df_var = pd.DataFrame(var_data, index=states, columns=columns)

        # Plot heatmaps
        plt.figure(figsize=(16,5* total_pairs))
        for i in range(total_pairs):
            plt.subplot(2, total_pairs, i+1)
            sns.heatmap(df_mean[[x for x in columns if pairs[i] in x]], annot=True, cmap="RdBu_r", center=0, fmt=".2f")
            plt.title(f"Mean of Each Feature {pairs[i]} by Regime")


            plt.subplot(2, total_pairs, i+total_pairs+1)
            sns.heatmap(df_var[[x for x in columns if pairs[i] in x]], annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"Variance of Each Feature  {pairs[i]}  by Regime")

        plt.tight_layout()
        plt.show()
