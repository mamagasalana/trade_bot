import numpy as np
import pandas as pd
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from src.pattern.ccy_strength import CCY_STR, CCY_STR_DEBUG, SMOOTHING_METHOD
from src.pattern.helper import HELPER
import matplotlib.pyplot as plt  # Make sure this import is at the top
import itertools

class HMM:

    def __init__(self, windows=[300, 400, 500]):
        self.c = CCY_STR(['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS'])
        self.windows= windows
        data = []
        ranks = []
        for window in windows:
            data.append(self.c.smoothing_method(window).add_suffix(f'_{window}'))
            ranks.append(data[-1].rank(axis=1, method='min', ascending=False))

        self.data = pd.concat(data, axis=1).dropna()
        self.ranks = pd.concat(ranks, axis=1).dropna()
        self.prices = self.c.get_all_pairs()

    def plot(self, pairs=['EUR', 'USD'], use_rank=False, n_components=3, train_ratio=0.7, window_size=300):

        pairs_windows =list(itertools.product(pairs, self.windows))

        pw2 = ['_'.join(map(str, pw)) for pw in pairs_windows]
        if use_rank:
            combined = pd.concat([self.data[pw2], self.ranks[pw2].add_suffix('_rank')], axis=1)
        else:
            combined = pd.concat([self.data[pw2]])
        
        X_full = combined.values.astype(np.float64)  # shape: (n_days, 2 * n_assets)
        step_size = 1  # can increase to skip less overlapping windows
        windows = []
        rolling_index = []
        for i in range(0, len(X_full) - window_size + 1, step_size):
            w = X_full[i:i + window_size]
            windows.append(w)
            rolling_index.append(combined.index[i + window_size - 1])

        X_rolling = np.stack(windows)  
        split_idx = int(X_rolling.shape[0] * train_ratio)

        model = DenseHMM([Normal() for _ in range(n_components)], max_iter=100, verbose=False)
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
        df_probs.plot(ax=ax1)
        ax1.axvline(x=df_probs.index[split_idx], color='red', linestyle='--', label='Train/Test Split')
        ax1.set_ylabel("Regime Probabilities")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # Plot EURUSD price on second axis
        ax2 = ax1.twinx()
        ax2.plot(df_probs.index, price, color='black', label=f'{pair2} Price', alpha=0.4)
        ax2.set_ylabel(pair2)
        ax2.legend(loc="upper right")

        plt.title(f"HMM Regime Probabilities with {pair2} Overlay")
        plt.tight_layout()
        plt.show()