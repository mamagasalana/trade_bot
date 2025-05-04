import numpy as np
import pandas as pd
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from src.pattern.ccy_strength import CCY_STR, CCY_STR_DEBUG, SMOOTHING_METHOD
from src.pattern.helper import HELPER
import matplotlib.pyplot as plt  # Make sure this import is at the top
import itertools
import seaborn as sns
from src.csv.cache import CACHE

def before_exit(func):
    def wrapper(self, *args, **kwargs):
        try:
            self.updated=True
            return func(self, *args, **kwargs)
        finally:
            if self.updated:
                self.save_cache()
    return wrapper

class HMM:

    def __init__(self, windows=[300, 400, 500], method=SMOOTHING_METHOD.ROLLING_ZSCORE):
        self.c = CCY_STR(['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS'])
        self.windows= windows
        self.updated = False
        self.method = method
        self.cache = CACHE('hmm_cache.cache')
        data = []
        ranks = []
        spread = []
        for window in windows:
            data.append(self.c.smoothing_method(window, method=method).add_suffix(f'_{window}'))
            spread.append(self.c.mt5_export(window=window, method=method).add_suffix(f'_{window}'))
            ranks.append(data[-1].rank(axis=1, method='min', ascending=False))
            
        self.data = pd.concat(data, axis=1)
        self.ranks = pd.concat(ranks, axis=1)
        self.spreads = pd.concat(spread, axis=1)
        self.prices = self.c.get_all_pairs()
        self._data_cache = self.cache.get_pickle() or {}

    def save_cache(self):
        self.cache.set_pickle(self._data_cache)

    @before_exit
    def get_data(self, pairs=['EUR', 'USD'], diffs_shift=[], diffs=[],use_rank=False, use_spread=False) -> pd.DataFrame:
        """get data

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.

        Returns:
            pd.DataFrame: data to fit HMM model
        """

        param_dict = {
            'windows': sorted(self.windows),
            'pairs': sorted(pairs),  # make hashable
            'diffs_shift': sorted(diffs_shift),
            'diffs': sorted(diffs),
            'use_rank': use_rank,
            'use_spread': use_spread,
            'method' : self.method,
        }

        if use_spread:
            if pairs[0] + pairs[1] in [ x[:6] for x in self.spreads.columns]:
                pair2 = pairs[0] + pairs[1]
            else:
                pair2 = pairs[1] + pairs[0]

            pairs= [pair2]
            pairs_windows =list(itertools.product(pairs, self.windows))
            pw2 = ['_'.join(map(str, pw)) for pw in pairs_windows]
            data = self.spreads
        else:
            pairs_windows =list(itertools.product(pairs, self.windows))
            pw2 = ['_'.join(map(str, pw)) for pw in pairs_windows]
            data = self.data

        self.pairs = pairs

        cache_key = str(param_dict)  # or use hash(frozenset(param_dict.items()))
        # Example: check if result is in cache
        if cache_key in self._data_cache:
            self.updated = False
            return self._data_cache[cache_key]
    
        pairs_strength_diffs_shift = []
        for i in diffs_shift:
            pairs_strength_diffs_shift.append(data[pw2].diff(1).shift(i).add_suffix(f'_shift{i}'))

        pairs_strength_diffs = []
        for i in diffs:
            pairs_strength_diffs.append(data[pw2].diff(i).add_suffix(f'_diff{i}'))


        if use_rank and not use_spread:
            df = pd.concat([data[pw2], self.ranks[pw2].add_suffix('_rank')] +pairs_strength_diffs+pairs_strength_diffs_shift, axis=1)
        else:
            df = pd.concat([data[pw2]] + pairs_strength_diffs+pairs_strength_diffs_shift, axis=1)
        
        self._data_cache[cache_key] = df
        return df

    @before_exit
    def plot(self, df:pd.DataFrame, n_components=3, train_ratio=0.7, window_size=300, verbose=False, use_full=False) -> DenseHMM:
        """plot hmm

        Args:
        
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            verbose(bool, optional): used by HMM to display debug message

        Returns:
            DenseHMM: HMM model fit with the given parameters
        """

        split_idx = int(len(df) *train_ratio)
        split_dt = df.index[split_idx]
        print(split_dt)

        df.dropna(inplace=True)
        X_full = df.values.astype(np.float64)  # shape: (n_days, 2 * n_assets)
        step_size = 1  # can increase to skip less overlapping windows
        windows = []
        rolling_index = []
        for i in range(0, len(X_full) - window_size + 1, step_size):
            w = X_full[i:i + window_size]
            windows.append(w)
            rolling_index.append(df.index[i + window_size - 1])

        X_rolling = np.stack(windows)  
        split_idx = rolling_index.index(split_dt)

        param_dict = {
            'n_components': n_components,
            'train_ratio': train_ratio,
            'window_size': window_size,
            'method' : self.method,
        }

        cache_key =  '_'.join(df.columns)  + str(param_dict)
        if cache_key in self._data_cache:
            self.updated = False
            model =  self._data_cache[cache_key]
        else:
            model = DenseHMM([Normal() for _ in range(n_components)], max_iter=100, verbose=verbose)
            model.fit([X_rolling[:split_idx]])
            self._data_cache[cache_key] = model
        
        if use_full: 
            last_probs = model.predict_proba(X_full.reshape(1, X_full.shape[0], X_full.shape[1]))[0][window_size-1:]
        else:
            ret = model.predict_proba(X_rolling)
            last_probs = ret[:, -1, :].numpy() 

        
        df_probs = pd.DataFrame(last_probs, columns=[f"regime_{i}" for i in range(last_probs.shape[1])])
        df_probs.index = pd.to_datetime(rolling_index)  # restore original datetime index


        pairs= self.pairs
        if len(pairs[0]) ==3:
            if pairs[0] + pairs[1] in self.prices.columns:
                pair2 = pairs[0] + pairs[1]
            else:
                pair2 = pairs[1] + pairs[0]
            
            price = self.prices[pair2].loc[rolling_index]
        else:
            pair2 = pairs[0]
            if pair2 not in self.prices.columns:
                price = 1/self.prices[pair2[3:] + pair2[:3]].loc[pd.to_datetime(rolling_index).strftime('%Y-%m-%d')]
            else:
                price = self.prices[pair2].loc[pd.to_datetime(rolling_index).strftime('%Y-%m-%d')]
        
        # Add vertical line at the train/test split index
        fig, ax1 = plt.subplots(figsize=(12, 4))

        colors = ['tab:red', 'tab:orange', 'yellow', 'tab:green', 'tab:blue', 'tab:pink']
        df_probs.plot(ax=ax1 , color=colors, linewidth=3)
        ax1.axvline(x=df_probs.index[split_idx], color='red', linestyle='--', label='Train/Test Split', linewidth=3)
        ax1.set_ylabel("Regime Probabilities")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # Plot  price on second axis
        ax2 = ax1.twinx()
        ax2.plot(df_probs.index, price, color='black', label=f'{pair2} Price', alpha=0.4, linewidth=1)
        ax2.set_ylabel(pair2)
        ax2.legend(loc="upper right")

        plt.title(f"HMM Regime Probabilities with {pair2} Overlay")
        plt.tight_layout()
        plt.show()

        return model
    
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
            states.append(i)

        # pairs= sorted(list(set(x[:3] for x in columns)))
        pairs = self.pairs
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
