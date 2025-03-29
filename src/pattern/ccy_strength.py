import pandas as pd
import numpy as np
from scipy.stats import zscore
from  src.csv.fred  import FRED
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from src.pattern.helper import HELPER
from tqdm import tqdm
from src.csv.cache import CACHE

CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR']

class CCY_STR:
    def __init__(self):
        self.fr = FRED()
        self.get_all_pairs()
        self.cache = CACHE('ccy_str.cache')
        self.pnl_cache = self.cache.get_pickle() or {}

    def get_all_pairs(self) -> pd.DataFrame:
        """Compute currency strength index using USD as anchor."""
        # pairs= [ccy+denominated_ccy for ccy in CURRENCIES]
        pairs = [''.join(x) for x in itertools.combinations(CURRENCIES, 2)]
        data = self.fr.get_pairs(pairs)
        # data.columns = CURRENCIES
        return data
    
    def compute_strength(self, prices: pd.DataFrame, window=20) -> pd.DataFrame:
        """Compute log returns and infer currency strength."""
        log_ret = np.log(prices / prices.shift(window))
        return log_ret  # already aligned to non-USD currencies

    def compute_csi(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute currency strength index (CSI) for each currency from pairwise returns."""
        csi = pd.DataFrame(index=returns.index, columns=CURRENCIES)

        for ccy in CURRENCIES:
            ccy_returns = []
            for pair in returns.columns:
                base, quote = pair[:3], pair[3:]

                if ccy == base:
                    ccy_returns.append(returns[pair])
                elif ccy == quote:
                    ccy_returns.append(-returns[pair])

            if ccy_returns:
                csi[ccy] = pd.concat(ccy_returns, axis=1).mean(axis=1)

        return csi

    def rolling_zscore(self, series: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling z-score of a series."""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std

    # === Z-Score Interpretations ===

    # 1. Per-currency Z-score (Rolling z-score for each currency individually)
    # USD z = +2 → USD is stronger than its own recent history
    # EUR z = -2 → EUR is weaker than its own recent history
    # NOTE: This doesn't tell you how USD compares to EUR right now — only vs itself

    # 2. Cross-currency Z-score (Rolling z-score using mean/std across all currencies)
    # USD z = +2 → USD is stronger than the FX basket over the rolling window
    # EUR z = -2 → EUR is very weak vs the same basket
    # This helps compare currencies directly, as they share the same μ and σ at each time step

    # 3. Spread Z-score (Z-score of the CSI spread between two currencies)
    # Spread (USD - JPY) z = +2.5 → USD strength vs JPY is extreme
    # This is ideal for entry/exit timing in a mean-reversion trade
    # Example: z > 2 → short USDJPY (short strong, long weak)


    def rolling_zscore_df(self, df: pd.DataFrame, window: int=60)-> pd.DataFrame:
        # 1. Per-currency Z-score (Rolling z-score for each currency individually)
        return df.apply(lambda col: self.rolling_zscore(col, window=window))

    def rolling_cross_csi_zscore(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        # 2. Cross-currency Z-score (Rolling z-score using mean/std across all currencies)
        """Compute rolling z-score for each currency using window-wide mean/std across all currencies."""
        zscore_df = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(window - 1, len(df)):
            window_df = df.iloc[i - window + 1 : i + 1]
            
            global_mean = window_df.values.flatten().mean()
            global_std = window_df.values.flatten().std()

            if global_std == 0:
                print('nan')
                zscore_df.iloc[i] = np.nan
            else:
                zscore_df.iloc[i] = (df.iloc[i] - global_mean) / global_std

        return zscore_df
    
    def compare_pnl(self, train_ratio, method=['spread']):
        key = ''.join([str(train_ratio)] + sorted(method))

        def weighted_mean(x):
            return x[('return', 'sum')].sum() / x[('return', 'count')].sum()
        
        out = []
        idx = []
        for window in tqdm(range(10, 700, 10)):
            df = self.get_pnl(window, train_ratio=train_ratio, method=method)
            if df.empty:
                break

            df2 = self.get_pnl(window, train_ratio=train_ratio, method=method, test_only=True)
            if df2.empty:
                break
            
            idx.append(window)
            # Calculate sum of returns and count for weighted mean
            xx = df.groupby(['ccy', 'method']).agg({'return': ['sum', 'count']})
            xx2 = df2.groupby(['ccy', 'method']).agg({'return': ['sum', 'count']})

            # Calculate weighted mean: sum of returns / total count
            full_means = [weighted_mean(xx.xs(key=m, level=1)) for m in method]
            test_means = [weighted_mean(xx2.xs(key=m, level=1)) for m in method]

            out.append(full_means + test_means)

        columns = ['full_%s' % m for m in method] + ['test_%s' % m for m in method]
        dfout = pd.DataFrame(out, columns=columns, index=idx)
        HELPER.plot_chart(dfout, title=key)
        return dfout

    def get_scaled_params(self, train_ratio=0.7) -> pd.DataFrame:
        """This generates mean and standard deviation of the Standard Scaler

        Args:
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.

        Returns:
            pd.DataFrame: DataFrame containing currency pair, window, train ratio, mean, and std.
        """
        cache_key = ('get_scaled_params', train_ratio)
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]
        
        out = []
        prices = self.get_all_pairs()
        for window in tqdm(range(10, 700, 10)):
            returns = self.compute_strength(prices, window=window)
            csi = self.compute_csi(returns)
            csi2 = self.rolling_cross_csi_zscore(csi, window=window)

            for pair in prices.columns:
                ccy1, ccy2 = pair[:3], pair[3:]

                spread = csi2[ccy1] - csi2[ccy2]
                mean, std = HELPER.get_scaled_mean_std(spread, train_ratio=train_ratio)
                if mean is None or std is None:
                    break
                out.append({
                    'ccy': pair,
                    'window' : window,
                    'train_ratio' : train_ratio,
                    'mean' : mean,
                    'std'  : std
                     })
                
        df_result = pd.DataFrame(out)
        self.pnl_cache[cache_key] = df_result  # Cache the result
        self.cache.set_pickle(self.pnl_cache)
        return df_result
    
    def get_pnl(self, window=200, window2=None, train_ratio=0.7, method=['spread'], test_only=False) -> pd.DataFrame:
        """This generates PnL

        Args:
            window (int, optional): window used for compute strength. Defaults to 200.
            window2 (_type_, optional): window used for compute rolling csi zscore. Defaults to None. If None is used, it will use window instead.
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.
            method (list, optional): Spread and Zspread. Defaults to ['spread'].
            test_only (bool, optional): To generate PnL for test only. Defaults to False.

        Returns:
            pd.DataFrame: PnL dataframe
        """
        if not window2:
            window2 = window
        
        # Create a unique cache key based on function parameters
        cache_key = (window, window2, train_ratio, tuple(sorted(method)), test_only)
        
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]

        out = []
        prices = self.get_all_pairs()
        returns = self.compute_strength(prices, window=window)
        csi = self.compute_csi(returns)
        csi2 = self.rolling_cross_csi_zscore(csi, window=window2)
        ranks = csi2.rank(axis=1, ascending=True, method='min')

        for pair in prices.columns:
            ccy1, ccy2 = pair[:3], pair[3:]

            spread = csi2[ccy1] - csi2[ccy2]

            if 'zspread' in method:
                zspread = self.rolling_zscore(spread, window=window)
                entry1, exit1 = self.get_threshold_crosses(zspread, train_ratio=train_ratio, threshold=2, test_only=test_only)
                if entry1 is not None:
                    for o1, o2 in zip(entry1, exit1):
                        direction = -1 if zspread.loc[o1] > 0 else 1
                        out.append({
                            'year': o1[:4],
                            'ccy': pair,
                            'entry_idx': o1,
                            'exit_idx': o2,
                            'entry_price': float(prices.loc[o1, pair]),
                            'exit_price': float(prices.loc[o2, pair]),
                            'spread': zspread.loc[o1],
                            'ccy1_rank': int(ranks.loc[o1, ccy1]),
                            'ccy2_rank': int(ranks.loc[o1, ccy2]),
                            'method': 'zspread',
                            'return': float(prices.loc[o2, pair] / prices.loc[o1, pair] - 1) * direction
                        })

            if 'spread' in method:
                entry2, exit2 = self.get_threshold_crosses(spread, train_ratio=train_ratio, threshold=2, test_only=test_only)
                if entry2 is not None:
                    for o1, o2 in zip(entry2, exit2):
                        direction = -1 if spread.loc[o1] > 0 else 1
                        out.append({
                            'year': o1[:4],
                            'ccy': pair,
                            'entry_idx': o1,
                            'exit_idx': o2,
                            'entry_price': float(prices.loc[o1, pair]),
                            'exit_price': float(prices.loc[o2, pair]),
                            'spread': spread.loc[o1],
                            'ccy1_rank': int(ranks.loc[o1, ccy1]),
                            'ccy2_rank': int(ranks.loc[o1, ccy2]),
                            'method': 'spread',
                            'return': float(prices.loc[o2, pair] / prices.loc[o1, pair] - 1) * direction
                        })

        df_result = pd.DataFrame(out)
        self.pnl_cache[cache_key] = df_result  # Cache the result
        self.cache.set_pickle(self.pnl_cache)
        return df_result

        # return pd.DataFrame(out, columns=['year', 'ccy', 'entry_idx', 'exit_idx', 'entry_price', 'exit_price', 'spread', 'ccy1_rank', 'ccy2_rank', 'method', 'return'])

    def get_threshold_crosses(self, df: pd.Series, train_ratio=0.7, threshold=2.0, reset_level=0.0, test_only=False):
        """
        Detects points where df crosses threshold and resets after returning to reset_level.
        
        Returns:
            entries: list of entry timestamps (df hits threshold)
            exits: list of exit timestamps (df returns to reset_level)
        """
        entries = []
        exits = []
        active = False  # whether we're inside a "position"
        scaled_df = HELPER.get_scaled(df, train_ratio=train_ratio, test_only=test_only)
        if scaled_df is None:
            return None, None
        
        scaled_df = scaled_df[0]
        for i in range(1, len(scaled_df)):
            val = scaled_df.iloc[i]
            prev = scaled_df.iloc[i - 1]
            idx = scaled_df.index[i]

            if not active:
                # Looking for threshold hit (positive or negative)
                if prev < threshold <= val or prev > -threshold >= val:
                    entries.append(idx)
                    active = True
            else:
                # Already in a position, wait for reset to ~0
                if prev * val < 0 or abs(val) < 0.1:  # crossed zero or very close
                    exits.append(idx)
                    active = False

        return entries, exits



if __name__ == '__main__':
    c = CCY_STR()
    prices = c.get_all_pairs()  
    dt = '2017-01-01'
    pair ='USDJPY'
    window = 100
    returns = c.compute_strength(prices, window=window)
    csi = c.compute_csi(returns)
    csi2 = csi[csi.index > dt]
    prices2 = prices[prices.index > dt].copy()
    if not pair in prices2.columns:
        prices2[pair] = 1/ prices2[pair[3:] + pair[:3]]
        
    spread = csi2[pair[:3]] -csi2[pair[3:]]
    zspread = c.rolling_zscore(spread, window=window)

    HELPER.plot_chart(prices2[pair], csi2)  # Choose what to plot