import pandas as pd
import numpy as np
from scipy.stats import zscore
from  src.csv.fred  import FRED
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from src.pattern.helper import HELPER, SCALE
from tqdm import tqdm
from src.csv.cache import CACHE
from pykalman import KalmanFilter
from enum import Enum
from typing import List

FR = FRED() # declare only once

class SMOOTHING_METHOD(Enum):
    NONE = 0
    KALMAN = 1
    ROLLING_ZSCORE = 2


def before_exit(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            self.save_cache()
    return wrapper

# Possible combinations
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS', 'XPD', 'XPT']
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS', 'XPD']
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS']
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS', 'NDQ']
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'GAS', 'XPD', 'XPT']
# CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR',]


class CCY_STR:
    def __init__(self, ccy=['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR',]):
        self.CURRENCIES = ccy
        self.fr = FR
        self.get_all_pairs()
        self.cache = CACHE('ccy_str.cache')
        self.full_pnl_cache = self.cache.get_pickle() or {}
        self.current_key = tuple(sorted(self.CURRENCIES))
        if not self.current_key in self.full_pnl_cache:
            self.full_pnl_cache[self.current_key] = {}
        # If the z-score exceeds a certain threshold (e.g., ±5), it can be capped to prevent extreme spikes from distorting your PnL.
        self.cap_threshold = 999

    @property
    def pnl_cache(self):
        return self.full_pnl_cache[self.current_key]
    
    def save_cache(self):
        self.cache.set_pickle(self.full_pnl_cache)

    def get_all_pairs(self, comb=True) -> pd.DataFrame:
        """Compute currency strength index using USD as anchor."""
        # pairs= [ccy+denominated_ccy for ccy in self.CURRENCIES]
        if comb:
            pairs = [''.join(x) for x in itertools.combinations(self.CURRENCIES, 2)]
        else:
            pairs = [''.join(x) for x in itertools.permutations(self.CURRENCIES, 2)]
        data = self.fr.get_pairs(pairs)
        # data.columns = self.CURRENCIES
        return data
    
    def compute_strength(self, prices: pd.DataFrame, window=20) -> pd.DataFrame:
        """Compute log returns and infer currency strength."""
        log_ret = np.log(prices / prices.shift(window))
        return log_ret  # already aligned to non-USD currencies

    def compute_csi(self, window:int =60) -> pd.DataFrame:
        """Compute currency strength index (CSI) for each currency from pairwise returns."""
        prices = self.get_all_pairs()
        returns = self.compute_strength(prices, window=window)
        csi = pd.DataFrame(index=returns.index, columns=self.CURRENCIES)

        for ccy in self.CURRENCIES:
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

    def smoothing_method(self, window: int = 60, window2=None, method: SMOOTHING_METHOD = SMOOTHING_METHOD.ROLLING_ZSCORE):
        """
        Return smoothed Currency Strength Index (CSI) using the selected smoothing method.

        Args:
            window (int, optional): Window size used for smoothing or CSI computation. Defaults to 60.
            window2 (int, optional): only use in rolling_zscore
            method (SMOOTHING_METHOD, optional): Method used to smooth the CSI. 
                - SMOOTHING_METHOD.NONE: Raw CSI without smoothing
                - SMOOTHING_METHOD.KALMAN: Apply Kalman filter to denoise CSI
                - SMOOTHING_METHOD.ROLLING_ZSCORE: Apply cross-currency rolling z-score

        Returns:
            pd.DataFrame: Smoothed CSI dataframe with currencies as columns and dates as index.

        Raises:
            AssertionError: If an invalid smoothing method is provided.
        """
        assert method in SMOOTHING_METHOD, f"Invalid smoothing method: {method}"

        if method == SMOOTHING_METHOD.NONE:
            return self.compute_csi(window=window)
        elif method == SMOOTHING_METHOD.KALMAN:
            return self.kalman_csi(window=window)
        elif method == SMOOTHING_METHOD.ROLLING_ZSCORE:
            return self.rolling_cross_csi_zscore(window=window, window2=window2)

    def kalman_csi(self, window:int =60) -> pd.DataFrame:
        cache_key = ('kalman', window)
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]
        
        csi = self.compute_csi(window=window).dropna()
        
        smoothed_csi = pd.DataFrame(index=csi.index, columns=csi.columns)
        for ccy in csi.columns:
            series = csi[ccy]
            kf = KalmanFilter(
                transition_matrices = [1],
                observation_matrices = [1],
                initial_state_mean = series.iloc[0],
                initial_state_covariance = 1,
                observation_covariance = 1,     # You may tune this per currency
                transition_covariance = 0.01    # Tune as needed
            )

            state_means, _ = kf.filter(series.values)
            # Reassign filtered series back with correct dates
            smoothed_csi.loc[series.index, ccy] = state_means.flatten()

        self.pnl_cache[cache_key] = smoothed_csi 
        return smoothed_csi
    
    def rolling_cross_csi_zscore(self, window:int =60, window2: int = None) -> pd.DataFrame:
        # 2. Cross-currency Z-score (Rolling z-score using mean/std across all currencies)
        """
        """
        """Compute rolling z-score for each currency using window-wide mean/std across all currencies."""
        if window2 is None:
            window2 = window

        cache_key = ('rolling_cross_csi_zscore', window, window2)
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]
        
        df = self.compute_csi(window=window)
        
        zscore_df = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(window2 - 1, len(df)):
            window_df = df.iloc[i - window2 + 1 : i + 1]
            
            global_mean = window_df.values.flatten().mean()
            global_std = window_df.values.flatten().std()

            if global_std == 0:
                print('nan')
                zscore_df.iloc[i] = np.nan
            else:
                zscores = (df.iloc[i] - global_mean) / global_std
                zscore_df.iloc[i] = np.clip(zscores, -self.cap_threshold, self.cap_threshold)

        self.pnl_cache[cache_key] = zscore_df  # Cache the result
        return zscore_df
    
    @before_exit
    def compare_pnl(self, train_ratio, 
                    methods:List[SMOOTHING_METHOD]=[
                        SMOOTHING_METHOD.NONE,
                        SMOOTHING_METHOD.KALMAN,
                        SMOOTHING_METHOD.ROLLING_ZSCORE]):
        
        def weighted_mean(x):
            return x[('return', 'sum')].sum() / x[('return', 'count')].sum()
        
        dffinal= None

        for method in methods:
            cache_key = ('compare_pnl', train_ratio, method)
            if not cache_key in self.pnl_cache:
                out = []
                idx = []
                for window in tqdm(range(10, 700, 10)):
                    df = self.get_pnl_all(window, train_ratio=train_ratio, method=method)
                    if df.empty:
                        break

                    df2 = self.get_pnl_all(window, train_ratio=train_ratio, method=method, test_only=True)
                    if df2.empty:
                        break
                    
                    idx.append(window)
                    # Calculate sum of returns and count for weighted mean
                    xx = df.groupby(['ccy', 'method']).agg({'return': ['sum', 'count']})
                    xx2 = df2.groupby(['ccy', 'method']).agg({'return': ['sum', 'count']})

                    # Calculate weighted mean: sum of returns / total count
                    full_means = [weighted_mean(xx.xs(key=method, level=1)) ]
                    test_means = [weighted_mean(xx2.xs(key=method, level=1))]

                    out.append(full_means + test_means)

                columns = ['full_%s' % method, 'test_%s' % method]
                self.pnl_cache[cache_key] = pd.DataFrame(out, columns=columns, index=idx)
        
            
            dfout = self.pnl_cache[cache_key]

            if dffinal is None:
                dffinal = dfout
            else:
                dffinal = dffinal.merge(dfout, how='inner', left_index=True, right_index=True)
        title = '_'.join(map(str, [train_ratio] + [methods]))
        HELPER.plot_chart(dffinal, title=title)
        return dffinal

    @before_exit
    def get_scaled_params(self, train_ratio=0.7, method:SMOOTHING_METHOD=SMOOTHING_METHOD.ROLLING_ZSCORE) -> pd.DataFrame:
        """This generates mean and standard deviation of the Standard Scaler

        Args:
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.

        Returns:
            pd.DataFrame: DataFrame containing currency pair, window, train ratio, mean, and std.
        """
        cache_key = ('get_scaled_params', train_ratio, method)
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]
        
        out = []
        prices = self.get_all_pairs()
        for window in tqdm(range(10, 700, 10)):
            csi2 = self.smoothing_method(window=window, window2=window, method=method)

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
        return df_result
    
    @before_exit
    def get_pnl_all(self, window=200, window2=None, train_ratio=0.7, method:SMOOTHING_METHOD=SMOOTHING_METHOD.ROLLING_ZSCORE, test_only=False) -> pd.DataFrame:
        """This generates PnL for all ccy

        Args:
            window (int, optional): window used for compute strength. Defaults to 200.
            window2 (_type_, optional): window used for compute rolling csi zscore. Defaults to None. If None is used, it will use window instead.
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.
            method (list, optional): smoothing method. Options 'rolling_zscore', 'kalman', None. Defaults to rolling_zscore
            test_only (bool, optional): To generate PnL for test only. Defaults to False.

        Returns:
            pd.DataFrame: PnL dataframe
        """
        if not window2:
            window2 = window
        
        # Create a unique cache key based on function parameters
        cache_key = (window, window2, train_ratio, method, test_only)
        
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]

        out = []
        prices = self.get_all_pairs()
        csi2 = self.smoothing_method(window=window, window2=window2, method=method)
        ranks = csi2.rank(axis=1, ascending=True, method='min')

        for pair in prices.columns:
            ccy1, ccy2 = pair[:3], pair[3:]

            spread = csi2[ccy1] - csi2[ccy2]

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
                        'method': method,
                        'return': float(prices.loc[o2, pair] / prices.loc[o1, pair] - 1) * direction
                    })

        df_result = pd.DataFrame(out)
        self.pnl_cache[cache_key] = df_result  # Cache the result
        return df_result

        # return pd.DataFrame(out, columns=['year', 'ccy', 'entry_idx', 'exit_idx', 'entry_price', 'exit_price', 'spread', 'ccy1_rank', 'ccy2_rank', 'method', 'return'])

    
    def mt5_export(self, window=200, window2=None, train_ratio=0.7, method:SMOOTHING_METHOD=SMOOTHING_METHOD.ROLLING_ZSCORE) -> pd.DataFrame:
        """export csi to mt5 format

        Args:
            window (int, optional): window used for compute strength. Defaults to 200.
            window2 (_type_, optional): window used for compute rolling csi zscore. Defaults to None. If None is used, it will use window instead.
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.
            method (SMOOTHING_METHOD, optional): smoothing method for csi. Defaults to SMOOTHING_METHOD.ROLLING_ZSCORE.

        Returns:
            pd.DataFrame: csi of all pairs in mt5 format
        """
        prices = self.get_all_pairs()
        csi2 = self.smoothing_method(window=window, window2=window2, method=method)
        ret =[]
        for pair in prices.columns:
            ccy1, ccy2 = pair[:3], pair[3:]
            spread = csi2[ccy1] - csi2[ccy2]
            scaled_df = HELPER.get_scaled(spread, train_ratio=train_ratio, test_only=False)
            ret.append(scaled_df)
        
        dfout = pd.concat(ret, axis=1)
        dfout.columns = prices.columns
        dfout.reset_index(inplace=True)

        epoch = pd.Timestamp("1970-01-01")
        dfout.insert(0, 'timestamp', (pd.to_datetime(dfout['index'])-epoch).dt.days)
        
        dfout['index'] = pd.to_datetime(dfout['index']).dt.strftime('%Y.%m.%d %H:%M:%S')
        dfout.set_index('index',drop=True, inplace=True)

        fname = '%s_%s_%s.csv' % (window, train_ratio, method)
        dfout.dropna().to_csv(f'files/mt5_export/{fname}')
        return dfout

    @before_exit
    def get_pending_trades_all(self, window=200, window2=None, train_ratio=0.7, method:SMOOTHING_METHOD=SMOOTHING_METHOD.ROLLING_ZSCORE) -> pd.DataFrame:
        """This return pending trades

        Args:
            window (int, optional): window used for compute strength. Defaults to 200.
            window2 (_type_, optional): window used for compute rolling csi zscore. Defaults to None. If None is used, it will use window instead.
            train_ratio (float, optional): train ratio for train-test split. Defaults to 0.7.
            method (list, optional): Spread and Zspread. Defaults to ['spread'].
            test_only (bool, optional): To generate PnL for test only. Defaults to False.

        Returns:
            pd.DataFrame: pending trades dataframe
        """
        if not window2:
            window2 = window
        
        # Create a unique cache key based on function parameters
        cache_key = ("pending", window, window2, train_ratio, method)
        
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]

        out = []
        prices = self.get_all_pairs()
        csi2 = self.smoothing_method(window=window, window2=window2, method=method)
        ranks = csi2.rank(axis=1, ascending=True, method='min')

        for pair in prices.columns:
            ccy1, ccy2 = pair[:3], pair[3:]
            spread = csi2[ccy1] - csi2[ccy2]

            entry2, exit2 = self.get_threshold_crosses(spread, train_ratio=train_ratio, threshold=2)
            if len(entry2) != len(exit2):
                o1 = entry2[-1]
                direction = -1 if spread.loc[o1] > 0 else 1
                out.append({
                    'year': o1[:4],
                    'ccy': pair,
                    'entry_idx': o1,
                    'entry_price': float(prices.loc[o1, pair]),
                    'spread': spread.loc[o1],
                    'ccy1_rank': int(ranks.loc[o1, ccy1]),
                    'ccy2_rank': int(ranks.loc[o1, ccy2]),
                    'method': method,
                    'direction' : direction
                })

        df_result = pd.DataFrame(out)
        self.pnl_cache[cache_key] = df_result  # Cache the result
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

    def get_spread(self, pair: str, window: int, window2: int=None, method:SMOOTHING_METHOD= SMOOTHING_METHOD.ROLLING_ZSCORE, 
                   train_ratio=0.7, visualize=False):
        """This return rolling csi spread for selected ccy

        Args:
            pair (str): Selected currency pair
            window (int): used in csi calculation
            window2 (int): used in rolling csi calculation
        """
        if window2 is None:
            window2 = window
        prices = self.get_all_pairs()
        csi2 = self.smoothing_method(window=window, window2=window2, method=method)
        ccy1, ccy2 = pair[:3], pair[3:]

        if not pair in prices.columns:
            prices[pair] = 1/ prices[pair[3:] + pair[:3]]

        spread = csi2[ccy1] - csi2[ccy2]
        if visualize:
            prices['%s_ma' % pair] = prices[pair].rolling(window=window).mean()
            print(HELPER.get_scaled_mean_std(spread, train_ratio=train_ratio))
            HELPER.plot_chart(prices[[pair, '%s_ma' % pair]], spread, scale=SCALE.OTHER, hline=2, train_ratio=train_ratio)  # Choose what to plot
        return spread

class CCY_STR_DEBUG(CCY_STR):
    def __init__(self, pair: str, window: int, window2: int=None, train_ratio=0.7):
        super().__init__()
        self.pair = pair
        self.window =  window
        if window2 is None:
            window2 = window
        self.window2 = window2
        self.train_ratio = train_ratio
        self.prices = self.get_all_pairs()

    def debug_spread(self):
        self.get_spread(self.pair, self.window, self.window2, train_ratio=self.train_ratio, visualize=True)

    def debug_pair(self, dt1, dt2, extra_window=0, scale=SCALE.OTHER):
        """
        Debug and visualize the spread between two currencies over a specified time range.

        This function extracts and plots the spread for the given currency pair within 
        the specified date range, along with its scaled version using a Standard Scaler. 
        It can include additional data before and after the range using `extra_window` for context.

        Args:
            dt1 (str, optional): 
                Start date for the plot in 'YYYY-MM-DD' format. 
            dt2 (str, optional): 
                End date for the plot in 'YYYY-MM-DD' format. 
            extra_window (int, optional): 
                Number of additional data points to include before and after the specified date range 
                for better context. Defaults to 0.

        Behavior:
            - Fetches the spread using `get_spread()` for the specified currency pair and windows.
            - Extracts the range between `dt1` and `dt2` with optional extra data using `extra_window`.
            - Applies scaling using `HELPER.get_scaled()` to plot the normalized spread.
            - Visualizes the actual and scaled spreads using `HELPER.plot_chart()` with an optional threshold line at ±2.

        Example:
            >>> self.debug_pair(dt1='2000-01-01', dt2='2000-06-01', extra_window=10)

        """
        spread = self.get_spread(self.pair, self.window, self.window2, train_ratio=self.train_ratio)
        pos1 = spread.index.get_loc(dt1) - extra_window
        pos2 = spread.index.get_loc(dt2) + extra_window
        HELPER.plot_chart(self.prices[self.pair], spread, scale=scale, hline=2, train_ratio=self.train_ratio, window=(pos1, pos2) )  

    def debug_pair_v2(self, idx: int, extra_window: int=0, scale:SCALE=SCALE.OTHER):
        """        
        Debug and visualize the spread between two currencies over a specified time range by index of pnl_all

        Args:
            idx (int): index in get_pnl_all
            extra_window (int, optional): add extra window before and after chart . Defaults to 0.
            scale (SCALE, optional): see plot_chart SCALE description. Defaults to SCALE.OTHER.
        """
        
        spread = self.get_spread(self.pair, self.window, self.window2, train_ratio=self.train_ratio)
        pnl = self.get_pnl_all(window=self.window, window2=self.window2, train_ratio=self.train_ratio)

        inverse = False
        pair = self.pair
        if self.pair not in pnl.ccy.unique():
            inverse = True
            pair = self.pair[3:] + self.pair[:3]

        if self.pair not in self.prices:
            self.prices[self.pair] = 1/self.prices[pair]
        
        pnl2 = pnl[pnl.ccy==pair].copy() 

        assert idx < pnl2.shape[0], "Max idx is %s" % pnl2.shape[0]

        dt1 = pnl2.iloc[idx].entry_idx
        dt2 = pnl2.iloc[idx].exit_idx
        pos1 = spread.index.get_loc(dt1) - extra_window
        pos2 = spread.index.get_loc(dt2) + 1+ extra_window
        
        if inverse:
            pnl2['ccy'] =  self.pair
            pnl2['entry_price'] =  1/pnl2['entry_price']
            pnl2['exit_price'] =  1/pnl2['exit_price']
            pnl2['spread']  = - pnl2['spread']
            ccy_rank = pnl2['ccy1_rank']
            pnl2['ccy1_rank']  = pnl2['ccy2_rank']
            pnl2['ccy2_rank'] = ccy_rank
            
        print(pnl2.iloc[idx])
        self.prices['%s_ma' % self.pair] = self.prices[self.pair].rolling(window=self.window).mean()
        HELPER.plot_chart(self.prices[[self.pair, '%s_ma' % self.pair]], spread, scale=scale, hline=2, train_ratio=self.train_ratio, window=(pos1, pos2) )  
        return pnl2
    
if __name__ == '__main__':
    c = CCY_STR()
    c.mt5_export()
    # prices = c.get_all_pairs()  
    # dt = '2017-01-01'
    # pair ='USDJPY'
    # window = 100

    # csi = c.compute_csi(window=window)
    # csi2 = csi[csi.index > dt]
    # prices2 = prices[prices.index > dt].copy()
    # if not pair in prices2.columns:
    #     prices2[pair] = 1/ prices2[pair[3:] + pair[:3]]
        
    # spread = csi2[pair[:3]] -csi2[pair[3:]]
    # zspread = c.rolling_zscore(spread, window=window)

    # HELPER.plot_chart(prices2[pair], csi2)  # Choose what to plot