
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from src.pattern.ccy_strength import CCY_STR, SMOOTHING_METHOD
from src.csv.cache import CACHE2

class CORR:
    def __init__(self):
        pass

    def get_cov1(self, fxraw: pd.DataFrame, window=200, visualize=False):
        """ using pandas default covariance

        Args:
            fxraw (pd.DataFrame): two asset prices
            window (int, optional): rolling window for covariance. Defaults to 200.
            visualize (bool, optional): to display chart. Defaults to False.
        """

        
        fx = fxraw.copy()
        pairs = itertools.permutations(fx.columns, 2)
        log_fx = np.log(fx)

        for pair1, pair2 in pairs:
            rolling_cov_price = log_fx[pair1].rolling(window).cov(log_fx[pair2])
            rolling_var_price = log_fx[pair2].rolling(window).var()

            beta_price = rolling_cov_price / rolling_var_price
            alpha_price = (
                log_fx[pair1].rolling(window).mean()
                - beta_price * log_fx[pair2].rolling(window).mean()
            )

            proj_log_price = alpha_price + beta_price * log_fx[pair2]
            fx['projected_%s' % pair1] = np.exp(proj_log_price)

            if visualize:
                self.plot_chart(fx[[pair1, 'projected_%s' % pair1]], rolling_cov_price, scale=True)

    def get_cov2(self, fxraw: pd.DataFrame, window=200, visualize=False):
        """ using pandas log diff

        Args:
            fxraw (pd.DataFrame): two asset prices
            window (int, optional): rolling window for covariance. Defaults to 200.
            visualize (bool, optional): to display chart. Defaults to False.
        """

        
        fx = fxraw.copy()
        pairs = itertools.permutations(fx.columns, 2)
        returns = np.log(fx).diff().dropna()
        window=200
        rolling_cov = returns.rolling(window,min_periods=window).cov()
        rolling_var = returns.rolling(window,min_periods=window).var()

        for pair1, pair2 in pairs:
            cov1 = rolling_cov[pair1].xs(pair2, level=1)
            rolling_beta = cov1 / rolling_var[pair2]
            log_beta = rolling_beta * returns[pair2]
            projected_price = fx[pair1].shift(1) * np.exp(log_beta)
            fx['projected_%s' % pair1] = projected_price

            if visualize:
                self.plot_chart(fx[[pair1, 'projected_%s' % pair1]], cov1)

    def plot_chart(self, f1: pd.DataFrame, f2: pd.DataFrame, scale=False):
        """ plot chart

        Args:
            f1 (pd.DataFrame): this is to be plotted on axes 1
            f2 (pd.DataFrame): this is to be plotted on axes 2
            scale: apply standard scaler for f2
        """
        fig, ax1 = plt.subplots(figsize=(13, 4))
        # Plot actual and modeled oil prices on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        f1.plot(color=['blue', 'orange'], ax=ax1, legend=False)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis 
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('spread', color=color)  # we already handled the x-label with ax1
        
        if scale:
            scaler = StandardScaler()
            if isinstance(f2, pd.Series):
                prescale = f2.to_frame()
            else:
                prescale = f2

            df_scaled = pd.DataFrame(
                scaler.fit_transform(prescale),
                index=prescale.index,
                columns=prescale.columns)
            df_scaled.plot( color='green', ax=ax2, legend=False, linestyle=':')
            ax2.axhline(-2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
            ax2.axhline(2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
        else:
            f2.plot( color='green', ax=ax2, legend=False, linestyle=':')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0

        # Adding a title and legend
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('correlation')
        fig.legend(loc="upper left", bbox_to_anchor=(1.02,1), bbox_transform=ax1.transAxes)
        plt.show()


my_cache =  CACHE2('corr2.cache', ['windows', 'CURRENCIES', 'days_range'])

class CORR2:
    def __init__(self, ccys=['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS'], simulation=False, force_reset=False):
        
        self.windows = range(10, 1000, 10)
        self.CURRENCIES = ccys
        self.c = CCY_STR(ccys)
        self.cache = CACHE2('corr2.cache')
        self.sim = SIMULATION()
        self.simulation=simulation
        self.force_reset =force_reset
        
        if simulation:    
            self.force_reset= True
            self.days_range  = range(10, 100, 10)
            self.df = self.sim.df
        else:
            self.days_range  = range(10, 1000, 10)
            self.df = self.get_all_pairs()

    @property
    def all_pairs(self):
        return self.df.columns
    
    @my_cache
    def get_all_pairs(self):
        return self.c.get_all_pairs()
    
    @my_cache
    def _log_return_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logreturn_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_range_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].rolling(window=x+1).max() / self.df[ccy].rolling(window=x+1).min()).shift(-x)
            s.name = f"{ccy}_logrange_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_max_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].rolling(window=x+1).max().shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logbull_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_min_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = -np.log(self.df[ccy].rolling(window=x+1).min().shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logbear_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)


    def get_future(self):
        dfs = []
        for ccy in self.all_pairs:
            dfs.append(self._log_return_future(ccy))
            dfs.append(self._log_range_future(ccy))
            dfs.append(self._log_max_future(ccy))
            dfs.append(self._log_min_future(ccy))
        self.future = pd.concat(dfs, axis=1)
    
    def get_feature(self):
        dfs = []
        for ccy in self.all_pairs:
            dfs.append(self._feature_meanret_oneday(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_meanret_xday(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_range(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_std_oneday(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_std_xday(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_rsi(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_hurst(ccy, force_reset = self.force_reset))
            if not self.simulation:
                dfs.append(self._feature_csi(ccy))
        self.features=  pd.concat(dfs, axis=1)
    
    def apply_cross_sectional(self):
        self.features2 = self.features.copy()
        unique_cols = set([ '_'.join(x.split('_')[1:]) for x in self.features2.columns])
        for col in unique_cols:
            if 'featrsi' in col:
                continue
            if 'feathurst' in col:
                continue
            selected_cols = [x for x in self.features2.columns if col in x]
            self.features2[selected_cols] = (
                self.features2[selected_cols]
                .sub(self.features2[selected_cols].mean(axis=1), axis=0)
                .div(self.features2[selected_cols].std(axis=1, ddof=0), axis=0)
            )
            
    def feature(self, key, mode=1):
        keys = key.split(',')
        if mode ==1:
            return self.features[[ x for x in self.features.columns if all(k in x for k in keys) ]]
        else:
            return self.features2[[ x for x in self.features2.columns if all(k in x for k in keys) ]]


    def demo(self, keywords=[], mode=1):

        for col in set([x.split('_')[1] for x in self.features.columns]):
            plt.figure()            
            ax = self.feature(f"{','.join(([col] + keywords))}", mode=mode).plot()
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.title(f'{col}')


    @my_cache
    def _feature_hurst(self, ccy: str, force_reset=False) -> pd.DataFrame:
        def hurst_rs(series: pd.Series) -> float:
            """Compute the Hurst exponent using rescaled range (R/S) method"""
            series = series.dropna()

            mean = series.mean()
            dev = series - mean
            cum_dev = dev.cumsum()
            R = cum_dev.max() - cum_dev.min()
            S = series.std()

            if S == 0:
                return np.nan
            return np.log(R / S)

        dfs = []
        for x in self.days_range:
            s = self.df[ccy].rolling(window=x).apply(hurst_rs, raw=False) / np.log(x)
            s.name = f"{ccy}_feathurst_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_range(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            max = self.df[ccy].rolling(window=x).max()
            min = self.df[ccy].rolling(window=x).min()
            s = np.log(max / min)
            s.name = f"{ccy}_featrange_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)


    @my_cache
    def _feature_meanret_oneday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        logret = np.log(self.df[ccy] / self.df[ccy].shift(1))
        for x in self.days_range:
            s = logret.rolling(window=x).mean()
            s.name = f"{ccy}_featmean1_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_meanret_xday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        
        for x in self.days_range:
            logret = np.log(self.df[ccy] / self.df[ccy].shift(x-1))
            s = logret.rolling(window=x).mean()
            s.name = f"{ccy}_featmeanx_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_std_oneday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        logret = np.log(self.df[ccy] / self.df[ccy].shift(1))
        for x in self.days_range:
            s = logret.rolling(window=x).std()
            s.name = f"{ccy}_featstd1_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    
    @my_cache
    def _feature_std_xday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            logret = np.log(self.df[ccy] / self.df[ccy].shift(x-1))
            s = logret.rolling(window=x).std()
            s.name = f"{ccy}_featstdx_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @my_cache
    def _feature_rsi(self, ccy: str, force_reset = False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = self.compute_rsi(self.df[ccy], x)
            s.name = f"{ccy}_featrsi_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    
    @my_cache
    def all_csi(self, window):
        return self.c.compute_csi(window=window)
    
    @my_cache
    def _feature_csi(self, pair: str) -> pd.DataFrame:
        dfs = []
        ccy1, ccy2 = pair[:3], pair[3:]

        for x in self.days_range:
            # ytee 27 June 2025: we use x-1 instead of x to standardize the behavior of features.
            # CSI is computed as price_t / price_t.shift(x), where price_t.shift(x) = price at (t - x),
            # effectively covering the period [t - x, t].
            # This aligns with a standard rolling window of size w, where w = x + 1,
            # which also spans [t - w + 1, t] — for example, a 2-day rolling window covers [t - 1, t].
            csi = self.all_csi(x-1)
            s = csi[ccy1] - csi[ccy2]
            s.name = f"{pair}_featcsi_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

class SIMULATION:
    def __init__(self):
        n = 100
        t = np.arange(n)

        # Scenario 1: Trending slowly (no noise)
        trend1 = np.linspace(0, 10, n)

        # Scenario 2: Trending fast (no noise)
        trend2 = np.linspace(0, 20, n)

        # Scenario 3: Ranging zigzag with more frequent reversals
        num_zigs = 10
        points_per_zig = n // (2 * num_zigs)
        zig = np.linspace(0, 10, points_per_zig)
        zag = np.linspace(10, 0, points_per_zig)
        full_cycle = np.concatenate([zig, zag])
        range_market = np.tile(full_cycle, num_zigs)[:n]

        self.df = pd.DataFrame({
            'Slow': trend1,
            'Fast': trend2,
            'Range': range_market
        })
        self.df += 5

class CORR3:
    def __init__(self, ccys=['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS'], simulation=False, force_reset=False):
        
        self.windows = range(10, 1000, 10)
        self.CURRENCIES = ccys
        self.c = CCY_STR(ccys)
        self.cache = CACHE2('corr3.cache')
        self.sim = SIMULATION()
        self.simulation=simulation
        self.force_reset =force_reset
        
        if simulation:    
            self.force_reset= True
            self.days_range  = range(10, 100, 10)
            self.df = self.sim.df
        else:
            self.days_range  = range(10, 1000, 10)
            self.df = self.get_all_pairs(force_reset)

    @property
    def all_pairs(self):
        return self.df.columns
    
    @my_cache
    def get_all_pairs(self, force_reset=False):
        df = self.c.compute_csi()
        dfs = []
        for col in df:
            cumulative_log_return = np.cumsum(df[col])
            start_price = 100
            price_index = start_price * np.exp(cumulative_log_return)
            dfs.append(price_index)

        return pd.concat(dfs, axis=1)
        
    
    @my_cache
    def _log_return_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logreturn_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_range_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].rolling(window=x+1).max() / self.df[ccy].rolling(window=x+1).min()).shift(-x)
            s.name = f"{ccy}_logrange_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_max_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = np.log(self.df[ccy].rolling(window=x+1).max().shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logbull_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    @my_cache
    def _log_min_future(self, ccy: str) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = -np.log(self.df[ccy].rolling(window=x+1).min().shift(-x) / self.df[ccy])
            s.name = f"{ccy}_logbear_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)


    def get_future(self):
        dfs = []
        for ccy in self.all_pairs:
            dfs.append(self._log_return_future(ccy))
            dfs.append(self._log_range_future(ccy))
            dfs.append(self._log_max_future(ccy))
            dfs.append(self._log_min_future(ccy))
        self.future = pd.concat(dfs, axis=1)
    
    def get_feature(self):
        dfs = []
        for ccy in self.all_pairs:
            dfs.append(self._feature_meanret_oneday(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_meanret_xday(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_range(ccy, force_reset =  self.force_reset))
            dfs.append(self._feature_std_oneday(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_std_xday(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_rsi(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_hurst(ccy, force_reset = self.force_reset))
            dfs.append(self._feature_csi(ccy))
        self.features=  pd.concat(dfs, axis=1)
    
    def apply_cross_sectional(self):
        self.features2 = self.features.copy()
        unique_cols = set([ '_'.join(x.split('_')[1:]) for x in self.features2.columns])
        for col in unique_cols:
            if 'featrsi' in col:
                continue
            if 'feathurst' in col:
                continue
            selected_cols = [x for x in self.features2.columns if col in x]
            self.features2[selected_cols] = (
                self.features2[selected_cols]
                .sub(self.features2[selected_cols].mean(axis=1), axis=0)
                .div(self.features2[selected_cols].std(axis=1, ddof=0), axis=0)
            )
            
    def feature(self, key, mode=1):
        keys = key.split(',')
        if mode ==1:
            return self.features[[ x for x in self.features.columns if all(k in x for k in keys) ]]
        else:
            return self.features2[[ x for x in self.features2.columns if all(k in x for k in keys) ]]


    def demo(self, keywords=[], mode=1):

        for col in set([x.split('_')[1] for x in self.features.columns]):
            plt.figure()            
            ax = self.feature(f"{','.join(([col] + keywords))}", mode=mode).plot()
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.title(f'{col}')


    @my_cache
    def _feature_hurst(self, ccy: str, force_reset=False) -> pd.DataFrame:
        def hurst_rs(series: pd.Series) -> float:
            """Compute the Hurst exponent using rescaled range (R/S) method"""
            series = series.dropna()
            if len(series) < 20:
                return np.nan

            mean = series.mean()
            dev = series - mean
            cum_dev = dev.cumsum()
            R = cum_dev.max() - cum_dev.min()
            S = series.std()

            if S == 0:
                return np.nan
            return np.log(R / S)

        dfs = []
        for x in self.days_range:
            s = self.df[ccy].rolling(window=x).apply(hurst_rs, raw=False) / np.log(x)
            s.name = f"{ccy}_feathurst_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_range(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            max = self.df[ccy].rolling(window=x).max()
            min = self.df[ccy].rolling(window=x).min()
            s = np.log(max / min)
            s.name = f"{ccy}_featrange_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)


    @my_cache
    def _feature_meanret_oneday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        logret = np.log(self.df[ccy] / self.df[ccy].shift(1))
        for x in self.days_range:
            s = logret.rolling(window=x).mean()
            s.name = f"{ccy}_featmean1_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_meanret_xday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        
        for x in self.days_range:
            logret = np.log(self.df[ccy] / self.df[ccy].shift(x-1))
            s = logret.rolling(window=x).mean()
            s.name = f"{ccy}_featmeanx_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    @my_cache
    def _feature_std_oneday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        logret = np.log(self.df[ccy] / self.df[ccy].shift(1))
        for x in self.days_range:
            s = logret.rolling(window=x).std()
            s.name = f"{ccy}_featstd1_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    
    @my_cache
    def _feature_std_xday(self, ccy: str, force_reset=False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            logret = np.log(self.df[ccy] / self.df[ccy].shift(x-1))
            s = logret.rolling(window=x).std()
            s.name = f"{ccy}_featstdx_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @my_cache
    def _feature_rsi(self, ccy: str, force_reset = False) -> pd.DataFrame:
        dfs = []
        for x in self.days_range:
            s = self.compute_rsi(self.df[ccy], x)
            s.name = f"{ccy}_featrsi_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)
    
    @my_cache
    def all_csi(self, window):
        return self.c.compute_csi(window=window)
    
    @my_cache
    def _feature_csi(self, pair: str) -> pd.DataFrame:
        
        dfs = []
        for x in self.days_range:
            
            # ytee 27 June 2025: we use x-1 instead of x to standardize the behavior of features.
            # CSI is computed as price_t / price_t.shift(x), where price_t.shift(x) = price at (t - x),
            # effectively covering the period [t - x, t].
            # This aligns with a standard rolling window of size w, where w = x + 1,
            # which also spans [t - w + 1, t] — for example, a 2-day rolling window covers [t - 1, t].
            csi = self.all_csi(x-1)
            s = csi[pair]
            s.name = f"{pair}_featcsi_{x}d"
            dfs.append(s)
        return pd.concat(dfs, axis=1)


if __name__ == '__main__':
    a  = CORR2()
    a.get_future()
    a.get_feature()