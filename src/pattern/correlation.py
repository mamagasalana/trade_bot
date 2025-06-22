
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
    def __init__(self, ccys=['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS']):
        self.days_range  = range(10, 1000, 10)
        self.windows = range(10, 1000, 10)
        self.CURRENCIES = ccys
        self.c = CCY_STR(ccys)
        self.cache = CACHE2('corr2.cache')

        self.df = self.get_all_pairs()
        self.all_pairs = self.df.columns

        self.df2 = self.get_future()

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
        return pd.concat(dfs, axis=1)
    

if __name__ == '__main__':
    a  = CORR2()
    a.get_future()