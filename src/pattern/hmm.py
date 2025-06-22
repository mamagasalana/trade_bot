import numpy as np
import pandas as pd
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from src.pattern.ccy_strength import CCY_STR, CCY_STR_DEBUG, SMOOTHING_METHOD
from src.pattern.helper import HELPER
import matplotlib.pyplot as plt  # Make sure this import is at the top
import itertools
import seaborn as sns
from src.csv.cache import CACHE2
from tqdm import tqdm
import logging

logging.basicConfig(
    filename='hmm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # <- Python 3.8+: force reconfiguration
)

class HMM:

    def __init__(self, windows=[300, 400, 500], method=SMOOTHING_METHOD.ROLLING_ZSCORE,
                 ccylist=['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS']):
        self.ccylist= ccylist
        self.windows= windows
        self.method = method
        self.cache = CACHE2('hmm_cache.cache')
        self.get_init()

        logging.info("started")

    def re_init(self, windows=None, method=None,ccylist=None):
        if windows is not None:
            self.windows =  windows
        
        if method is not None:
            self.method = method
        
        if ccylist is not None:
            self.ccylist= self.ccylist
        self.get_init()

    def get_init(self):

        param_dict= {
            'ccy': sorted(self.ccylist),
            'window' : self.windows,
            'method' : self.method,
            'func' : 'get_init'
        }
        cache_key = str(param_dict)  
        
        # Example: check if result is in cache
        if cache_key not in self.cache:
            self.c = CCY_STR(self.ccylist)

            data = []
            ranks = []
            spread = []
            for window in self.windows:
                data.append(self.c.smoothing_method(window, method=self.method).add_suffix(f'_{window}'))
                spread.append(self.c.mt5_export(window=window, method=self.method).add_suffix(f'_{window}'))
                ranks.append(data[-1].rank(axis=1, method='min', ascending=False))
            
            data_cache = {}
            data_cache['data'] =  pd.concat(data, axis=1)
            data_cache['ranks'] =  pd.concat(ranks, axis=1)
            data_cache['spreads'] = pd.concat(spread, axis=1)
            self.cache[cache_key] = data_cache

            if not 'prices' in self.cache:
                self.cache['prices'] = self.c.get_all_pairs(comb=False)

        data_cache = self.cache[cache_key]
        self.prices =  self.cache['prices']
        self.data =  data_cache['data']
        self.ranks =  data_cache['ranks']
        self.spreads =  data_cache['spreads']
        
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
            'ccy': sorted(self.ccylist),
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
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            logging.info(f"cachekey data not found: {cache_key}")
    
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
        
        self.cache[cache_key] = df
        return df

    def model(self, df:pd.DataFrame, n_components=3, train_ratio=0.7, window_size=300, verbose=False, use_full=False, visualize=False,
              model_only=False) -> DenseHMM:
        """plot hmm

        Args:
        
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            verbose(bool, optional): used by HMM to display debug message
            visualize(bool, optional): to display chart. Defaults to False.
        Returns:
            DenseHMM: HMM model fit with the given parameters
        """


        param_dict = {
            'ccy': sorted(self.ccylist),
            'n_components': n_components,
            'train_ratio': train_ratio,
            'window_size': window_size,
            'method' : self.method,
        }
        cache_key =  '_'.join(df.columns)  + str(param_dict)

        if model_only:
            if cache_key in self.cache:
                model =  self.cache[cache_key]
                return model

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
        split_idx = int(len(rolling_index) *train_ratio)

        if cache_key in self.cache:
            model =  self.cache[cache_key]
        else:
            logging.info(f"cachekey model not found: {cache_key}")
            model = DenseHMM([Normal() for _ in range(n_components)], max_iter=100, verbose=verbose)
            model.fit([X_rolling[:split_idx]])
            self.cache[cache_key] = model
        
        if use_full: 
            last_probs = model.predict_proba(X_full.reshape(1, X_full.shape[0], X_full.shape[1]))[0][window_size-1:]
        else:
            ret = model.predict_proba(X_rolling)
            last_probs = ret[:, -1, :].numpy() 

        
        df_probs = pd.DataFrame(last_probs, columns=[f"regime_{i}" for i in range(last_probs.shape[1])])
        df_probs.index = pd.to_datetime(rolling_index).strftime('%Y-%m-%d')  # restore original datetime index


        pairs= self.pairs
        if len(pairs[0]) ==3:
            pair2 = pairs[0] + pairs[1]
            price = self.prices[pair2].loc[rolling_index]
        else:
            pair2 = pairs[0]
            price = self.prices[pair2].loc[pd.to_datetime(rolling_index).strftime('%Y-%m-%d')]
        
        if visualize:
            # Add vertical line at the train/test split index
            fig, ax1 = plt.subplots(figsize=(12, 4))

            colors = ['tab:red', 'tab:orange', 'yellow', 'tab:green', 'tab:blue', 'tab:pink']
            df_probs.plot(ax=ax1 , color=colors, linewidth=3)
            ax1.axvline(x=split_idx, color='red', linestyle='--', label='Train/Test Split', linewidth=3)
            ax1.text(split_idx, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05,
                f'{df_probs.index[split_idx]}', 
                color='red', 
                fontsize=10, 
                ha='center', 
                va='bottom')
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

        return model, df_probs
    
    def evaluate_model(self, model: DenseHMM, columns: pd.DataFrame, visualize=False):
        """study the attribute of HMM model to understand the regime

        Args:
            model (DenseHMM): from self.model
            columns (pd.DataFrame): from self.get_data
        """
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

        if visualize:
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
        
        return df_mean

    def get_pnl(self, pairs=['EUR', 'USD'], diffs_shift=[], diffs=[],use_rank=False, use_spread=False, 
                    n_components=3, train_ratio=0.7, window_size=300, test_only=False,
                    crossing_threshold=0.5, pending=False
                    ) -> pd.DataFrame:
        """This generates PnL for selected ccy

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            test_only (bool, optional): To generate PnL for test only. Defaults to False.
            crossing_threshold (float, optional): Entry signal, create entry when prob crosses threshold. Defaults to 0.5.

        Returns:
            pd.DataFrame: pnl
        """
        param_dict = {
            'pairs' : sorted(pairs),
            'ccy': sorted(self.ccylist),
            'windows': sorted(self.windows),
            'diffs_shift': sorted(diffs_shift),
            'diffs': sorted(diffs),
            'use_rank': use_rank,
            'use_spread': use_spread,
            'method' : self.method,
            'func' : 'get_pnl_all',
            'n_components' : n_components,
            'train_ratio' : train_ratio,
            'window_size' : window_size,
        }
        
        # Create a unique cache key based on function parameters
        cache_key = str(param_dict)
        
        if cache_key in self.cache:
            data_cache = self.cache[cache_key] 
            if data_cache:
                if pending:
                    return data_cache['pending']
                else:
                    return data_cache['closed']


        df =self.get_data(sorted(pairs), diffs_shift=diffs_shift, diffs=diffs, use_rank=use_rank, use_spread=use_spread)
        _, probs = self.model(df, n_components=n_components, train_ratio=train_ratio, window_size=window_size)

        if test_only:
            split_idx = int(len(probs.index) *train_ratio)
            probs = probs[:split_idx]

        closed, pending = self.extract_regime_trades(probs, ''.join(sorted(pairs)), threshold=crossing_threshold)

        data_cache = {
            'closed' : closed,
            'pending': pending
        } 
        self.cache[cache_key] =  data_cache

        if pending:
            return data_cache['pending']
        else:
            return data_cache['closed']



    def validate_cache(self, pairs=['EUR', 'USD'], diffs_shift=[], diffs=[],use_rank=False, use_spread=False, 
                    n_components=3, train_ratio=0.7, window_size=300, test_only=False,
                    crossing_threshold=0.5, pending=False
                    ) -> pd.DataFrame:
        """This generates PnL for selected ccy

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            test_only (bool, optional): To generate PnL for test only. Defaults to False.
            crossing_threshold (float, optional): Entry signal, create entry when prob crosses threshold. Defaults to 0.5.

        """
        df =self.get_data(sorted(pairs), diffs_shift=diffs_shift, diffs=diffs, use_rank=use_rank, use_spread=use_spread)
        m = self.model(df, n_components=n_components, train_ratio=train_ratio, window_size=window_size, model_only=True)
        return df, m
        
    def extract_regime_trades(self, df_probs: pd.DataFrame, 
                            pair: str, 
                            threshold: float = 0.5,) -> list:
        """
        Extracts entry/exit points where any regime exceeds threshold.
        
        Args:
            df_probs: DataFrame with shape (T, n_regimes), regime probabilities
            pair: str, name of the pair (column in prices) to extract entry/exit prices from
            threshold: float, probability threshold to trigger regime selection
        Returns:
            List of trade dicts with year, pair, entry/exit info and regime
        """

        closed = []
        pending = []
        in_trade = False
        current_regime = None
        entry_idx = None

        for idx, row in df_probs.iterrows():
            max_prob = row.max()
            regime = row.idxmax()

            if in_trade:
                if (regime != current_regime) or (row[current_regime] < threshold):
                    # Exit due to regime switch or threshold drop
                    closed.append({
                        'year': str(entry_idx)[:4],
                        'ccy': pair,
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': float(self.prices.loc[entry_idx, pair]),
                        'exit_price': float(self.prices.loc[idx, pair]),
                        'regime': current_regime,
                        'return': float(self.prices.loc[idx, pair] / self.prices.loc[entry_idx, pair] - 1) 
                    })
                    in_trade = False
                    current_regime = None
                    entry_idx = None

            if not in_trade and max_prob >= threshold:
                # Entry signal
                in_trade = True
                current_regime = regime
                entry_idx = idx

        # Handle open trade at end
        if in_trade:
            pending.append({
                'year': str(entry_idx)[:4],
                'ccy': pair,
                'entry_idx': entry_idx,
                'exit_idx': None,
                'entry_price': float(self.prices.loc[entry_idx, pair]),
                'exit_price': None,
                'regime': current_regime,
                'return': None
            })

        return closed, pending


    def get_pnl_usd_denominated(self, pair='EUR', diffs_shift=[], diffs=[], 
                    n_components=3, train_ratio=0.7, window_size=300, test_only=False,
                    crossing_threshold=0.5, pending=False
                    ) -> pd.DataFrame:
        """This generates PnL for usd-related pair

        Args:
            pairs (str, optional): list of currency. Defaults to 'EUR'.
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            test_only (bool, optional): To generate PnL for test only. Defaults to False.
            crossing_threshold (float, optional): Entry signal, create entry when prob crosses threshold. Defaults to 0.5.

        Returns:
            pd.DataFrame: pnl
        """
        param_dict = {
            'pairs' : pair,
            'ccy': sorted(self.ccylist),
            'windows': sorted(self.windows),
            'diffs_shift': sorted(diffs_shift),
            'diffs': sorted(diffs),
            'method' : self.method,
            'func' : 'get_pnl_usd_denominated',
            'n_components' : n_components,
            'train_ratio' : train_ratio,
            'window_size' : window_size,
        }
        
        # Create a unique cache key based on function parameters
        cache_key = str(param_dict)
        
        if cache_key in self.cache:
            data_cache = self.cache[cache_key]
            if pending:
                return data_cache['pending']
            else:
                return data_cache['closed']


        closed_all =[]
        pending_all =[]

        for c in self.data.columns:
            if pair in c:
                df =self.get_data([c[:3] , c[3:]], diffs_shift=diffs_shift, diffs=diffs, use_rank=False, use_spread=True)
                _, probs = self.model(df, n_components=n_components, train_ratio=train_ratio, window_size=window_size)

                if test_only:
                    split_idx = int(len(probs.index) *train_ratio)
                    probs = probs[:split_idx]

                if c[:3] == pair:
                    pair2 = pair+'USD'
                else:
                    pair2 = 'USD'+pair

                closed, pending = self.extract_regime_trades(probs, pair2, threshold=crossing_threshold)
                closed_all.extend(closed)
                pending_all.extend(pending)

        data_cache = {
            'closed' : closed_all,
            'pending': pending_all
        } 
        self.cache[cache_key] =  data_cache

        if pending:
            return data_cache['pending']
        else:
            return data_cache['closed']


    def validate_cache_datafile_and_model(self, diffs_shift=[], diffs=[],use_rank=False, use_spread=False, 
                    n_components=3, train_ratio=0.7, window_size=300, test_only=False,
                    crossing_threshold=0.5
                    ) -> pd.DataFrame:
        """This generates PnL for all ccy

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            test_only (bool, optional): To generate PnL for test only. Defaults to False.
            crossing_threshold (float, optional): Entry signal, create entry when prob crosses threshold. Defaults to 0.5.

        """
        out = {}
        if use_spread:
            col = set([x[:6] for x in self.spreads.columns if 'timestamp' not in x])
            for ccy in col:
                df, m  = self.validate_cache([ccy[:3], ccy[3:]], diffs_shift=diffs_shift, diffs=diffs,
                             use_spread=use_spread, n_components=n_components, train_ratio=train_ratio,
                             window_size=window_size, test_only=test_only, crossing_threshold=crossing_threshold)

                
                pnl = self.get_pnl([ccy[:3], ccy[3:]], diffs_shift=diffs_shift, diffs=diffs,
                            use_spread=use_spread, n_components=n_components, train_ratio=train_ratio,
                            window_size=window_size, test_only=test_only, crossing_threshold=crossing_threshold)

                out[tuple([ccy[:3], ccy[3:]])] = {
                    'df' : df,
                    'model': m,
                    'pnl': pnl
                }        
        return out

    def get_pnl_all(self, diffs_shift=[], diffs=[],use_rank=False, use_spread=False, 
                    n_components=3, train_ratio=0.7, window_size=300, test_only=False,
                    crossing_threshold=0.5
                    ) -> pd.DataFrame:
        """This generates PnL for all ccy

        Args:
            pairs (list, optional): list of currency. Defaults to ['EUR', 'USD'].
            diffs_shift (list, optional): list of lags to measure strength momentum. Defaults to [].
            diffs (list, optional): list of size of diff to measure strength momentum. Defaults to [].
            use_rank (bool, optional): when use, add rank to fit HMM model. Defaults to False.
            use_spread (bool, optional): when use, use spread instead of original ccy values. Defaults to False.
            n_components (int, optional): number of HMM states. Defaults to 3.
            train_ratio (float, optional): train ratio to fit HMM model. Defaults to 0.7.
            window_size (int, optional): each batch size to fit HMM model. Defaults to 300.
            test_only (bool, optional): To generate PnL for test only. Defaults to False.
            crossing_threshold (float, optional): Entry signal, create entry when prob crosses threshold. Defaults to 0.5.

        """
        if use_spread:
            col = set([x[:6] for x in self.spreads.columns if 'timestamp' not in x])
            for ccy in col:
                self.get_pnl([ccy[:3], ccy[3:]], diffs_shift=diffs_shift, diffs=diffs,
                             use_spread=use_spread, n_components=n_components, train_ratio=train_ratio,
                             window_size=window_size, test_only=test_only, crossing_threshold=crossing_threshold)

        