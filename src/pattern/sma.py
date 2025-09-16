from src.csv.reader  import reader2
from src.csv.cache import CACHE3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import json 
import pandas as pd
import pickle
import numpy as np

my_cache =  CACHE3('sma.cache')
r = reader2()

class sma:
    def __init__(self, max_workers=None):
        self.dfs = {}
        if max_workers is None:
            self.max_workers = os.cpu_count()-3
        else:
            self.max_workers = min(os.cpu_count()-3, max_workers)

    def get_file(self, ccy, tf):
        if not (ccy, tf) in self.dfs:
            self.dfs[(ccy, tf)]  = r.get_file_tf(ccy, tf)
        return self.dfs[(ccy, tf)]

    def get_raw(self, ccy, tf, fast, slow, scale):
        df = self.get_file(ccy, tf).copy()
        df['profit'] =  df['Close'].diff().shift(-1)
        df['ema_f'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['ema_s'] = df['Close'].ewm(span=slow, adjust=False).mean()
        # df['ema_f'] = df.Close.rolling(window=fast).mean()
        # df['ema_s'] = df.Close.rolling(window=slow).mean()
        df['sig'] = df.ema_f - df.ema_s

        df['dead'] = df['Close'].pct_change().rolling(slow).std().fillna(0) * scale  # volatility-scaled band
        df['direction'] = 0
        df.loc[df.sig > df.dead, 'direction'] = 1
        df.loc[df.sig < -df.dead, 'direction'] = -1
        return df
    

    def score(self, g):
        p = g['profit']
        wins = p[p > 0]
        losses = p[p <= 0]
        wr = wins.size / p.size if p.size else np.nan
        avg_win = wins.mean() if wins.size else 0.0
        avg_loss = -losses.mean() if losses.size else 0.0  # positive magnitude
        sum_w = wins.sum()
        sum_l = -losses.sum()  # positive magnitude
        pf = (sum_w / sum_l) if sum_l > 0 else np.inf
        payoff = (avg_win / avg_loss) if avg_loss > 0 else np.inf
        return pd.Series({
            'n': p.size,
            'win_rate': wr,
            'count_win' : wins.size,
            'count_loss' : losses.size,
            'sum_win' : sum_w,
            'sum_loss' : sum_l, 
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'payoff_ratio': payoff,
            'profit_factor': pf,
            'expectancy_per_trade': p.mean(),
            'median': p.median(),
            'iqr': p.quantile(0.75) - p.quantile(0.25),
            'std': p.std(ddof=1)
        })

    @my_cache
    def get_profit(self, ccy, tf, fast, slow, scale=0.2, force_reset=False):
        df =  self.get_raw(ccy, tf, fast, slow, scale)
        # profit1 = df.groupby(["direction"]).profit.sum().to_dict()
        # profit1_count = df.groupby(["direction"]).profit.count().to_dict()

        # group by consecutive direction streak
        df["groupref"] = (df['direction'] != df['direction'].shift()).cumsum()
        df2 = df.groupby(["groupref", "direction"], as_index=False).profit.sum()
        ret = df2.groupby('direction', group_keys=False, dropna=False).apply(self.score, include_groups=False)
        ret['meta_ccy'] = ccy
        ret['meta_tf'] = tf
        ret['meta_fast'] = fast
        ret['meta_slow'] = slow
        ret['meta_scale'] = scale
        return ret.to_dict()
    
    def get_profit_by_batch(self, ccy, tf, result=False, force_reset=False):

        combinations = [
            (idx_start, idx_end, threshold / 10)
            for idx_start in range(10, 500, 10)
            for idx_end in range(idx_start+10, idx_start + 500, 10)
            for threshold in range(0, 11)
        ]

        def process_combination(combo):
            idx_start, idx_end, threshold = combo
            profits = self.get_profit(
                ccy, tf, idx_start, idx_end, threshold, force_reset=force_reset
            )
            if result:
                return [ccy, tf, idx_start, idx_end, threshold, profits]

        ret = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for out in tqdm(executor.map(process_combination, combinations), total=len(combinations)):
                if out:
                    ret.append(out)
        return ret


    def summary(self, filters={'func': 'sma.get_profit'}):
        q = my_cache.qry(filters=filters)
        metaraw= [x.meta for x in q]
        meta = pd.json_normalize(pd.Series(metaraw).map(json.loads))

        dataraw= [x.v for x in q]
        data2 = pd.Series(dataraw).map(pickle.loads)
        data = pd.json_normalize(data2, sep='_')
        return pd.concat([meta, data], axis=1)

if __name__ == '__main__':
    a  = sma()
    df = a.get_profit('AUDUSD', '1h', 20, 50)
