from src.csv.reader  import reader2
from src.csv.cache import CACHE3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import json 
import pandas as pd
import pickle

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
        df = self.get_file(ccy, tf)
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
    
    @my_cache
    def get_profit(self, ccy, tf, fast, slow, scale=0.2, force_reset=False):
        df =  self.get_raw(ccy, tf, fast, slow, scale)
        # profit1 = df.groupby(["direction"]).profit.sum().to_dict()
        # profit1_count = df.groupby(["direction"]).profit.count().to_dict()

        # group by consecutive direction streak
        df["groupref"] = (df['direction'] != df['direction'].shift()).cumsum()
        df2 = df.groupby(["groupref", "direction"], as_index=False).profit.sum()
                
        df2_count = pd.crosstab(
                index=df2['direction'],
                columns=df2['profit'].gt(0),   # True = profit>0
                values=df2['profit'],
                aggfunc='count'
            ).rename(columns={True:'count>0', False:'count<0'}).to_dict()

        df2_sum= pd.crosstab(
                index=df2['direction'],
                columns=df2['profit'].gt(0),   # True = profit>0
                values=df2['profit'],
                aggfunc='sum'
            ).rename(columns={True:'sum>0', False:'sum<0'}).to_dict()
               
        df2_count.update(df2_sum)
        return df2_count
    
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
