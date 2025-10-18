import pandas as pd
import numpy as np
from src.csv.reader  import reader2
from src.csv.cache import CACHE3
import re
import glob

my_cache =  CACHE3('momentum.cache')
r = reader2()
  
class MOMENTUM:
    def __init__(self):
        self.dfs = {}
        self.raw=  {}


    def get_file(self, ccy, tf):
        if not (ccy, tf) in self.dfs:
            self.dfs[(ccy, tf)]  = r.get_file_tf(ccy, tf)
        return self.dfs[(ccy, tf)]

    def get_all_file(self, tf):
        CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR',]
        codes = '.*(?:%s).*' % "|".join(CURRENCIES)
        for f in glob.glob('files/ccy/*.csv'):
            if re.match(codes, f):
                ccy = re.findall('ccy/(.*).csv', f)[0]
                df = self.get_file(ccy, tf)
                df['ccy'] = ccy

        return pd.concat(self.dfs.values())
    
    def get_raw_vectorized(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        # df must contain columns: ['ccy', 'Close'] and be time-sorted within ccy
        df = df.copy()
        g = df.groupby("ccy")["Close"]
        s = df["Close"].astype(float)

        prev_min = g.shift(1).rolling(k-1, min_periods=k-1).min()
        prev_max = g.shift(1).rolling(k-1, min_periods=k-1).max()

        is_peak   = s > prev_max          # use >= to include ties
        is_trough = s < prev_min          # use <= to include ties

        # absolute gaps
        df["gap_max"] = np.where(is_peak,   s - prev_min, np.nan)
        df["gap_min"] = np.where(is_trough, prev_max - s, np.nan)

        # percent gaps
        df["gap_max_pct"] = np.where(is_peak , (s / prev_min) - 1.0, np.nan)
        df["gap_min_pct"] = np.where(is_trough,      (prev_max / s) - 1.0, np.nan)
        df['window']=  k
        return df.reset_index()
    

    def compile_raw(self, tf):
        if not tf in self.raw:
            dfs = []
            df= self.get_all_file(tf)
            for i in range(20, 500, 10):
                tmp = self.get_raw_vectorized(df, i)
                dfs.append(tmp)

            raw = pd.concat(dfs)
            raw['year'] = raw.Date.dt.year
            raw['gap_pct'] = raw['gap_min_pct'].combine_first(raw['gap_max_pct'])
            self.raw[tf] = raw.copy()
        return self.raw[tf]
    
    def summary_by_window(self, tf):
        raw = self.compile_raw(tf)
        summary = raw.groupby(["ccy", "window"]).agg(
                    gap_min_cnt=("gap_min_pct", "count"),
                    gap_max_cnt=("gap_max_pct", "count"),
                    gap_min_mean=("gap_min_pct", "mean"),
                    gap_max_mean=("gap_max_pct", "mean"),
                    gap_min_q25=("gap_min_pct", lambda s: s.quantile(0.25)),
                    gap_min_q50=("gap_min_pct", "median"),
                    gap_min_q75=("gap_min_pct", lambda s: s.quantile(0.75)),
                    gap_max_q25=("gap_max_pct", lambda s: s.quantile(0.25)),
                    gap_max_q50=("gap_max_pct", "median"),
                    gap_max_q75=("gap_max_pct", lambda s: s.quantile(0.75)),

                    gap_cnt=("gap_pct", "count"),
                    gap_mean=("gap_pct", "mean"),
                    gap_q25=("gap_pct", lambda s: s.quantile(0.25)),
                    gap_q50=("gap_pct", "median"),
                    gap_q75=("gap_pct", lambda s: s.quantile(0.75)),

                ).sort_index()
        return summary
    
    def summary_by_window_year(self, tf):
        raw = self.compile_raw(tf)
        summary = raw.groupby(["ccy", "window", 'year']).agg(
                    gap_min_cnt=("gap_min_pct", "count"),
                    gap_max_cnt=("gap_max_pct", "count"),
                    gap_min_mean=("gap_min_pct", "mean"),
                    gap_max_mean=("gap_max_pct", "mean"),
                    gap_min_q25=("gap_min_pct", lambda s: s.quantile(0.25)),
                    gap_min_q50=("gap_min_pct", "median"),
                    gap_min_q75=("gap_min_pct", lambda s: s.quantile(0.75)),
                    gap_max_q25=("gap_max_pct", lambda s: s.quantile(0.25)),
                    gap_max_q50=("gap_max_pct", "median"),
                    gap_max_q75=("gap_max_pct", lambda s: s.quantile(0.75)),

                    gap_cnt=("gap_pct", "count"),
                    gap_mean=("gap_pct", "mean"),
                    gap_q25=("gap_pct", lambda s: s.quantile(0.25)),
                    gap_q50=("gap_pct", "median"),
                    gap_q75=("gap_pct", lambda s: s.quantile(0.75)),

                ).sort_index()
        return summary