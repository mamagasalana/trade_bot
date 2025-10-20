import pandas as pd
import numpy as np
from src.csv.reader  import reader2
from src.csv.cache import CACHE3
import re
import glob
from src.pattern.functions import get_barrier, get_barrier_by_batch
import itertools
from tqdm import tqdm

my_cache =  CACHE3('momentum.cache')
r = reader2()
  
class MOMENTUM:
    def __init__(self):
        self.dfs = {}
        self.raw=  {}
        self.barrier_preprocess = {}

    def get_file(self, ccy, tf):
        if not (ccy, tf) in self.dfs:
            self.dfs[(ccy, tf)]  = r.get_file_tf(ccy, tf)
        return self.dfs[(ccy, tf)]

    def get_all_file(self, tf):
        CURRENCIES = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR',]
        codes = '.*(?:%s).*' % "|".join(CURRENCIES)
        for f in sorted(glob.glob('files/ccy/*.csv')):
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
            raw['trigger'] =np.where(pd.notnull(raw["gap_max_pct"]), 1, 
                    np.where(pd.notnull(raw["gap_min_pct"]), -1, 0)
                    )
        
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
    
    def _get_barrier_preprocess(self, tf):
        if not tf in self.barrier_preprocess:
            raw = self.compile_raw(tf)
            summaryw = self.summary_by_window(tf)
            raw2 = raw.reset_index(drop=True)
            raw2=  raw2.join(summaryw['gap_q25'], on=['ccy','window'])
            raw2=  raw2.join(summaryw['gap_q50'], on=['ccy','window'])
            raw2=  raw2.join(summaryw['gap_q75'], on=['ccy','window'])
            self.barrier_preprocess[tf] =  raw2
        return self.barrier_preprocess[tf]
    
    @my_cache
    def _get_barrier(self, tf='4h', thresh_col='gap_q75', barrier_window=5, ratio=0.5, overlap=False, debug=True, force_reset=False):
        raw2 = self._get_barrier_preprocess(tf)
        raw2['thresh'] = raw2[thresh_col]
        ret = get_barrier(raw2, ['ccy', 'window'], k2=barrier_window, ratio=ratio, overlap=overlap, debug=debug)
        return ret
    
    @my_cache
    def _get_barrier2(self, tf='4h', thresh_col='gap_q75', barrier_window=5, ratio=0.5, overlap=False, debug=True, force_reset=False):
        print('not running from cache', tf, thresh_col, barrier_window, ratio, overlap, debug, force_reset)
        raw2 = self._get_barrier_preprocess(tf)
        raw2['thresh'] = raw2[thresh_col]
        ret = get_barrier_by_batch(raw2, ['ccy', 'window'], k2=barrier_window, ratio=ratio, overlap=overlap, debug=debug)
        return ret
    

    def main(self, tf, force_reset=False):
        thresh_options = ['gap_q25', 'gap_q50', 'gap_q75']
        ratio_options = [0.25, 0.5, 0,75, 1.0, 1.25]
        barrier_window_options = list(range(20, 100, 10))
        overlap_options = [0, 1]

        # Create Cartesian product of all parameters
        all_combs = list(itertools.product(
            thresh_options,
            ratio_options,
            barrier_window_options,
            overlap_options,
        ))

        ret = []
        mask = self._get_barrier_preprocess(tf).trigger !=0
        for comb in tqdm(all_combs):
            thresh, ratio, barrier_window, overlap  = comb

            tmp = self.summary_for_barrier_state(tf, comb, self._get_barrier2(tf, thresh, barrier_window, ratio, overlap, debug=False, force_reset=force_reset)[mask])
            ret.append(tmp.reset_index())

        return pd.concat(ret, ignore_index=True)


    def summary_for_barrier_state(self, tf, comb, df):
        thresh, ratio, barrier_window, overlap  = comb
        dfmain = self._get_barrier_preprocess(tf)
        dfret =pd.concat([dfmain[dfmain.trigger!=0],df], axis=1)

        yy_use =dfret[dfret.keep]
        # State indicators
        is_tp  = (yy_use['state'] ==  1)
        is_none= (yy_use['state'] ==  0)
        is_sl  = (yy_use['state'] == -1)

        # Threshold masks (cumulative: gap_pct > qX)
        m25 = yy_use['gap_pct'] > yy_use['gap_q25']
        m50 = yy_use['gap_pct'] > yy_use['gap_q50']
        m75 = yy_use['gap_pct'] > yy_use['gap_q75']

        g = yy_use.groupby(['ccy','window'])
        summary  =g.agg(
            n_total = ('state', 'size'),

            n_gt25  = ('state', lambda s: m25.loc[s.index].sum()),
            n_gt50  = ('state', lambda s: m50.loc[s.index].sum()),
            n_gt75  = ('state', lambda s: m75.loc[s.index].sum()),

            tp_gt25 = ('state', lambda s: (is_tp.loc[s.index][m25.loc[s.index]]).mean()),
            tp_gt50 = ('state', lambda s: (is_tp.loc[s.index][m50.loc[s.index]]).mean()),
            tp_gt75 = ('state', lambda s: (is_tp.loc[s.index][m75.loc[s.index]]).mean()),

            none_gt25 = ('state', lambda s: (is_none.loc[s.index][m25.loc[s.index]]).mean()),
            none_gt50 = ('state', lambda s: (is_none.loc[s.index][m50.loc[s.index]]).mean()),
            none_gt75 = ('state', lambda s: (is_none.loc[s.index][m75.loc[s.index]]).mean()),

            sl_gt25 = ('state', lambda s: (is_sl.loc[s.index][m25.loc[s.index]]).mean()),
            sl_gt50 = ('state', lambda s: (is_sl.loc[s.index][m50.loc[s.index]]).mean()),
            sl_gt75 = ('state', lambda s: (is_sl.loc[s.index][m75.loc[s.index]]).mean()),


            gt25 = ('state', lambda s: (is_tp.loc[s.index][m25.loc[s.index]]).mean() - (is_sl.loc[s.index][m25.loc[s.index]]).mean()),
            gt50 = ('state', lambda s: (is_tp.loc[s.index][m50.loc[s.index]]).mean() - (is_sl.loc[s.index][m50.loc[s.index]]).mean()),
            gt75 = ('state', lambda s: (is_tp.loc[s.index][m75.loc[s.index]]).mean() - (is_sl.loc[s.index][m75.loc[s.index]]).mean()),
            
        )

        summary['thresh'] = thresh
        summary['ratio'] = ratio
        summary['barrier_window'] = barrier_window
        summary['overlap'] = overlap
        return summary