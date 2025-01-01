
from sklearn.preprocessing import StandardScaler
from src.pattern.functions import cumsum, get_dendrogram, apply_methodologies, METHODS
from src.csv.reader import reader
import datetime
import pandas as pd
import itertools

class ANALYSIS:
    def __init__(self, ccy):
        self.scaler = StandardScaler()
        self.ccy = ccy
        self.reader = reader(ccy)
        self.events = self.reader.query_ff_current()
        self.excl1 = self.reader.event_metadata(ccy[:3])
        self.excl2 = self.reader.event_metadata(ccy[3:])

        self.ca = self.optimized_cusum_analysis()

    def optimized_cusum_analysis(self, threshold=0.05, raw=False,mode=2, shift=0):
        """
        :param mode: 
        -1 = before, between
        0 = between
        1 = after, between
        2 = before, between, after
        
        :return: pandas DataFrame
        """
        r = self.reader
        close = r.fx.Close.copy()
        close2 = close.shift(shift).dropna()
        out =cumsum(close2[close2.index > datetime.datetime(2010,1,1)], threshold)

        if raw:
            changes_idx = (None, None, 'changes')
        else:
            changes_idx = (None, None, None, 'changes')

        ret = {}
        for (dt1, _) , (dt2, changes) in zip(out, out[1:]):
            k = (dt1, dt2)

            ret[k] = [pd.Series([int(changes)], index=[changes_idx])]

        
        for event_original, excl in zip(self.events, [self.excl1, self.excl2]):

            # these event only have data after Jan 2010 (could be after 2018)
            
            excl_events= excl[excl['min'] > datetime.datetime(2010,1,1)].index
            event = event_original[~event_original.event.isin(excl_events)].copy()
            event['event2'] = event[['currency', 'eclass', 'event']].apply(tuple, axis=1)
            for k, v in ret.items():
                dt1, dt2 = k
                before = None; after = None

                if mode in [-1, 2]:
                    before = event[event.datetime < dt1].groupby('event2').apply(lambda group: [group.loc[group.datetime.idxmax(), 'actual']], include_groups=False)
                
                if mode in [1, 2]: 
                    after = event[event.datetime > dt2].groupby('event2').apply(lambda group: [group.loc[group.datetime.idxmin(), 'actual']], include_groups=False)

                between = event[(event.datetime>=dt1)&(event.datetime<=dt2)].groupby('event2').actual.apply(list, include_groups=False)

                d = [x for x in [before, between, after] if x is not None]
                d_merged =pd.concat(d, axis=1).apply(lambda x: [float(z) for y in x if isinstance(y, list) for z in y ], axis=1)
                if raw:
                    v.append(d_merged)
                else:
                    idx_raw = list(itertools.product(d_merged.index, METHODS))
                    new_index = [tup + (num,) for tup, num in idx_raw]
                    d_merged_method = d_merged.apply(apply_methodologies).explode()
                    d_merged_method.index = new_index
                    v.append(d_merged_method)

        frames = []
        for k, v in ret.items():
            fname = '-'.join([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in k])
            frame = pd.concat(v)
            frame.name = fname
            frames.append(frame)
            
        return pd.concat(frames,axis=1)

    def plot_chart_in_the_same_cluster(self, ccy, thres=6, scaled=False):
        df = self.reader.get_corr(ccy)
        dp = self.reader.get_corr(ccy, format=1)
        df2 = get_dendrogram(df, distance_threshold=thres)
        scaler = StandardScaler()

        for i in range(1, df2.max()):
            selected_ind = df2[df2==i].index
            if scaled:
                X = dp[selected_ind]
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                X_scaled.plot(kind='line')
            else:
                dp[selected_ind].plot(kind='line')
                
    def plot_chart_ccy_changes(self, ccy2, thres=20):
        dft = self.ca.T
        dendrogram = get_dendrogram(dft.corr(), distance_threshold=thres)
        
        changes_index= dendrogram.iloc[0]
        
        dp = self.reader.get_corr(ccy2, format=0)

        fx  = self.reader.fx.copy()
        fx['datetime'] = fx.index
        fx['datetime'] = fx['datetime'].apply(lambda x :x.replace(hour=0, minute=0, second=0))
        fx2 = fx.groupby('datetime').Close.last()
        fx3 = dp.merge(fx2,how='left', on='datetime').fillna(method='ffill')
        selected_ind = dendrogram[dendrogram==changes_index].index
        s2 = [x[2] if pd.notnull(x[2]) else 'Close'  for x in selected_ind]

        X = fx3[s2]
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X_scaled.plot(kind='line')

if __name__ == '__main__':
    a = ANALYSIS('EURUSD')
    a.plot_chart_ccy_changes("USD")