
from sklearn.preprocessing import StandardScaler
from src.pattern.functions import cumsum, get_dendrogram, apply_methodologies, parse_list, METHODS
from src.csv.reader import reader
import datetime
import pandas as pd


class ANALYSIS:
    def __init__(self, ccy):
        self.scaler = StandardScaler()
        self.ccy = ccy
        self.reader = reader(ccy)
        self.events = self.reader.query_ff_current()
        self.excl1 = self.reader.event_metadata(ccy[:3])
        self.excl2 = self.reader.event_metadata(ccy[3:])

        self.ca = self.cumsum_analysis()
        

    def cumsum_analysis(self, threshold=0.05, raw=False):
        r = self.reader
        out =cumsum(r.fx.Close[r.fx.index > datetime.datetime(2010,1,1)], threshold)

        ret = {}
        for (dt1, _) , (dt2, changes) in zip(out, out[1:]):
            k = (dt1, dt2)
            ret[k] = {(None, None, 'changes') : {'changes' : [int(changes)]}}

        for event, excl in zip(self.events, [self.excl1, self.excl2]):
            for r in event.iterrows():
                row = r[1]
                if row.event in excl[excl['min'] > datetime.datetime(2010,1,1)].index:
                    continue
                for k, v in ret.items():
                    dt1, dt2 = k
                    k2 = (row.currency, row.eclass,  row.event)
                    if not k2 in v:
                        v[k2] = {}
                        v[k2]['data'] = []

                    if row.datetime < dt1:
                        v[k2]['before'] = [row.actual]
                        continue

                    if row.datetime > dt2:
                        if not 'after' in v[k2]:
                            v[k2]['after'] = [row.actual]
                        continue
                    
                    v[k2]['data'].append(row.actual)

        frames = []
        
        for idx, data in ret.items():
            data2 = {k: [y for x in v.values() for y in x] for k, v in data.items()}
            if not raw:
                data3 = {}
                for k, v in data2.items():
                    if 'changes' in k:
                        k2= (None, None, None, 'changes')
                        data3[k2] = v[0]
                        continue

                    method_ret = apply_methodologies(parse_list(v))
                    for method, mret in zip(METHODS, method_ret):
                        k2 = k + (method, )
                        data3[k2] = mret
            else:
                data3 = data2
            idx2 = '-'.join([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in idx])
            df = pd.Series(data3, name=idx2)
            frames.append(df)
        
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
    a.plot_chart_ccy_changes()