
from scipy.signal import find_peaks
import pandas as pd
import datetime
import re
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from src.csv.reader import reader

METHODS =['sum', 'mean', 'slope', 'std_dev']
def optimum_size(arr, interval = 0.0005):
    
    mem = []
    sizes = []
    for i in range(1, 20):
        size  = interval*i
        count = find_peaks(arr, prominence=size*i)[0].size
        mem.append(count)
        sizes.append(size)

    ret = [(cnt1 - cnt2)/ cnt1 for cnt1, cnt2 in zip(mem, mem[1:])]
    print(list(zip(ret, sizes)))
    for idx, (r1, r2, r3) in enumerate(zip(ret, ret[1:], ret[2:])):
        if r1 > r2 and r3 > r2:
            return sizes[idx+1]


def cumsum(arr: pd.Series, threshold=0.05):
    """
    create new group at threshold
    """
    df = arr.pct_change()
    change = None
    out = []

    for idx, (dt, price) in enumerate(df.items()):
        if pd.isnull(price):
            out.append((dt, None))
            continue

        if change is None:
            change = (1+price)
        else:
            change*= (1+price)

        if abs(change -1) >= threshold:
            out.append((dt, (change -1)>0 ))
            change = None
            

    return out

def effective_cumsum(arr: pd.Series, threshold=0.05):
    from_cumsum = cumsum(arr, threshold)

    # TODO

    
def cumsum_analysis(currency, threshold=0.05, raw=False):
    r = reader(currency)
    out =cumsum(r.fx.Close[r.fx.index > datetime.datetime(2010,1,1)], threshold)
    events = r.query_ff_current()
    excl_all = [r.event_metadata(currency[:3]),r.event_metadata(currency[3:])]

    ret = {}
    for (dt1, _) , (dt2, changes) in zip(out, out[1:]):
        k = (dt1, dt2)
        ret[k] = {(None, None, 'changes') : {'changes' : [int(changes)]}}

    for event, excl in zip(events, excl_all):
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


def parse_list(cell):
    try:
        ret = []
        for x in cell:
            ret.append(parse_format(x))
        return ret
    except Exception as e :
        return str(e)

def parse_format(x):
    x =str(x)
    if  '|' in x:
        x = x.split('|')[0]

    mult = 1
    if 'k' in x.lower():
        mult = 1e3
    elif 'b' in x.lower():
        mult = 1e9
    elif 'm' in x.lower():
        mult = 1e6

    try:
        x = re.findall('[-\.\d]+',x)[0]
    except Exception as e:
        print("%s has error" % x)
        raise e
    return float(x)*mult

def apply_methodologies(data_in: list):
    """
    Choosing the Methodology
    1. Use cumulative metrics (e.g., product, returns) if compounding or cumulative effects matter.
    2. Use averages (mean, median) for central tendencies.
    3. Explore trends (slope, rolling stats) for directional or time-series data.
    4. Incorporate variability (standard deviation, entropy) if spread matters.
    """
    _sum = sum(data_in)
    _mean = np.mean(data_in)
    slope = np.polyfit(range(len(data_in)), data_in, 1)[0]
    std_dev = np.std(data_in)

    return _sum, _mean, slope, std_dev

def get_dendrogram(df, distance_threshold=5, method=None, fig=False):

    """
    method : sum, mean, slope, std_dev (see METHODS)
    input from cumsum analysis?
    """
    scaler = StandardScaler()
    if method:
        assert method in METHODS , f"method should exist in {METHODS}"
        X = df[df.index.get_level_values(-1) == method]
    else:
        X = df
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    hier_comp = linkage(X_scaled, method='complete', metric='euclidean')
    if not fig:
        clusters = fcluster(hier_comp, t=distance_threshold, criterion='distance')
        X['cluster'] = clusters
        return X['cluster']
    else:
        plt.figure(figsize=(100, 80))
        plt.title('Dendrogram of FX Indicators', fontsize=14)
        plt.xlabel('Distance', fontsize=20)
        plt.ylabel('Indicator', fontsize=20)
        dendrogram(
            hier_comp,
            orientation='right',
            #     leaf_rotation=90.,
            leaf_font_size=20,
            labels=X.index.values,
            color_threshold=3
        )
        fig = plt.gcf()
        plt.close(fig)  # Prevent immediate display of the plot
        return fig

def get_corr(ccy='EUR'):
    r = reader()
    df = r.ff
    excl = r.event_metadata(ccy)
    filter = excl[excl['min'] > datetime.datetime(2010,1,1)].index

    date_range_first_day = pd.date_range(start="2009-01-01", end="2023-12-31", freq="D")
    df_full = pd.DataFrame({"datetime": date_range_first_day})

    df_ccy = df[(df.currency==ccy) & ~(df.event.isin(filter))][['event', 'datetime', 'actual']]
    df_ccy['datetime'] = df_ccy['datetime'].apply(lambda x :x.replace(hour=0, minute=0, second=0))
    df_ccy['actual'] = df_ccy['actual'].apply(parse_format)

    df_pivot= df_ccy.pivot(columns='event', values='actual')
    df_pivot.index=  df_ccy['datetime']
    df_pivot_no_na = df_pivot.groupby('datetime').mean().reset_index()

    ret = df_full.merge(df_pivot_no_na,how='left', on='datetime').fillna(method='ffill')
    ret=  ret[ret.datetime > datetime.datetime(2010,1,1)].drop('datetime', axis=1)
    return ret.corr()


if __name__ == '__main__':
    from src.pattern.functions import cumsum_analysis
    df = cumsum_analysis('EURUSD')
    df.to_csv('review.csv')