
from scipy.signal import find_peaks
import pandas as pd
import datetime

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
            out.append(dt)
            continue

        if change is None:
            change = (1+price)
        else:
            change*= (1+price)

        if abs(change -1) >= threshold:
            change = None
            out.append(dt)

    return out

def effective_cumsum(arr: pd.Series, threshold=0.05):
    from_cumsum = cumsum(arr, threshold)
    
    # TODO

    
def cumsum_analysis(currency, threshold=0.05):
    from src.csv.reader import reader

    r = reader(currency)
    out =cumsum(r.fx.Close[r.fx.index > datetime.datetime(2010,1,1)], threshold)
    events = r.query_ff()
    excl_all = r.event_metadata()

    ret = {}
    for dt1 , dt2 in zip(out, out[1:]):
        k = (dt1, dt2)
        ret[k] = {}

    for event, excl in zip(events, excl_all):
        for r in event.iterrows():
            row = r[1]
            if row.event in excl[excl['min'] > datetime.datetime(2010,1,1)].index:
                continue
            for k, v in ret.items():
                dt1, dt2 = k
                k2 = (row.currency, row.event)
                if not k2 in v:
                    v[k2] = {}
                    v[k2]['data'] = []

                if row.datetime < dt1:
                    v[k2]['before'] = row.actual
                    continue

                if row.datetime > dt2:
                    if not 'after' in v[k2]:
                        v[k2]['after'] = row.actual
                    continue
                
                v[k2]['data'].append(row.actual)

    frames = []
    for idx, data in ret.items():
        
        series = pd.Series(data, name=idx)
        df = series.apply(pd.Series)[['before', 'data', 'after']].T
        idx2 = '-'.join([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in idx])
        df.index =  pd.MultiIndex.from_product([[idx2], df.index]) 
        frames.append(df)
    
    return pd.concat(frames)