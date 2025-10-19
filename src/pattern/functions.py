
from scipy.signal import find_peaks
import pandas as pd
import datetime
import re
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


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
    if not data_in:
        _mean = 0
        std_dev = 0
    else:
        _mean = np.mean(data_in)
        std_dev = np.std(data_in)

    if len(data_in) < 2:
        slope=  0
    else:
        slope = np.polyfit(range(len(data_in)), data_in, 1)[0]
    

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


def build_fut_mat(x: pd.Series, size=0) -> np.ndarray:
    return np.column_stack([x.shift(-h).to_numpy() for h in range(1, size + 1)])

def first_true_idx(mat: pd.DataFrame, size=0) -> pd.DataFrame:
    # argmax gives 0 if all False, so detect rows with any True
    any_true = mat.any(axis=1)
    idx = mat.idxmax(axis=1) + 1       # 1..k2 for first True, 1 if none
    idx[~any_true] = size + 1            # set sentinel for "no hit"
    return idx


def get_barrier(df, group_keys, k2 =5, ratio=0.5, overlap=True, debug=False):

    assert "thresh" in df.columns
    assert "trigger" in df.columns
    
    g =df.groupby(group_keys)["Close"]
    fut_close  = g.transform(lambda col: build_fut_mat(col, k2).tolist())
    fut = pd.DataFrame(fut_close.tolist())
    
    q = df['thresh']
    
    if debug:
        df['bull_tp'] = df["Close"] * (1.0 + q * ratio)
        df['bear_tp'] = df["Close"] * (1.0 - q * ratio)
        df['fut'] = fut_close


        bull_mat = fut >= df.bull_tp.to_numpy()[:, None]     
        bear_mat = fut <= df.bear_tp.to_numpy()[:, None] 

        # First-touch indices (in bars ahead) for each row
        df['bull_idx'] = first_true_idx(bull_mat, k2)
        df['bear_idx'] = first_true_idx(bear_mat, k2)
        df['state'] = np.where(df.trigger== 1, 
                        np.where(df['bull_idx'] < df['bear_idx'],  1, np.where(df['bear_idx'] < df['bull_idx'], -1, 0)),
                        np.where(df.trigger== -1, 
                            np.where(df['bull_idx'] < df['bear_idx'], -1, np.where(df['bear_idx'] < df['bull_idx'], 1, 0)),
                            np.nan
                            ))
        
        has_full = (
            df.groupby(group_keys)["Close"]
                .transform(lambda s_: s_.rolling(k2, min_periods=k2).count().shift(-k2))
                .notna()
        )

        #this remove overlapping trends
        df['prev_trigger_sum'] = df.groupby(group_keys).trigger.transform(lambda s_: abs(s_).astype(int).rolling(k2, min_periods=1).sum().shift(1).fillna(0))
        
        if overlap:
            df['keep'] = (df.trigger!=0) & has_full & (df.prev_trigger_sum == 0)
        else:
            df['keep'] = (df.trigger!=0) & has_full 

        return df
    else:
        bull_tp = df["Close"] * (1.0 + q * ratio)
        bear_tp = df["Close"] * (1.0 - q * ratio)

        bull_mat = fut >= bull_tp.to_numpy()[:, None]     
        bear_mat = fut <= bear_tp.to_numpy()[:, None] 

        # First-touch indices (in bars ahead) for each row
        bull_idx= first_true_idx(bull_mat, k2)
        bear_idx = first_true_idx(bear_mat, k2)
        state = np.where(df.trigger== 1, 
                        np.where(bull_idx< bear_idx,  1, np.where(bear_idx < bull_idx, -1, 0)),
                        np.where(df.trigger== -1, 
                            np.where(bull_idx < bear_idx, -1, np.where(bear_idx < bull_idx, 1, 0)),
                            np.nan
                            ))
        
        has_full = (
            df.groupby(group_keys)["Close"]
                .transform(lambda s_: s_.rolling(k2, min_periods=k2).count().shift(-k2))
                .notna()
        )

        #this remove overlapping trends
        prev_trigger_sum = df.groupby(group_keys).trigger.transform(lambda s_: abs(s_).astype(int).rolling(k2, min_periods=1).sum().shift(1).fillna(0))
        
        if overlap:
            keep = (df.trigger!=0) & has_full & (prev_trigger_sum == 0)
        else:
            keep = (df.trigger!=0) & has_full 

        ret =  pd.concat([pd.Series(state), keep], axis=1)
        ret.columns = ['state', 'keep']
        return ret
        

 
def get_barrier_by_batch(dfraw, group_keys, k2 =5, ratio=0.5, overlap=True, debug=False):

    assert "thresh" in dfraw.columns
    assert "trigger" in dfraw.columns
    
    
    g =dfraw.groupby(group_keys)
    out= []
    for _ , df in g:
        q = df['thresh']
        fut_close  =build_fut_mat(df.Close, k2)
        fut = pd.DataFrame(fut_close.tolist())
    
    
        bull_tp = df["Close"] * (1.0 + q * ratio)
        bear_tp = df["Close"] * (1.0 - q * ratio)

        bull_mat = fut >= bull_tp.to_numpy()[:, None]     
        bear_mat = fut <= bear_tp.to_numpy()[:, None] 

        # First-touch indices (in bars ahead) for each row
        bull_idx= first_true_idx(bull_mat, k2)
        bear_idx = first_true_idx(bear_mat, k2)
        state = np.where(df.trigger== 1, 
                        np.where(bull_idx< bear_idx,  1, np.where(bear_idx < bull_idx, -1, 0)),
                        np.where(df.trigger== -1, 
                            np.where(bull_idx < bear_idx, -1, np.where(bear_idx < bull_idx, 1, 0)),
                            np.nan
                            ))
        
        has_full = (
            df.groupby(group_keys)["Close"]
                .transform(lambda s_: s_.rolling(k2, min_periods=k2).count().shift(-k2))
                .notna()
        )

        #this remove overlapping trends
        prev_trigger_sum = df.groupby(group_keys).trigger.transform(lambda s_: abs(s_).astype(int).rolling(k2, min_periods=1).sum().shift(1).fillna(0))
        
        if overlap:
            keep = (df.trigger!=0) & has_full & (prev_trigger_sum == 0)
        else:
            keep = (df.trigger!=0) & has_full 

        ret = pd.concat([pd.Series(state), keep.reset_index(drop=True)], axis=1)
        ret.index= keep.index
        out.append(ret)
    final = pd.concat(out)
    final.columns = ['state', 'keep']
    return final.sort_index()


if __name__ == '__main__':
    pass