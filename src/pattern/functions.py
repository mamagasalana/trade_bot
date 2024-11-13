
from scipy.signal import find_peaks
import pandas as pd

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
