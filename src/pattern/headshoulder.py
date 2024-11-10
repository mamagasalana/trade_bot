
from scipy.signal import find_peaks


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





def head_and_shoulders(data, window=5, threshold=1.02):
    """
    Identify head and shoulders pattern in price data.
    
    Parameters:
        data (pd.Series): A series of price data.
        window (int): Rolling window size to find local maxima.
        threshold (float): Ratio to check if the head is significantly higher than shoulders.
        
    Returns:
        head_and_shoulders (list of tuples): Indices of the shoulders and head.
    """
    # Find peaks in the data
    peaks, _ = find_peaks(data)
    head_and_shoulders = []

    for i in range(1, len(peaks) - 1):
        left_shoulder = peaks[i - 1]
        head = peaks[i]
        right_shoulder = peaks[i + 1]
        
        # Condition for head and shoulders: head is higher than shoulders by the threshold
        if data[head] > data[left_shoulder] * threshold and data[head] > data[right_shoulder] * threshold:
            head_and_shoulders.append((left_shoulder, head, right_shoulder))
            
    return head_and_shoulders
