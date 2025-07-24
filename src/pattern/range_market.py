import numpy as np
import pandas as pd
from src.csv.cache import CACHE2
from src.csv.reader  import reader2

import mplfinance as mpf

r = reader2()

my_cache =  CACHE2('range_market.cache')

def fill_between_trues(mask: np.ndarray) -> np.ndarray:
    """
    Fill all False values between the first and last True as True.
    """
    if not np.any(mask):
        return mask.copy()

    first = np.argmax(mask)
    last = len(mask) - 1 - np.argmax(mask[::-1])

    filled = mask.copy()
    filled[first:last + 1] = True
    return filled

def compute_emptiness_from_barwise_overlap(lows, highs, boundary_min, boundary_max):
    """
    Compute per-bar overlap with boundary band, normalize by bar size,
    then return 1 - mean(coverage).
    """
    if boundary_min >= boundary_max:
        return 1.0
    # first identify for area under prices to support (S_R_S)
    support_hits = (lows <= boundary_min) & (highs >= boundary_min)
    resistance_hits = (lows <= boundary_max) & (highs >= boundary_max)

    count =0
    is_support= False
    is_resistance = False
    for sidx, ridx in zip(support_hits, resistance_hits):
        if sidx and ridx:
            count+=2
        
        if not is_resistance and ridx:
            is_resistance =True
            is_support =False
            count +=1

        if not is_support and sidx:
            is_resistance =False
            is_support =True
            count +=1

        if count >=4:
            break

    if count < 4:
        return 1.0
    

    support2 = fill_between_trues(support_hits)
    resistance2 = fill_between_trues(resistance_hits)
    boundary_range = boundary_max - boundary_min

    # lows_clipped = np.clip(lows, boundary_min, boundary_max)
    # highs_clipped = np.clip(highs, boundary_min, boundary_max)

    # coverages = []
    # for lo, hi, sidx, ridx in zip(lows_clipped, highs_clipped, support2, resistance2):

    #     if sidx and ridx:
    #         coverages.append(1.0)
        
    #     elif ridx:
    #         overlap = max(0, boundary_max - lo)
    #         coverage = overlap / boundary_range
    #         coverages.append(coverage)

    #     elif sidx:
    #         overlap = max(0, hi - boundary_min)
    #         coverage = overlap / boundary_range
    #         coverages.append(coverage)
    #     else:
    #         coverages.append(0)

    both = support2 & resistance2
    only_r = resistance2 & ~support2
    only_s = support2 & ~resistance2
    coverages = np.zeros_like(lows, dtype=np.float64)
    coverages[both] = 1.0
    coverages[only_r] = np.maximum(0, boundary_max - lows[only_r]) / boundary_range
    coverages[only_s] = np.maximum(0, highs[only_s] - boundary_min) / boundary_range

    mean_coverage = np.mean(coverages)
    emptiness = 1.0 - mean_coverage
    return emptiness

def widest_band_with_target_emptiness(df_slice, threshold=0.15, tolerance=0.005):
    """
    Find widest price band [a, b] such that |emptiness - threshold| < tolerance.
    Returns: (emptiness, (a, b))
    """
    lows = df_slice['Low'].values
    highs = df_slice['High'].values
    prices = np.unique(np.concatenate([lows, highs]))
    
    best_band = (np.nan, np.nan)
    best_width = -np.inf
    best_emptiness = None

    ret = []

    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            a = prices[i]
            b = prices[j]
            emptiness = compute_emptiness_from_barwise_overlap(lows, highs, a, b)
            if emptiness == 1:
                continue
            
            if abs(emptiness - threshold) <= tolerance:
                width = b - a
                ret.append((emptiness, a, b, width))
                if width > best_width:
                    best_width = width
                    best_band = (a, b)
                    best_emptiness = emptiness

    return best_emptiness, best_band, ret
