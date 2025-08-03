import numpy as np
import pandas as pd
from src.csv.cache import CACHE2
from src.csv.reader  import reader2

import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm

r = reader2()

my_cache =  CACHE2('range_market.cache')

class MY_RANGE:
    def __init__(self):
        self.dfs = {}

    def get_file(self, ccy, tf):
        if not (ccy, tf) in self.dfs:
            self.dfs[(ccy, tf)]  = r.get_file_tf(ccy, tf)
        return self.dfs[(ccy, tf)]

    def __fill_between_trues(self, mask: np.ndarray) -> np.ndarray:
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


    def __compute_emptiness_from_barwise_overlap(self, lows, highs, boundary_min, boundary_max):
        """
        Compute per-bar overlap with boundary band, normalize by bar size,
        then return 1 - mean(coverage).
        """
        if boundary_min >= boundary_max:
            return 1.0, None
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
            return 1.0, None
        

        support2 = self.__fill_between_trues(support_hits)
        resistance2 = self.__fill_between_trues(resistance_hits)
        boundary_range = boundary_max - boundary_min

        both = support2 & resistance2
        only_r = resistance2 & ~support2
        only_s = support2 & ~resistance2
        coverages = np.zeros_like(lows, dtype=np.float64)
        coverages[both] = 1.0
        coverages[only_r] = np.maximum(0, boundary_max - lows[only_r]) / boundary_range
        coverages[only_s] = np.maximum(0, highs[only_s] - boundary_min) / boundary_range

        coverages_upper = np.full_like(lows, np.nan, dtype=np.float64)
        coverages_upper[both] = boundary_max
        coverages_upper[only_r] =  boundary_max
        coverages_upper[only_s] = np.maximum(boundary_min, highs[only_s])

        coverages_lower = np.full_like(lows, np.nan, dtype=np.float64)
        coverages_lower[both] = boundary_min
        coverages_lower[only_r] = np.minimum(boundary_max, lows[only_r])
        coverages_lower[only_s] = boundary_min

        mean_coverage = np.mean(coverages)
        emptiness = 1.0 - mean_coverage
        return emptiness,(coverages_upper, coverages_lower)

    @my_cache
    def get_band(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False):
        """
        Find widest price band [a, b] such that |emptiness - threshold| < tolerance.
        Returns: (emptiness, (a, b))
        """
        df_slice = self.get_file(ccy, tf).iloc[idx_start:idx_end]
        lows = df_slice['Low'].values
        highs = df_slice['High'].values
        prices = np.unique(np.concatenate([lows, highs]))
        
        best_band = (np.nan, np.nan)
        best_width = -np.inf
        best_emptiness = None
        best_coverage = None

        ret = []

        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                a = prices[i]
                b = prices[j]
                emptiness, coverage = self.__compute_emptiness_from_barwise_overlap(lows, highs, a, b)
                if emptiness == 1:
                    continue
                
                if abs(emptiness - threshold) <= tolerance:
                    width = b - a
                    ret.append((emptiness, a, b, width))
                    if width > best_width:
                        best_width = width
                        best_band = (a, b)
                        best_emptiness = emptiness
                        best_coverage = coverage

        return best_emptiness, best_band, best_coverage


    def view_band(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False):
        df_slice = self.get_file(ccy, tf).iloc[idx_start:idx_end]
        best_emptiness, best_band, best_coverage  = self.get_band(ccy, tf, idx_start, idx_end, threshold, tolerance, force_reset)
        if best_coverage:
            fig, axlist = mpf.plot(
                df_slice,
                type='candle',
                style='charles',
                ylabel='Price',
                returnfig=True,
                title=f"Price Band Fill | Emptiness: {best_emptiness:.3f}"
            )

            ax = axlist[0]
            graph_idx = np.arange(len(df_slice))

            # Draw the full band background
            ax.fill_between(graph_idx, min(best_band), max(best_band), color='grey', alpha=0.35, label='Band')

            # Draw the actual filled region using your computed bounds
            ax.fill_between( graph_idx, best_coverage[0], best_coverage[1], color='cyan', alpha=0.35, label='Filled Coverage')

            # Support & Resistance lines
            ax.axhline(min(best_band), color='green', linestyle='--', linewidth=1.2, label='Support')
            ax.axhline(max(best_band), color='red', linestyle='--', linewidth=1.2, label='Resistance')

            ax.legend()
            fig.tight_layout()
            plt.show()

    def get_band_by_batch(self,  ccy, tf, result=False, force_reset=False):
        df = self.get_file(ccy, tf)
        
        combinations = [
            (idx_start, idx_end, threshold / 10)
            for idx_start in range(0, len(df), 50)
            for idx_end in range(idx_start, idx_start + 500, 50)
            for threshold in range(1, 5)
        ]

        ret = []
        for idx_start, idx_end, threshold in tqdm(combinations):
            # just cache data?
            best_emptiness, best_band, _ = self.get_band(ccy, tf, idx_start, idx_end, threshold, 0.05, force_reset=force_reset)
            if result:
                ret.append([ccy, tf, idx_start, idx_end, best_emptiness, best_band[0], best_band[1]])

        if result:
            return ret
        
