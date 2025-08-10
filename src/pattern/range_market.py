
import numpy as np
import pandas as pd
from src.csv.cache import CACHE2
from src.csv.reader  import reader2

import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from moviepy import ImageSequenceClip
import range_market as rngmkt
from concurrent.futures import ThreadPoolExecutor


r = reader2()

my_cache =  CACHE2('range_market.cache')

class MY_RANGE:
    def __init__(self, max_workers=None):
        self.dfs = {}
        self.VERBOSE = False
        if max_workers is None:
            self.max_workers = os.cpu_count()-3
        else:
            self.max_workers = min(os.cpu_count()-3, max_workers)


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
    def get_band_c(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False):
        df_slice = self.get_file(ccy, tf).iloc[idx_start:idx_end]
        lows  = np.asarray(df_slice['Low'],  dtype=np.float64)
        highs = np.asarray(df_slice['High'], dtype=np.float64)
        prices = np.unique(np.concatenate([lows, highs])).astype(np.float64)

        return rngmkt.get_band(lows, highs, prices, threshold, tolerance, int(self.VERBOSE))

    @my_cache
    def get_band_c_threadsafe(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False):
        df_slice = self.get_file(ccy, tf).iloc[idx_start:idx_end]
        lows  = np.asarray(df_slice['Low'],  dtype=np.float64)
        highs = np.asarray(df_slice['High'], dtype=np.float64)
        prices = np.unique(np.concatenate([lows, highs])).astype(np.float64)

        return rngmkt.get_band_threadsafe(lows, highs, prices, threshold, tolerance, int(self.VERBOSE))
    
    # @my_cache
    # def get_band(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False):
    #     """
    #     Find widest price band [a, b] such that |emptiness - threshold| < tolerance.
    #     Returns: (emptiness, (a, b))
    #     """
    #     df_slice = self.get_file(ccy, tf).iloc[idx_start:idx_end]
    #     lows = df_slice['Low'].values
    #     highs = df_slice['High'].values
    #     prices = np.unique(np.concatenate([lows, highs]))
        
    #     best_band = (np.nan, np.nan)
    #     best_width = -np.inf
    #     best_emptiness = None
    #     best_coverage = None

    #     ret = []

    #     for i in range(len(prices)):
    #         for j in range(i + 1, len(prices)):
    #             a = prices[i]
    #             b = prices[j]
    #             emptiness, coverage = self.__compute_emptiness_from_barwise_overlap(lows, highs, a, b)
    #             if emptiness == 1:
    #                 continue
    #             if self.VERBOSE:
    #                 print('%.5f, %.5f, %.5f' % (emptiness, a, b))

    #             if abs(emptiness - threshold) <= tolerance + 1e-12:
    #                 width = b - a
    #                 ret.append((emptiness, a, b, width))
    #                 if width > best_width:
    #                     best_width = width
    #                     best_band = (a, b)
    #                     best_emptiness = emptiness
    #                     best_coverage = coverage

    #     return best_emptiness, best_band, best_coverage

    def view_video(self, ccy, tf, interval, threshold=0.15, tolerance=0.005):
        os.makedirs("frames", exist_ok=True)
        df = self.get_file(ccy, tf)
        for idx_start in tqdm(range(0, len(df), 1)):
            spath = 'frames/%s.png' % idx_start
            self.view_band(ccy=ccy, tf=tf, idx_start=idx_start, idx_end=idx_start+interval, threshold=threshold, tolerance=tolerance, save_path=spath)
    
        frame_paths = sorted([f"frames/{f}" for f in os.listdir("frames") if f.endswith(".png")])
        clip = ImageSequenceClip(frame_paths, fps=10)
        clip.write_videofile("%s_%s_%s_%s.mp4" % (ccy, tf, interval, threshold), codec='libx264', fps=10)


    def view_band(self, ccy, tf, idx_start, idx_end, threshold=0.15, tolerance=0.005, force_reset=False, save_path=None):
        sz= idx_end - idx_start
        df = self.get_file(ccy, tf)
        df_slice_wide = df.iloc[max(0, idx_start-sz):min(df.shape[0], idx_end+sz*4)]
        df_slice = df.iloc[idx_start:idx_end]
        best_emptiness, best_band, best_coverage  = self.get_band_c(ccy, tf, idx_start, idx_end, threshold, tolerance, force_reset)
        
        fig, axlist = mpf.plot(
            df_slice_wide,
            type='candle',
            style='charles',
            ylabel='Price',
            returnfig=True,
            title=f"Price Band Fill",
            figsize=(14, 6))
        ax = axlist[0]
        ax.plot([], [], label=' ')

        if best_coverage:
            
            graph_idx = np.arange(len(df_slice)) + min(idx_start, sz)

            # Draw the full band background
            ax.fill_between(graph_idx, min(best_band), max(best_band), color='grey', alpha=0.35, label=f'Emptiness: {best_emptiness:.3f}')

            # Draw the actual filled region using your computed bounds
            ax.fill_between( graph_idx, best_coverage[0], best_coverage[1], color='cyan', alpha=0.35, label='Filled Coverage')

            # Support & Resistance lines
            ax.axhline(min(best_band), color='green', linestyle='--', linewidth=1.2, label='Support')
            ax.axhline(max(best_band), color='red', linestyle='--', linewidth=1.2, label='Resistance')

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    # def get_band_by_batch(self,  ccy, tf, result=False, force_reset=False):
    #     df = self.get_file(ccy, tf)
        
    #     combinations = [
    #         (idx_start, idx_end, threshold / 10)
    #         for idx_start in range(0, len(df), 50)
    #         for idx_end in range(idx_start+50, idx_start + 500, 50)
    #         for threshold in range(1, 5)
    #     ]

    #     ret = []
    #     for idx_start, idx_end, threshold in tqdm(combinations):
    #         # just cache data?
    #         best_emptiness, best_band, _ = self.get_band(ccy, tf, idx_start, idx_end, threshold, 0.05, force_reset=force_reset)
    #         if result:
    #             ret.append([ccy, tf, idx_start, idx_end, best_emptiness, best_band[0], best_band[1]])

    #     if result:
    #         return ret
        

    def get_band_by_batch_c(self,  ccy, tf, result=False, force_reset=False):
        df = self.get_file(ccy, tf)
        
        combinations = [
            (idx_start, idx_end, threshold / 10)
            for idx_start in range(0, len(df), 50)
            for idx_end in range(idx_start+50, idx_start + 500, 50)
            for threshold in range(1, 5)
        ]

        ret = []
        for idx_start, idx_end, threshold in tqdm(combinations):
            # just cache data?
            best_emptiness, best_band, _ = self.get_band_c(ccy, tf, idx_start, idx_end, threshold, 0.05, force_reset=force_reset)
            if result:
                ret.append([ccy, tf, idx_start, idx_end, best_emptiness, best_band[0], best_band[1]])

        if result:
            return ret
        
    def get_band_by_batch_c_threadsafe(self, ccy, tf, result=False, force_reset=False):
        df = self.get_file(ccy, tf)

        combinations = [
            (idx_start, idx_end, threshold / 10)
            for idx_start in range(0, len(df), 1)
            for idx_end in range(idx_start+50, idx_start + 500, 50)
            for threshold in range(1, 5)
        ]

        def process_combination(combo):
            idx_start, idx_end, threshold = combo
            best_emptiness, best_band, _ = self.get_band_c_threadsafe(
                ccy, tf, idx_start, idx_end, threshold, 0.05, force_reset=force_reset
            )
            if result:
                return [ccy, tf, idx_start, idx_end, best_emptiness, best_band[0], best_band[1]]

        ret = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for out in tqdm(executor.map(process_combination, combinations), total=len(combinations)):
                if out:
                    ret.append(out)

        if result:
            return ret
                

if __name__ == '__main__':
    import sys

    mr = MY_RANGE()
    # a,b,c = mr.get_band('AUDUSD', '1h', 11700, 11800, threshold=0.1, tolerance=0.05, force_reset=True)

    mr.VERBOSE= True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '0':
            a,b,c = mr.get_band('AUDUSD', '1h', 65400, 65800, threshold=0.4, tolerance=0.05, force_reset=True)
        else:
            a,b,c = mr.get_band_c('AUDUSD', '1h', 65400, 65800, threshold=0.4, tolerance=0.05, force_reset=True)
