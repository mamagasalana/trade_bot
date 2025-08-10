import sys, time
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import range_market as rngmkt
from src.pattern.range_market import MY_RANGE

# ---- worker globals & init (set once per process) ----
_LOWS = None
_HIGHS = None

def _init_worker(lows, highs):
    global _LOWS, _HIGHS
    _LOWS = lows
    _HIGHS = highs

def _worker_run(task):
    """
    task: (idx_start, idx_end, thresholds_tuple)
    Compute prices ONCE, then evaluate multiple thresholds to avoid duplicate work.
    """
    idx_start, idx_end, thresholds = task
    ls = _LOWS[idx_start:idx_end]
    hs = _HIGHS[idx_start:idx_end]
    prices = np.unique(np.concatenate([ls, hs])).astype(np.float64)

    out = []
    for threshold in thresholds:
        best_emptiness, best_band, _ = rngmkt.get_band(ls, hs, prices, threshold, 0.05, 0)
        out.append((idx_start, idx_end, threshold, best_emptiness, best_band))
    return out
# ------------------------------------------------------

def main():
    # if len(sys.argv) <= 2:
    #     print("Usage: script.py <ccy> <tf>")
    #     sys.exit(1)

    # ccy = sys.argv[1]
    # tf  = sys.argv[2]

    ccy  = 'AUDUSD'
    tf  = '1h'
    mr = MY_RANGE()
    df = mr.get_file(ccy, tf)
    n = len(df)

    LOWS  = df['Low'].to_numpy(dtype=np.float64)
    HIGHS = df['High'].to_numpy(dtype=np.float64)

    # Build (start,end) windows once; then group thresholds to avoid repeated unique()
    step_start, step_end, max_window = 50, 50, 500
    windows = []
    for idx_start in range(0, n, step_start):
        min_end = idx_start + step_start
        if min_end > n:
            break
        last_end = min(idx_start + max_window, n)
        for idx_end in range(min_end, last_end + 1, step_end):
            windows.append((idx_start, idx_end))

    # thresholds to test for every window
    thresholds = tuple(t / 10.0 for t in range(1, 5))

    # Pack tasks: one window -> one task -> compute all thresholds inside
    tasks = [(i0, i1, thresholds) for (i0, i1) in windows]

    if not tasks:
        print("No tasks generated.")
        return

    procs = min(cpu_count() or 2, 10)  # 10 was your prototype; cap to CPUs
    chunksize = 64  # tune: 32/64/128 depending on task size

    start = time.time()
    results_output = []
    _init_worker(LOWS, HIGHS)
    _worker_run(tasks[0])
    # IMPORTANT on Windows/macOS: under __main__ guard (this function) and initializer
    with Pool(processes=procs, initializer=_init_worker, initargs=(LOWS, HIGHS)) as pool:
        it = pool.imap_unordered(_worker_run, tasks, chunksize=chunksize)
        for batch in tqdm(it, total=len(tasks), desc="Bands (mp)"):
            # each batch is a list of 4 results for the thresholds
            results_output.extend(batch)

    elapsed = time.time() - start
    print(f"Process took: {elapsed:.2f}s")

    # results_output entries look like:
    # (idx_start, idx_end, threshold, best_emptiness, best_band)
    # best_band is a (a,b) pair from your C extension
    # do whatever you need next (save, aggregate, etc.)

if __name__ == "__main__":
    main()
