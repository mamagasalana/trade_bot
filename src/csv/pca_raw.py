import tqdm
import itertools
import numpy as np
import multiprocessing as mp
import pandas as pd
from src.pattern.vecm import VECM
from tqdm.contrib.concurrent import process_map  # or thread_map
import gc
import os
v = VECM()
data = v.get_data()

# Define transformation functions
def compute_product(args):
    col1, col2 = args
    return pd.Series(data[col1] * data[col2], name=f'{col1}_{col2}_product')

def compute_sum(args):
    col1, col2 = args
    return pd.Series(data[col1] + data[col2], name=f'{col1}_{col2}_sum')

def compute_difference(args):
    col1, col2 = args
    return pd.Series(data[col1] - data[col2], name=f'{col1}_{col2}_difference')

def compute_ratio(args):
    col1, col2 = args
    return pd.Series(data[col1] / data[col2], name=f'{col1}_{col2}_ratio')
    

columns = data.columns.tolist()  # Adjust list as needed to include only relevant columns
#create derivatives of variables before applying PCA,

epsilon = 1e-10
dflist= []
for col in columns:
    dflist.append( np.log(data[col] + epsilon).rename(f'log_{col}'))

tasks = list(itertools.combinations(columns, 2))
task_len = 2
chunksize = len(tasks)//task_len +1

outfiles=  []
foos = [compute_difference, compute_ratio]
filenames = ['differences', 'ratio']
for foo, fname in zip(foos, filenames):
    for i in range(task_len):
        fname2 = f'files/pca/{fname}{i}.parquet'
        outfiles.append(fname2)
        if not os.path.exists(fname2):
            ret = []
            ret = process_map(foo, tasks[chunksize*(i):chunksize*(i+1)], max_workers=mp.cpu_count())
            df = pd.concat(ret, axis=1)
            df.to_parquet(fname2)
            gc.collect()  


for f in outfiles:
    dflist.append(pd.read_parquet(f))

dffinal = pd.concat(dflist, axis=1)
dffinal.to_parquet(f'files/pca/pca_raw.parquet')
print('debug')